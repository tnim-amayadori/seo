from config import api_name, csv_arch, secrets
from generative import anticipate_cost
import numpy as np
import openai
import pandas as pd
from sklearn.mixture import GaussianMixture

# for debug.
_in_np = '../data/04_vector.npy'
_in_df = '../data/05_vectorized.csv'
_out_cost = '../data/06_cluster-cost.csv'
_out_cluster = '../data/07_cluster.csv'
_out_final = '../data/categorize_output.csv'
_in_original = '../data/categorize_input.csv'

# for this.
_run_auto_name = True
_use_model = api_name.model_4o
_role = "You are an assistant that helps understand user search intent."
_max_tokens = 20
_prompt = "以下の検索ワードと意図に対して適切なラベルを生成してください。"
_prompt += "日本語で{max_str}文字以内で説明してください。"
_prompt += ":\n検索ワード: {search}\n意図:{intent}"

# 各クラスターのデータ数目安.
_min_data = 5
_max_data = 50


def _prepare_data(in_np=_in_np, in_df=_in_df):
    # Read data.
    df = pd.read_csv(in_df)
    embeddings = np.load(in_np)

    # DataFrameの初期化.
    df = anticipate_cost.remove_column(df, csv_arch.col_category)
    df = anticipate_cost.remove_column(df, csv_arch.col_c_name)
    df[csv_arch.col_category] = None
    return df, embeddings


def _gmm_aic(embeddings, df_words):
    data_count = df_words.shape[0]
    # データ数に対してクラスター毎の目安のデータ数で割った数のクラスターを上限・下限クラスター数として設定.
    max_cluster = data_count // _min_data + 1
    print(f"max_cluster[{max_cluster}] = data_count[{data_count}] // min_data[{_min_data}] + 1")

    mincluster = data_count // _max_data
    print(f"mincluster[{mincluster}] = data_count[{data_count}] // max_data[{_max_data}]")

    # 最適なクラスター数をAICにより決定する.
    min_aic = float('inf')
    opt_gmm = None
    opt_cluster_n = 0
    for cluster_n in range(mincluster, max_cluster + 1):
        gmm = GaussianMixture(n_components=cluster_n, random_state=42)
        gmm.fit(embeddings)
        aic = gmm.aic(embeddings)

        # AICが小さければ更新.
        if aic > min_aic:
            continue
        print(f"Update optimal. cluster=[{cluster_n}]")
        min_aic = aic
        opt_gmm = gmm
        opt_cluster_n = cluster_n

    # 最適なクラスタ数でのGMMでクラスタリングを実行.
    if opt_gmm is None:
        print("Unexpected Error. Debug and search [cluster.py].")
        return None, opt_cluster_n

    df_words[csv_arch.col_category] = opt_gmm.predict(embeddings)

    # カテゴライズ名を自動でつけるための平均ベクトルを保存しておく.
    if not _run_auto_name:
        return None, opt_cluster_n
    cluster_centers = opt_gmm.means_
    return cluster_centers, opt_cluster_n


def _init_df_cost(cluster_count) -> pd.DataFrame:
    col_names = [csv_arch.col_c_name]
    df_cost = pd.DataFrame(np.nan, index=range(cluster_count), columns=col_names)
    return df_cost


def _prepare_send(embeddings, df_words, df_cost, cluster_centers, cluster_i, total_usd, total_jy):
    # クラスター内の全ベクトルと平均ベクトル（中心）を取得.
    cluster_points = embeddings[df_words[csv_arch.col_category] == cluster_i]
    cluster_center = cluster_centers[cluster_i]

    # 中心からの距離を取得.
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    closest_idx = np.argmin(distances)

    # 最近傍のデータの検索ワードと意図を取得する.
    center_df = df_words[df_words[csv_arch.col_category] == cluster_i]
    search_word = center_df.iloc[closest_idx][csv_arch.col_target]
    intent = center_df.iloc[closest_idx][csv_arch.col_intent]

    # プロンプト生成.
    prompt = _prompt.format(max_str=_max_tokens, search=search_word, intent=intent)

    # 利用料金計算（送信分）.
    anticipate_cost.init_cost_col_both(df_cost)

    all_prompt = _role + prompt
    total_usd, total_jy = anticipate_cost.calculate_cost(df_cost, cluster_i, total_usd, total_jy, _use_model,
                                                         send_word=all_prompt)
    return total_usd, total_jy, prompt


def _anticipate_output(df_cost: pd.DataFrame, cluster_i, total_usd, total_jy, response: str = None):
    if response:
        total_usd, total_jy = anticipate_cost.calculate_cost(df_cost, cluster_i, total_usd, total_jy, _use_model,
                                                             response=response)
    else:
        total_usd, total_jy = anticipate_cost.calculate_cost(df_cost, cluster_i, total_usd, total_jy, _use_model,
                                                             token_count=_max_tokens)
    return total_usd, total_jy


def _auto_name(embeddings, df_words, cluster_centers, cluster_count, out_cost):
    # 利用料金計算用の変数初期化.
    df_cost = _init_df_cost(cluster_count)
    total_usd = 0
    total_jy = 0

    # クラスター毎に平均ベクトル(＝クラスターの中心最近傍)の検索ワードと意図を元にラベルを生成する.
    for cluster_i in range(cluster_count):
        total_usd, total_jy, prompt = _prepare_send(embeddings, df_words, df_cost, cluster_centers, cluster_i,
                                                    total_usd, total_jy)

        response = openai.chat.completions.create(
            model=_use_model,
            messages=[
                {"role": "system", "content": _role},
                {"role": "user", "content": prompt}
            ],
            max_tokens=_max_tokens
        )

        # Set response.
        label = response.choices[0].message.content
        df_words.loc[df_words[csv_arch.col_category] == cluster_i, csv_arch.col_c_name] = label
        print(f"Received API response. cluster_i=[{cluster_i}], label=[{label}]")

        # 送信結果を元に利用料金算出.
        df_cost.at[cluster_i, csv_arch.col_c_name] = label
        total_usd, total_jy = _anticipate_output(df_cost, cluster_i, total_usd, total_jy, response=label)

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Auto Name[real]")

    df_cost.to_csv(out_cost, index=True, index_label=csv_arch.col_category)
    print(f"Costs saved to [{out_cost}].")


def pre_anticipate(in_np=_in_np, in_df=_in_df, out_cost=_out_cost):
    # 自動でカテゴリー名決定しない場合はAPI利用料金なし.
    if not _run_auto_name:
        print("Clustering needs no cost.")
        return

    # クラスター数を決定してAPI利用料金を算出する.
    df_words, embeddings = _prepare_data(in_np, in_df)

    # 最適なクラスター数をAICにより決定する.
    cluster_centers, cluster_count = _gmm_aic(embeddings, df_words)

    # 利用料金計算用の変数初期化.
    df_cost = _init_df_cost(cluster_count)
    total_usd = 0
    total_jy = 0
    for cluster_i in range(cluster_count):
        # Send cost.
        total_usd, total_jy, _ = _prepare_send(embeddings, df_words, df_cost, cluster_centers, cluster_i, total_usd,
                                               total_jy)

        # Return cost.
        total_usd, total_jy = _anticipate_output(df_cost, cluster_i, total_usd, total_jy)

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Auto Name[pre]")

    df_cost.to_csv(out_cost, index=True, index_label=csv_arch.col_category)
    print(f"Costs saved to [{out_cost}].")


def main(in_np=_in_np, in_df=_in_df, out_cost=_out_cost, out_cluster=_out_cluster, out_final=_out_final,
         in_original=_in_original):
    df_words, embeddings = _prepare_data(in_np, in_df)

    # 最適なクラスター数をAICにより決定する.
    cluster_centers, cluster_count = _gmm_aic(embeddings, df_words)

    # OpenAIのAPIを利用して自動でカテゴリ名をつける.
    if cluster_centers:
        df_words[csv_arch.col_c_name] = None
        _auto_name(embeddings, df_words, cluster_centers, cluster_count, out_cost)

    # 中間ファイルの出力.
    df_words.to_csv(out_cluster)
    print(f"Clusters saved to [{out_cluster}].")

    # 大元のinputにcategoryを追加しただけの最終出力ファイル.
    tmp_category = df_words[csv_arch.col_category]
    if cluster_centers:
        tmp_c_name = df_words[csv_arch.col_c_name]
    else:
        tmp_c_name = None
    df = pd.read_csv(in_original)

    df = anticipate_cost.remove_column(df, csv_arch.col_category)
    df[csv_arch.col_category] = tmp_category

    if tmp_c_name:
        df = anticipate_cost.remove_column(df, csv_arch.col_c_name)
        df[csv_arch.col_category] = tmp_c_name

    df.to_csv(out_final, index=True, index_label=csv_arch.col_df_index)
    print(f'Categories saved to [{out_final}] finally.')


if __name__ == "__main__":
    # Anticipate Cost.
    pre_anticipate()

    # Debug.
    secrets.set_api_key('../config/secrets.json')
    # main()
