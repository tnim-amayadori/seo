from config import api_name, csv_arch, secrets
from generative import anticipate_cost, common_util, get_intent
import logging
import numpy as np
import openai
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_score

# for debug.
_path_data = '../data/'
_in_np = _path_data + '13_vector.npy'
_in_df = _path_data + '14_vectorized.csv'
out_cost_name = 'cluster-cost'
out_cost_pre = '03_' + out_cost_name + '_pre.csv'
out_cost_name = '15' + out_cost_name + '.csv'
_out_cost_pre = _path_data + out_cost_pre
_out_cost_run = _path_data + out_cost_name
out_cluster_name = '16_cluster.csv'
_out_cluster = _path_data + out_cluster_name
out_final_name = 'categorize_output.csv'
_out_final = _path_data + out_final_name
_in_original = _path_data + 'input_sample.csv'

# for this.
_run_auto_name = True
_use_model = api_name.model_4o
_role = "You are an assistant that helps understand user search intent."
_max_tokens = 20
_prompt = "以下の検索ワードと意図に対して適切なラベルを生成してください。"
_prompt += "日本語で{max_str}文字以内で説明してください。ラベル名の文字列のみを返却してください。"
_prompt += ":\n検索ワード: {search}\n意図:{intent}\nラベル: "

# 各クラスターのデータ数目安.
_min_data = 20
_data_count_minus = 2

_max_data = 100
_min_cluster = 2


def _prepare_data(in_np=_in_np, in_df=_in_df):
    # Read data.
    df = pd.read_csv(in_df)
    embeddings = np.load(in_np)

    # DataFrameの初期化.
    df = anticipate_cost.remove_column(df, csv_arch.col_category)
    df = anticipate_cost.remove_column(df, csv_arch.col_c_name)
    df[csv_arch.col_category] = None
    return df, embeddings


def _cluster_range(df_words):
    data_count = df_words.shape[0]
    # データ数に対してクラスター毎の目安のデータ数で割った数のクラスターを上限・下限クラスター数として設定.
    max_cluster = data_count // _min_data + 1
    tmp_cluster = data_count - _data_count_minus
    if max_cluster >= tmp_cluster:
        max_cluster = tmp_cluster
        logging.info(f"max_cluster[{max_cluster}] = data_count[{data_count}] - {_data_count_minus}")
    else:
        logging.info(f"max_cluster[{max_cluster}] = data_count[{data_count}] // min_data[{_min_data}] + 1")

    mincluster = data_count // _max_data
    if mincluster < _min_cluster:
        mincluster = _min_cluster
        logging.info(f"mincluster[{mincluster}] = _min_cluster[{_min_cluster}]")
    else:
        logging.info(f"mincluster[{mincluster}] = data_count[{data_count}] // max_data[{_max_data}]")
    return mincluster, max_cluster


def _kmeans_silhouette(embeddings, df_words):
    mincluster, max_cluster = _cluster_range(df_words)

    # 最適なクラスター数をシルエットスコアにより決定する.
    best_silhouette = -1
    opt_kmeans = None
    opt_cluster_n = 0
    for cluster_n in range(mincluster, max_cluster + 1):
        kmeans = cluster.KMeans(n_clusters=cluster_n, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, labels)

        # シルエットスコアが高ければ更新.
        if silhouette_avg > best_silhouette:
            logging.info(f"Update optimal. cluster=[{cluster_n}], Silhouette Score=[{silhouette_avg}]")
            best_silhouette = silhouette_avg
            opt_kmeans = kmeans
            opt_cluster_n = cluster_n

    # 最適なクラスタ数でのK-meansでクラスタリングを実行.
    if opt_kmeans is None:
        logging.error("Unexpected Error. Debug and search [cluster.py].")
        return None, opt_cluster_n

    logging.info(f"Find optimal cluster count. cluster=[{opt_cluster_n}]")
    df_words[csv_arch.col_category] = opt_kmeans.predict(embeddings)

    # カテゴライズ名を自動でつけるための平均ベクトルを保存しておく.
    if not _run_auto_name:
        return None, opt_cluster_n
    cluster_centers = opt_kmeans.cluster_centers_
    return cluster_centers, opt_cluster_n


def _init_df_cost(cluster_count, initial_name: str = '') -> pd.DataFrame:
    df_cost = pd.DataFrame({csv_arch.col_c_name: [initial_name] * cluster_count})
    anticipate_cost.init_cost_col_both(df_cost)
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
        logging.info(f"Received API response. cluster_i=[{cluster_i}], label=[{label}]")

        # 送信結果を元に利用料金算出.
        df_cost.at[cluster_i, csv_arch.col_c_name] = label
        total_usd, total_jy = _anticipate_output(df_cost, cluster_i, total_usd, total_jy, response=label)

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Auto Name[real]")

    df_cost.to_csv(out_cost, index=True, index_label=csv_arch.col_category)
    logging.info(f"Costs saved to [{out_cost}].")


def _pre_anticipate(in_np=_in_np, in_df=_in_df, out_cost=_out_cost_pre):
    # 自動でカテゴリー名決定しない場合はAPI利用料金なし.
    if not _run_auto_name:
        logging.info("Clustering needs no cost.")
        return

    # クラスター数を決定してAPI利用料金を算出する.
    df_words, embeddings = _prepare_data(in_np, in_df)

    # 最適なクラスター数をAICにより決定する.
    cluster_centers, cluster_count = _kmeans_silhouette(embeddings, df_words)

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
    logging.info(f"Costs saved to [{out_cost}].")


def pre_anticipate(input_path, output_path):
    # 自動でカテゴリー名を決定しない場合はAPI利用料金なし.
    if not _run_auto_name:
        logging.info("Clustering needs no cost.")
        return

    df_words = pd.read_csv(input_path)

    # クラスター数をMax想定とする。
    _, max_cluster = _cluster_range(df_words)

    # 各クラスターMaxの検索ワードと意図でプロンプトを生成する想定でコスト予測.
    max_word_count = df_words[csv_arch.col_target].str.len().max()

    search_sample = anticipate_cost.get_sample_str(max_word_count)
    intent_sample = anticipate_cost.get_sample_str(get_intent.max_tokens)
    prompt = _prompt.format(max_str=_max_tokens, search=search_sample, intent=intent_sample)
    prompt = _role + prompt

    sample_category = 'OpenAIのAPIにより自動でカテゴリ名が決定されます。'
    df_cost = _init_df_cost(max_cluster, sample_category)
    total_usd = 0
    total_jy = 0

    for cluster_i in range(max_cluster):
        total_usd, total_jy = anticipate_cost.calculate_cost(df_cost, cluster_i, total_usd, total_jy, _use_model,
                                                             send_word=prompt)
        total_usd, total_jy = _anticipate_output(df_cost, cluster_i, total_usd, total_jy)

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Auto Name[pre]")
    df_cost.to_csv(output_path, index=True, index_label=csv_arch.col_category)
    logging.info(f"Costs saved to [{output_path}].")

    return total_usd, total_jy


def main(in_np=_in_np, in_df=_in_df, out_cost=_out_cost_run, out_cluster=_out_cluster, out_final=_out_final,
         in_original=_in_original):
    df_words, embeddings = _prepare_data(in_np, in_df)

    # 最適なクラスター数をAICにより決定する.
    cluster_centers, cluster_count = _kmeans_silhouette(embeddings, df_words)

    # OpenAIのAPIを利用して自動でカテゴリ名をつける.
    if cluster_centers is not None:
        df_words[csv_arch.col_c_name] = None
        _auto_name(embeddings, df_words, cluster_centers, cluster_count, out_cost)

    # 中間ファイルの出力.
    df_words.to_csv(out_cluster)
    logging.info(f"Clusters saved to [{out_cluster}].")

    # 大元のinputにcategoryを追加しただけの最終出力ファイル.
    tmp_category = df_words[csv_arch.col_category]
    if cluster_centers is not None:
        tmp_c_name = df_words[csv_arch.col_c_name]
    else:
        tmp_c_name = None
    df = pd.read_csv(in_original)

    df = anticipate_cost.remove_column(df, csv_arch.col_category)
    df[csv_arch.col_category] = tmp_category

    if tmp_c_name is not None:
        df = anticipate_cost.remove_column(df, csv_arch.col_c_name)
        df[csv_arch.col_c_name] = tmp_c_name

    df.to_csv(out_final, index=True, index_label=csv_arch.col_df_index)
    logging.info(f'Categories saved to [{out_final}] finally.')


if __name__ == "__main__":
    # Anticipate Cost.
    # _pre_anticipate()

    # Debug.
    common_util.initialize_logging(_path_data)
    secrets.set_api_key('../config/secrets.json')
    main()
