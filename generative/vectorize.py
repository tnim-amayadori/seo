from config import api_name, csv_arch, secrets
from generative import anticipate_cost, common_util, get_intent
import logging
import numpy as np
import openai
import pandas as pd

# for debug.
_path_data = '../data/'
_in_path = _path_data + '11_intents.csv'
out_cost_name = 'vector-cost_pre.csv'
out_cost_pre = '02_' + out_cost_name
out_cost_name = '12_' + out_cost_name
_out_cost_pre = _path_data + out_cost_pre
_out_cost_run = _path_data + out_cost_name
out_vector = '13_vector.npy'
_out_path_vector = _path_data + out_vector
out_vectorize_df = '14_vectorized.csv'
_out_path_df = _path_data + out_vectorize_df

# for this.
_use_model = api_name.model_emb3l
_batch_size = 500


def _concat_intent(df: pd.DataFrame, intent_str):
    # 検索ワードとユーザーの意図を連結してVector変換する.
    df[csv_arch.col_vec_org] = 'search word :\n' + df[csv_arch.col_target]
    df[csv_arch.col_vec_org] = df[csv_arch.col_vec_org] + '\nuser intent :\n' + intent_str


def _pre_anticipate(df: pd.DataFrame, output_path):
    anticipate_cost.init_cost_col_in(df)
    i = 0
    total_usd = 0
    total_jy = 0
    for term in df[csv_arch.col_vec_org]:
        # Send cost.
        total_usd, total_jy = anticipate_cost.calculate_cost(df, i, total_usd, total_jy, _use_model, send_word=term)

        i += 1

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Vectorize[real]")

    df.to_csv(output_path)
    logging.info(f"Vectorize Cost saved to [{output_path}].")

    return total_usd, total_jy


def pre_anticipate(input_path, output_path):
    df = pd.read_csv(input_path)

    # 前のモジュールでAPI実行してないので計算用にダミー文字列でmax想定の送信文字列を生成する.
    sample_str = anticipate_cost.get_sample_str(get_intent.max_tokens)
    _concat_intent(df, sample_str)

    # Calculate.
    total_usd, total_jy = _pre_anticipate(df, output_path)

    return total_usd, total_jy


def main(input_path=_in_path, output_np=_out_path_vector, out_df=_out_path_df, out_cost=_out_cost_run,
         batch_size=_batch_size):
    # Send by data.
    df = pd.read_csv(input_path)
    _concat_intent(df, df[csv_arch.col_intent])
    _pre_anticipate(df, out_cost)

    # データの総数を取得
    total_size = len(df[csv_arch.col_vec_org])

    # バッチごとにEmbeddingを生成し保存するリスト
    all_embeddings = []

    logging.info(f"Request API.")
    for i in range(0, total_size, batch_size):
        # データをバッチサイズに分割
        batch_data = df[csv_arch.col_vec_org][i:i + batch_size].tolist()

        # OpenAI APIを呼び出してEmbeddingを取得
        response = openai.embeddings.create(
            model=_use_model,
            input=batch_data
        )

        # 結果のembeddingを抽出
        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)

        logging.info(f"Processed batch {i // batch_size + 1}/{(total_size + batch_size - 1) // batch_size}")

    logging.info(f"Received API response.")

    embeddings = np.array(all_embeddings)

    np.save(output_np, embeddings)
    logging.info(f"Vectors saved to [{output_np}] .")

    df.to_csv(out_df, index=True, index_label=csv_arch.col_df_index)
    logging.info(f"DataFrame saved to [{out_df}].")


if __name__ == "__main__":
    # Anticipate Cost.
    # df = pd.read_csv(_in_path)
    # _concat_intent(df)
    # _pre_anticipate(df, _out_cost_pre)

    # Debug.
    common_util.initialize_logging(_path_data)
    secrets.set_api_key('../config/secrets.json')
    main()
