from config import api_name, csv_arch, secrets
from generative import anticipate_cost, get_intent
import logging
import numpy as np
import openai
import pandas as pd

# for debug.
_in_path = '../data/02_intents.csv'
out_cost = '03_vector-cost_pre.csv'
_out_path_cost = '../data/' + out_cost
_out_path_vector = '../data/04_vector.npy'
_out_path_df = '../data/05_vectorized.csv'

# for this.
_use_model = api_name.model_emb3l


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


def main(input_path=_in_path, output_np=_out_path_vector, out_df=_out_path_df):
    # Send by data.
    df = pd.read_csv(input_path)
    _concat_intent(df, df[csv_arch.col_intent])
    _pre_anticipate(df, _out_path_cost)

    logging.info(f"Request API.")
    response = openai.embeddings.create(
        model=_use_model,
        input=df[csv_arch.col_vec_org]
    )
    logging.info(f"Received API response.")

    embeddings = [data.embedding for data in response.data]
    embeddings = np.array(embeddings)

    np.save(output_np, embeddings)
    logging.info(f"Vectors saved to [{output_np}] .")

    df.to_csv(out_df, index=True, index_label=csv_arch.col_df_index)
    logging.info(f"DataFrame saved to [{out_df}].")


if __name__ == "__main__":
    # Anticipate Cost.
    # df = pd.read_csv(_in_path)
    # _concat_intent(df)
    # _pre_anticipate(df, _out_path_cost)

    # Debug.
    secrets.set_api_key('../config/secrets.json')
    main()
