from config import api_name, csv_arch, secrets
from generative import anticipate_cost
import numpy as np
import openai
import pandas as pd

# for debug.
_in_path = '../data/02_intents.csv'
_out_path_cost = '../data/03_vector-cost_pre.csv'
_out_path_vector = '../data/04_vector.npy'
_out_path_df = '../data/05_vectorized.csv'

# for this.
_use_model = api_name.model_emb3l


def _concat_intent(df: pd.DataFrame):
    # 検索ワードとユーザーの意図を連結してVector変換する.
    df[csv_arch.col_vec_org] = 'search word :\n' + df[csv_arch.col_target]
    df[csv_arch.col_vec_org] = df[csv_arch.col_vec_org] + '\nuser intent :\n' + df[csv_arch.col_intent]


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
    print(f"Vectorize Cost saved to [{output_path}].")


def pre_anticipate(input_path=_in_path, output_path=_out_path_cost):
    df = pd.read_csv(input_path)
    _concat_intent(df)
    _pre_anticipate(df, output_path)


def main(input_path=_in_path, output_np=_out_path_vector, out_df=_out_path_df):
    # Send by data.
    df = pd.read_csv(input_path)
    _concat_intent(df)
    _pre_anticipate(df, _out_path_cost)

    print(f"Request API.")
    response = openai.embeddings.create(
        model=_use_model,
        input=df[csv_arch.col_vec_org]
    )
    print(f"Received API response.")

    embeddings = response.data[0].embedding

    np.save(output_np, np.array(embeddings))
    print(f"Vectors saved to [{output_np}] .")

    df.to_csv(out_df, index=True, index_label=csv_arch.col_df_index)
    print(f"DataFrame saved to [{out_df}].")


if __name__ == "__main__":
    # Anticipate Cost.
    # pre_anticipate()

    # Debug.
    secrets.set_api_key('../config/secrets.json')
    main()
