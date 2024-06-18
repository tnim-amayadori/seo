from config import api_name, csv_arch
from generative import anticipate_cost
import openai
import pandas as pd

# for debug.
_in_path = '../data/input_sample.csv'
_out_path_intent = '../data/01_intents.csv'
_out_path_cost = '../data/01_intents-cost.csv'

# for this.
_prompt = "以下の検索ワードのユーザーの意図を説明してください:\n検索ワード: {term}\n意図:"
_max_tokens = 300
_use_model = api_name.model_4o


def _pre_anticipate(input_path=_in_path, output_path=_out_path_cost):
    df = pd.read_csv(input_path)
    target_words = df[csv_arch.col_target]

    all_in_token = []
    all_in_usd = []
    all_in_jy = []
    all_out_token = []
    all_out_usd = []
    all_out_jy = []
    total_usd = 0
    total_jy = 0
    for term in target_words:
        # Send cost.
        prompt = _prompt.format(term=term)
        token_n, usd, jy = anticipate_cost.calculate_in_cost(_use_model, prompt)
        all_in_token.append(token_n)
        all_in_usd.append(usd)
        all_in_jy.append(jy)
        total_usd += usd
        total_jy += jy

        # Return cost.
        token_n, usd, jy = anticipate_cost.calculate_out_cost(_use_model, _max_tokens)
        all_out_token.append(token_n)
        all_out_usd.append(usd)
        all_out_jy.append(jy)
        total_usd += usd
        total_jy += jy

    # Output cost.
    df[csv_arch.col_int_token] = all_out_token
    df[csv_arch.col_int_in_usd] = all_out_token
    df[csv_arch.col_int_in_jpy] = all_out_token
    df[csv_arch.col_int_out_usd] = all_out_token
    df[csv_arch.col_int_out_jpy] = all_out_token

    df.to_csv(output_path)
    print(f"Intents saved to [{output_path}].")

    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Get Intent[pre]")


def _main(input_path=_in_path, output_path=_out_path_intent):
    df = pd.read_csv(input_path)
    intents = []
    for term in df[csv_arch.col_target]:
        response = openai.Completion.create(
            model=_use_model,
            prompt=_prompt.format(term=term),
            max_tokens=_max_tokens,
            n=1,
            stop=None,
            temperature=0.7
        )
        intent = response.choices[0].text.strip()
        intents.append(intent)

    df[csv_arch.col_intent] = intents
    df.to_csv(output_path)
    print(f"Intents saved to [{output_path}].")


if __name__ == "__main__":
    # CSVファイルを読み込む
    _pre_anticipate()
