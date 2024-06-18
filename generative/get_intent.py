from config import api_name, csv_arch, secrets
from generative import anticipate_cost
import openai
import pandas as pd

# for debug.
_in_path = '../data/input_sample.csv'
_out_path_cost = '../data/01_intents-cost_pre.csv'
_out_path_intent = '../data/02_intents.csv'

# for this.
_role = "You are an assistant that helps understand user search intent."
_max_tokens = 300
_prompt = "以下の検索ワードを入力したユーザーの意図を説明してください。"
_prompt += "一番可能性が高いと推測されるものを一つだけ、日本語で{max_str}文字以内で説明してください。"
_prompt += ":\n検索ワード: {term}\n意図:"
_use_model = api_name.model_4o


def _init_cost_col(df: pd.DataFrame):
    df[csv_arch.col_int_in_token] = None
    df[csv_arch.col_int_in_usd] = None
    df[csv_arch.col_int_in_jpy] = None
    df[csv_arch.col_int_out_token] = None
    df[csv_arch.col_int_out_usd] = None
    df[csv_arch.col_int_out_jpy] = None


def _anticipate_input(word, df: pd.DataFrame, i, total_usd, total_jy):
    prompt = _prompt.format(term=word, max_str=_max_tokens)
    all_prompt = _role + prompt
    token_n, usd, jy = anticipate_cost.calculate_in_cost(_use_model, all_prompt)
    df.loc[i, csv_arch.col_int_in_token] = token_n
    df.loc[i, csv_arch.col_int_in_usd] = round(usd, 5)
    df.loc[i, csv_arch.col_int_in_jpy] = round(jy, 3)
    total_usd += usd
    total_jy += jy
    return total_usd, total_jy, prompt


def _anticipate_output(df: pd.DataFrame, i, total_usd, total_jy, response: str = None):
    if response:
        token_n, usd, jy = anticipate_cost.calculate_out_cost(_use_model, response=response)
    else:
        token_n, usd, jy = anticipate_cost.calculate_out_cost(_use_model, token_count=_max_tokens)

    df.loc[i, csv_arch.col_int_out_token] = token_n
    df.loc[i, csv_arch.col_int_out_usd] = round(usd, 5)
    df.loc[i, csv_arch.col_int_out_jpy] = round(jy, 3)
    total_usd += usd
    total_jy += jy
    return total_usd, total_jy


def pre_anticipate(input_path=_in_path, output_path=_out_path_cost):
    df = pd.read_csv(input_path)
    target_words = df[csv_arch.col_target]
    _init_cost_col(df)
    i = 0
    total_usd = 0
    total_jy = 0
    for term in target_words:
        # Send cost.
        total_usd, total_jy, _ = _anticipate_input(term, df, i, total_usd, total_jy)

        # Return cost.
        total_usd, total_jy = _anticipate_output(df, i, total_usd, total_jy)

        i += 1

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Get Intent[pre]")

    df.to_csv(output_path)
    print(f"Intents saved to [{output_path}].")


def main(input_path=_in_path, output_path=_out_path_intent):
    # Send by data.
    df = pd.read_csv(input_path)
    df[csv_arch.col_intent] = None
    _init_cost_col(df)

    total_usd = 0
    total_jy = 0

    i = 0
    for term in df[csv_arch.col_target]:
        # Anticipate cost for input.
        total_usd, total_jy, prompt = _anticipate_input(term, df, i, total_usd, total_jy)

        # Send OpenAI.
        response = openai.chat.completions.create(
            model=_use_model,
            messages=[
                {"role": "system", "content": _role},
                {"role": "user", "content": prompt}
            ],
            max_tokens=_max_tokens
        )
        print(f"Received API response. i=[{i}].")

        # Set response.
        intent = response.choices[0].message.content
        df.loc[i, csv_arch.col_intent] = intent

        # Calculate cost for outputs.
        total_usd, total_jy = _anticipate_output(df, i, total_usd, total_jy, response=intent)

        i += 1

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Get Intent[real]")

    df.to_csv(output_path)
    print(f"Intents saved to [{output_path}].")


if __name__ == "__main__":
    # Anticipate Cost.
    # pre_anticipate()

    # Debug.
    secrets.set_api_key('../config/secrets.json')
    main()
