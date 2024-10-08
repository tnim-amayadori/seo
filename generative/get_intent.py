from config import api_name, csv_arch, secrets
from generative import anticipate_cost
import logging
import openai
import pandas as pd

# for debug.
_in_path = '../data/input_sample.csv'
out_cost = '01_intents-cost_pre.csv'
_out_path_cost = '../data/' + out_cost
out_intent = '11_intents.csv'
_out_path_intent = '../data/' + out_intent

# for this.
_use_model = api_name.model_4o
_role = "You are an assistant that helps understand user search intent."
max_tokens = 200
_prompt = "以下の検索ワードを入力したユーザーの意図を説明してください。またどのような属性のユーザーかを説明してください。"
_prompt += "一番可能性が高いと推測される意図とユーザー属性の組み合わせを一つだけ、日本語で{max_str}文字以内で説明してください。"
_prompt += ":\n検索ワード: {term}\n検索の意図とユーザー属性:"


def _anticipate_input(word, df: pd.DataFrame, i, total_usd, total_jy):
    prompt = _prompt.format(term=word, max_str=max_tokens)
    all_prompt = _role + prompt
    total_usd, total_jy = anticipate_cost.calculate_cost(df, i, total_usd, total_jy, _use_model, send_word=all_prompt)
    return total_usd, total_jy, prompt


def _anticipate_output(df: pd.DataFrame, i, total_usd, total_jy, response: str = None):
    if response:
        total_usd, total_jy = anticipate_cost.calculate_cost(df, i, total_usd, total_jy, _use_model, response=response)
    else:
        total_usd, total_jy = anticipate_cost.calculate_cost(df, i, total_usd, total_jy, _use_model,
                                                             token_count=max_tokens)
    return total_usd, total_jy


def pre_anticipate(input_path=_in_path, output_path=_out_path_cost):
    df = pd.read_csv(input_path)
    target_words = df[csv_arch.col_target]
    anticipate_cost.init_cost_col_both(df)
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
    logging.info(f"Intents Cost saved to [{output_path}].")

    return total_usd, total_jy


def main(input_path=_in_path, output_path=_out_path_intent):
    # Send by data.
    df = pd.read_csv(input_path)
    df[csv_arch.col_intent] = None
    anticipate_cost.init_cost_col_both(df)

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
            max_tokens=max_tokens
        )
        logging.info(f"Received API response. i=[{i}].")

        # Set response.
        intent = response.choices[0].message.content
        df.loc[i, csv_arch.col_intent] = intent

        # Calculate cost for outputs.
        total_usd, total_jy = _anticipate_output(df, i, total_usd, total_jy, response=intent)

        i += 1

    # Output cost.
    anticipate_cost.print_cost(total_usd, total_jy, pre_msg="Get Intent[real]")

    df.to_csv(output_path, index=True, index_label=csv_arch.col_df_index)
    logging.info(f"Intents saved to [{output_path}].")


if __name__ == "__main__":
    # Anticipate Cost.
    # pre_anticipate()

    # Debug.
    secrets.set_api_key('../config/secrets.json')
    main()
