from config import api_name, csv_arch
import pandas as pd
import tiktoken

# コスト計算のための単価.
_d2j = 160  # 為替レート、1＄当たりの日本円

# モデルごとのトークン単価
# 単価取得のAPIは提供されていないため、下記を参照してメンテナンスしてください。
# https://openai.com/api/pricing/
_model_in_cst = {
    # 100万トークン(1M tokens)当たり.
    api_name.model_4o: 5,
    api_name.model_35t: 0.5,
    api_name.model_emb3l: 0.13,
    api_name.model_emb3s: 0.02,
    api_name.model_emb2: 0.10
}

_model_out_cst = {
    # 100万トークン当たり.
    api_name.model_4o: 15,
    api_name.model_35t: 1.5
}


def _remove_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column in df.columns:
        df = df.drop(columns=[target_column])
    return df


def init_cost_col_in(df: pd.DataFrame):
    # Clear Columns.
    df = _remove_column(df, csv_arch.col_in_token)
    df = _remove_column(df, csv_arch.col_in_usd)
    df = _remove_column(df, csv_arch.col_in_jpy)
    df = _remove_column(df, csv_arch.col_out_token)
    df = _remove_column(df, csv_arch.col_out_usd)
    df = _remove_column(df, csv_arch.col_out_jpy)

    # Generate in cost.
    df[csv_arch.col_in_token] = None
    df[csv_arch.col_in_usd] = None
    df[csv_arch.col_in_jpy] = None


def init_cost_col_both(df: pd.DataFrame):
    init_cost_col_in(df)
    df[csv_arch.col_out_token] = None
    df[csv_arch.col_out_usd] = None
    df[csv_arch.col_out_jpy] = None


# トークン数の計測
def _count_tokens(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = len(encoding.encode(string))
    return token_count


# 料金の概算.
def _calculate_cost(token_count, use_def: dict, model):
    cost_per_1m = use_def.get(model, 0)
    if cost_per_1m == 0:
        print(f"Error：The model [{model}] is not defined in [{use_def}] on [anticipate_cost.py].")
        return token_count, 0, 0

    cost_usd = (token_count / 1000000) * cost_per_1m
    cost_jy = cost_usd * _d2j
    return token_count, cost_usd, cost_jy


def calculate_cost(df: pd.DataFrame, i, total_usd, total_jy, model, send_word: str = None, token_count: int = 0,
                   response: str = None) -> tuple:
    if send_word:
        token_count = _count_tokens(send_word, model)
        model_cst = _model_in_cst
        col_token = csv_arch.col_in_token
        col_usd = csv_arch.col_in_usd
        col_jpy = csv_arch.col_in_jpy

    else:
        if response:
            token_count = _count_tokens(response, model)
        else:
            if token_count == 0:
                msg = "Error： One of [send_word] or [token_count] or [response] is requirement"
                msg += " on calling [calculate_cost()] on [anticipate_cost.py]."
                print(msg)
                return total_usd, total_jy

        model_cst = _model_out_cst
        col_token = csv_arch.col_out_token
        col_usd = csv_arch.col_out_usd
        col_jpy = csv_arch.col_out_jpy

    token_n, usd, jy = _calculate_cost(token_count, model_cst, model)

    df.loc[i, col_token] = token_n
    df.loc[i, col_usd] = round(usd, 5)
    df.loc[i, col_jpy] = round(jy, 3)

    total_usd += usd
    total_jy += jy
    return total_usd, total_jy


def print_cost(total_usd, total_jy, pre_msg=""):
    msg = pre_msg + "\n" + f"The anticipated cost is [{total_usd:.3f}]USD=[{total_jy:.1f}]JPY."
    msg += f"\nThe exchange rate is 1$ = [{_d2j}] yen."
    print(msg)
