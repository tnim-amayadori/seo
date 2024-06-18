from config import api_name
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


def calculate_in_cost(model, target) -> tuple:
    token_count = _count_tokens(target, model)
    token_count, cost_usd, cost_jy = _calculate_cost(token_count, _model_in_cst, model)
    return token_count, cost_usd, cost_jy


def calculate_out_cost(model, token_count: int = 0, response: str = None) -> tuple:
    if response:
        token_count = _count_tokens(response, model)
    else:
        if token_count == 0:
            msg = "Error： One of [token_count] or [response] is requirement"
            msg += " on calling [calculate_out_cost()] on [anticipate_cost.py]."
            print(msg)
            return 0, 0, 0

    token_count, cost_usd, cost_jy = _calculate_cost(token_count, _model_out_cst, model)
    return token_count, cost_usd, cost_jy


def print_cost(total_usd, total_jy, pre_msg=""):
    msg = pre_msg + "\n" + f"The anticipated cost is [{total_usd:.3f}]USD=[{total_jy:.1f}]JPY."
    msg += f"\nThe exchange rate is 1$ = [{_d2j}] yen."
    print(msg)
