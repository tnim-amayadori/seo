import json
import openai

_secrets_path = 'config/secrets.json'
_secrets: dict = {}


# JSONファイルから秘密情報を読み込む
def load_secrets(in_path):
    global _secrets, _secrets_path
    _secrets_path = in_path
    with open(_secrets_path, 'r') as file:
        _secrets = json.load(file)


def set_api_key(in_path: str = _secrets_path):
    load_secrets(in_path)
    openai.api_key = _secrets['openai_api_key']
