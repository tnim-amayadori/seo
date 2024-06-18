import json
import openai

_secrets_path = 'config/secrets.json'
_secrets: dict = {}


# JSONファイルから秘密情報を読み込む
def load_secrets():
    with open(_secrets_path, 'r') as file:
        global _secrets
        _secrets = json.load(file)


def set_api_key():
    load_secrets()
    openai.api_key = _secrets['openai_api_key']
