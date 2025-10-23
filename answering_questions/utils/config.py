# config.py
import json

_config = None


def get_config():
    global _config
    if _config is None:
        with open("./utils/json/config.json") as f:
            _config = json.load(f)
    return _config
