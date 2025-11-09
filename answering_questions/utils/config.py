# config.py
import json

_config = None

def get_config():    
    global _config
    if _config is None:        
        with open("./utils/json/config.json") as f:
            _config = json.load(f)        
    return _config

# I need to save it otherwise the per process changes will be lost
def set_config(attribute, value):
    global _config
    if _config is None:
        print("Configuration not loaded yet. Loading now...")
        get_config()    
    _config[attribute] = value
    # saving back to file
    with open("./utils/json/config.json", "w") as f:
        json.dump(_config, f, indent=4)