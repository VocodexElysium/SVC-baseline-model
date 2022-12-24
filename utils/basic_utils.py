from pyheaven import *

DEFAULT_CONFIG_PATH = "config.json"

def get_config(key, default=None, config_path=DEFAULT_CONFIG_PATH):
    """Return the config of a given key.
    
    Example
        >>> get_config("default_torchaudio_args")['n_fft']
            2048
    """
    config = LoadJson(config_path)
    if key in config:
        return config[key]
    else:
        return default

def set_config(key, value, override=True, config_path=DEFAULT_CONFIG_PATH):
    """Set the config.
    
    Example
        >>> set_config("data_dir", "/data")
        >>> get_config("data_dir")
        '/data'
    """
    config = LoadJson(config_path)
    if override == True or (key not in config):
        config[key] = value
    SaveJson(config, config_path, indent=4)