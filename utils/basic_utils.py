from pyheaven import *
import numpy as np
import torch
import torch.nn.functional as F

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

def get_mgc_padding(mgc_path=get_config("default_WORLD_vocoder_args")["opencpopbeta_mgc_output"]):
    res1 = 0
    res2 = 0
    data = ListFiles(mgc_path, ordered=True)
    files = [(pjoin(mgc_path, file), file.split('.')[0]) for file in data]
    for file in files:
        tmp = torch.load(file[0])
        res1 = max(res1, tmp.shape[0])
        res2 = tmp.shape[1]
    print(res1, res2)

def get_ppg_padding(ppg_path=get_config("default_transcription_args")["opencpop_PPG_output"]):
    res1 = 0
    res2 = 0
    data = ListFiles(ppg_path, ordered=True)
    files = [(pjoin(ppg_path, file), file.split('.')[0]) for file in data]
    for file in files:
        tmp = np.load(file[0])
        res1 = max(res1, tmp.shape[0])
        res2 = tmp.shape[1]
    print(res1, res2)

def pad_tensor(x, length):
    """Pad the given data.

    Example
        >>> test.shape
        torch.Size([823, 60])
        >>> res = pad_tensor(test, 1650)
        >>> res.shape
        torch.Size([1650, 60])
    """
    diff = length - x.shape[0]
    tmp = (0, 0, 0, diff)
    res = F.pad(x, tmp, "constant", 0)
    return res

def pad_numpy(x, length):
    diff = length - x.shape[0]
    tmp = ((0, diff), (0, 0))
    res = np.pad(x, tmp, 'constant', constant_values=(0, 0))
    return res