from torch.utils.data import Dataset
from utils.audio_utils import *

class AudioDataset(Dataset):
    """"""
    def __init__(self, name=None, splits=None, with_gradient=False):
        self.name = name
        self.splits = splits
        self.path = pjoin(get_config("data_dir"), name)
        data = ListFiles(self.path, ordered=True)
        self.ppg = [pjoin(self.path, file) for file in data if ]