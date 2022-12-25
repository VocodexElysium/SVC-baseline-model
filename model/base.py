import utils
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, name=None):
        self.name = name
        self.datapath = 