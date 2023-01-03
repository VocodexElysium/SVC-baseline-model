from torch.utils.data import Dataset
from utils.audio_utils import *

class ParamDataset(Dataset):
    def __init__(self, name=None):
        if name is not None:
            self.name = name
            self.ppg_path = get_config("default_transcription_args")["opencpopbeta_PPG_output"]
            self.mgc_path = get_config("default_WORLD_vocoder_args")["opencpopbeta_mgc_output"]
            ppg_data = ListFiles(self.ppg_path, ordered=True)
            mgc_data = ListFiles(self.mgc_path, ordered=True)
            self.ppg = [pjoin(self.ppg_path, file) for file in ppg_data]
            self.mgc = [pjoin(self.mgc_path, file) for file in mgc_data]
        else:
            self.name = None
            self.ppg_path = None
            self.mgc_path = None
            self.ppg = None
            self.mgc = None

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, i):
        ppg = torch.from_numpy(np.load(self.ppg[i])).int()
        mgc = torch.load(self.mgc[i])
        return ppg, mgc

    def __str__(self):
        if self.name is not None:
            res = f"{self.name}(len={len(self)})"
        else:
            res = "null"
        return res