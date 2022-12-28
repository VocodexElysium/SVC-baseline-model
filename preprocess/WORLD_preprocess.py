import tqdm
import diffsptk

from utils import *
import numpy as np

WORLDPATH = get_config("default_WORLD_vocoder_args")["WORLD_output"]
MGCPATH = get_config("default_WORLD_vocoder_args")["mgc_output"]

def SP2mgc(x):
    tmp = diffsptk.ScalarOperation("SquareRoot")(x.float())
    tmp = diffsptk.ScalarOperation("Multiplication", 32768.0)(tmp)
    mgc = diffsptk.MelCepstralAnalysis(
        cep_order=get_config("default_sptk_args")["mcsize"],
        fft_length=get_config("default_sptk_args")["nFFTHalf"],
        alpha=get_config("default_sptk_args")["alpha"],
        n_iter=1
    )(tmp)
    return mgc 

if __name__ == "__main__":
    AudioToWORLDComps()
    path = WORLDPATH
    data = ListFiles(path, ordered=True)
    files_path = [(pjoin(path, file), file.split('.')[0]) for file in data if file.split('.')[1] == 'sp']
    for file in files_path:
        file_sp = torch.from_numpy(np.load(file[0]))
        res = SP2mgc(file_sp)
        out_path = pjoin(MGCPATH, file[1] + '.mgc.pt')
        torch.save(res, out_path)
