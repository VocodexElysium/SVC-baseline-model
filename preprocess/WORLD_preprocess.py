import tqdm
import diffsptk

from utils import *
import numpy as np

OPENCPOP_INPUT = get_config("default_WORLD_vocoder_args")["opencpop_audio_input"]
OPENCPOPBETA_INPUT = get_config("default_WORLD_vocoder_args")["opencpopbeta_audio_input"]
OPENCPOP_WORLDPATH = get_config("default_WORLD_vocoder_args")["opencpop_WORLD_output"]
OPENCPOPBETA_WORLDPATH = get_config("default_WORLD_vocoder_args")["opencpopbeta_WORLD_output"]
OPENCPOP_MGCPATH = get_config("default_WORLD_vocoder_args")["opencpop_mgc_output"]
OPENCPOPBETA_MGCPATH = get_config("default_WORLD_vocoder_args")["opencpopbeta_mgc_output"]

def AudioToWORLDComps(audio_input, WORLD_output):
    """Convert raw audio into WORLD vocoder components."""
    wav_file_list = glob2.glob(f"{audio_input}/**/*.wav")
    print(f"Globbing {len(wav_file_list)} wav files.")
    os.makedirs(WORLD_output, exist_ok=True)
    for wav_file in tqdm(wav_file_list):
        audio, fs = sf.read(wav_file, always_2d=False)
        # print(f"this is {fs}")
        if fs != SAMPLE_RATE:
            # print(fs, SAMPLE_RATE)
            audio = librosa.resample(audio, orig_sr=fs, target_sr=SAMPLE_RATE)
            fs = SAMPLE_RATE
        # print(fs)
        _f0, t = pw.dio(audio, fs)
        f0 = pw.stonemask(audio, _f0, t, fs)
        sp = pw.cheaptrick(audio, f0, t, fs)
        ap = pw.d4c(audio, f0, t, fs)
        fid = os.path.basename(wav_file).split(".")[0]
        bnf_fname_f0 = f"{WORLD_output}/{fid}.f0.npy"
        bnf_fname_sp = f"{WORLD_output}/{fid}.sp.npy"
        bnf_fname_ap = f"{WORLD_output}/{fid}.ap.npy"
        np.save(bnf_fname_f0, f0, allow_pickle=False)
        np.save(bnf_fname_sp, sp, allow_pickle=False)
        np.save(bnf_fname_ap, ap, allow_pickle=False)

def WORLDCompsTomgc(WORLD_input, mgc_output):
    data = ListFiles(WORLD_input, ordered=True)
    files_path = [(pjoin(WORLD_input, file), file.split('.')[0]) for file in data if file.split('.')[1] == 'sp']
    for file in files_path:
        file_sp = torch.from_numpy(np.load(file[0]))
        res = SP2mgc(file_sp)
        out_path = pjoin(mgc_output, file[1] + '.mgc.pt')
        torch.save(res, out_path)

if __name__ == "__main__":
    AudioToWORLDComps(OPENCPOP_INPUT, OPENCPOP_WORLDPATH)
    AudioToWORLDComps(OPENCPOPBETA_INPUT, OPENCPOPBETA_WORLDPATH)
    WORLDCompsTomgc(OPENCPOP_WORLDPATH, OPENCPOP_MGCPATH)
    WORLDCompsTomgc(OPENCPOPBETA_WORLDPATH, OPENCPOPBETA_MGCPATH)
