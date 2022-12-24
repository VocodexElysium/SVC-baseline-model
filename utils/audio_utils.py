import os
import torch
import glob2
import librosa

from .basic_utils import *
from .conformer_ppg_model.build_ppg_model import load_ppg_model
import pyworld as pw
import soundfile as sf
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = get_config("global_sample_rate")

def AudioToWORLDComps(
    audio_input=get_config("default_WORLD_vocoder_args")["audio_input"],
    WORLD_output=get_config("default_WORLD_vocoder_args")["WORLD_output"],
):
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
        _f0, t = pw.dio(audio, fs)
        f0 = pw.stonemask(audio, _f0, t, fs)
        sp = pw.cheaptrick(audio, f0, t, fs)
        ap = pw.d4c(audio, f0, t, fs)
        fid = os.path.basename(wav_file).split(".")[0]
        bnf_fname_f0 = f"{WORLD_output}/{fid}.WROLD_f0.npy"
        bnf_fname_sp = f"{WORLD_output}/{fid}.WROLD_sp.npy"
        bnf_fname_ap = f"{WORLD_output}/{fid}.WROLD_ap.npy"
        np.save(bnf_fname_f0, f0, allow_pickle=False)
        np.save(bnf_fname_sp, sp, allow_pickle=False)
        np.save(bnf_fname_ap, ap, allow_pickle=False)

def AudioToPPG(
    PPG_output=get_config("default_conformer_ppg_model_args")["PPG_output"],
    audio_input=get_config("default_conformer_ppg_model_args")["audio_input"],
    train_config=get_config("default_conformer_ppg_model_args")["train_config"],
    model_file=get_config("default_conformer_ppg_model_args")["model_file"],
):
    """Convert raw audio into PPGs via a pretrained ASR model."""
    device = "cuda"
    ppg_model_local = load_ppg_model(train_config, model_file, device)
    wav_file_list = glob2.glob(f"{audio_input}/**/*.wav")
    print(f"Globbing {len(wav_file_list)} wav files.")
    os.makedirs(PPG_output, exist_ok=True)
    for wav_file in tqdm(wav_file_list):
        audio, fs = sf.read(wav_file, always_2d=False)
        if fs != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=SAMPLE_RATE)
            fs = SAMPLE_RATE
        wav_tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0)
        wav_length = torch.LongTensor([audio.shape[0]]).to(device)
        with torch.no_grad():
            bnf = ppg_model_local(wav_tensor, wav_length)
        bnf_npy = bnf.squeeze(0).cpu().numpy()
        fid = os.path.basename(wav_file).split(".")[0]
        bnf_fname = f"{PPG_output}/{fid}.ling_feat.npy"
        np.save(bnf_fname, bnf_npy, allow_pickle=False)
