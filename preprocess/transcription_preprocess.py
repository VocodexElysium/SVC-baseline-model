import numpy as np
import math

from utils import *

OPENCPOP_TRANSCRIPTION_PATH = get_config("default_transcription_args")["opencpop_transcription_path"]
OPENCPOPBETA_TRANSCRIPTION_PATH = get_config("default_transcription_args")["opencpopbeta_transcription_path"]
OPENCPOP_PPG_OUTPUT = get_config("default_transcription_args")["opencpop_PPG_output"]
OPENCPOPBETA_PPG_OUTPUT = get_config("default_transcription_args")["opencpopbeta_PPG_output"]
FRAME_SIZE = get_config("default_transcription_args")["frame_size"]

Lookup_Table_Phoneme = {
    'padding': 0,
    'SP': 1,
    'AP': 2,
    'a': 3,
    'ai': 4,
    'an': 5,
    'ang': 6,
    'ao': 7,
    'b': 8,
    'c': 9,
    'ch': 10,
    'd': 11,
    'e': 12,
    'ei': 13,
    'en': 14,
    'eng': 15,
    'er': 16,
    'f': 17,
    'g': 18,
    'h': 19,
    'i': 20,
    'ia': 21,
    'ian': 22,
    'iang': 23,
    'iao': 24,
    'ie': 25,
    'in': 26,
    'ing': 27,
    'iong': 28,
    'iu': 29,
    'j': 30,
    'k': 31,
    'l': 32,
    'm': 33,
    'n': 34,
    'ng': 35,
    'o': 36,
    'ong': 37,
    'ou': 38,
    'p': 39,
    'q': 40,
    'r': 41,
    's': 42,
    'sh': 43,
    't': 44,
    'u': 45,
    'ua': 46,
    'uai': 47,
    'uan': 48,
    'uang': 49,
    'ui': 50,
    'un': 51,
    'uo': 52,
    'v': 53,
    'van': 54,
    've': 55,
    'vn': 56,
    'w': 57,
    'x': 58,
    'y': 59,
    'z': 60,
    'zh': 61
}

def TranscriptionToPPG(transcription_input, PPG_output, mode):
    if mode == "opencpop":
        phoneme_duration_pos = 4
    elif mode == "opencpopbeta":
        phoneme_duration_pos = 5
    with open(transcription_input, "r") as f:
        for tmp_data in f.readlines():
            tmp_data = tmp_data.strip('\n')
            tmp_data = tmp_data.split('|')
            tmp_name = tmp_data[0]
            tmp_phonemes = tmp_data[2].split(" ")
            tmp_durations = tmp_data[phoneme_duration_pos].split(" ")
            for i in range(len(tmp_durations)):
                tmp_durations[i] = float(tmp_durations[i])
                tmp_durations[i] = int(tmp_durations[i] / FRAME_SIZE + 0.5)
            tmp_total_frames = sum(tmp_durations)
            tmp_res = np.zeros(shape=(tmp_total_frames, 1))
            tmp_sum = 0
            for i in range(len(tmp_durations)):
                tmp_res[tmp_sum: tmp_sum + tmp_durations[i]] = Lookup_Table_Phoneme[tmp_phonemes[i]]
                tmp_sum = tmp_sum + tmp_durations[i]
            tmp_dir = f"{PPG_output}/{tmp_name}.npy"
            tmp_res = pad_numpy(tmp_res, 2048)
            np.save(tmp_dir, tmp_res, allow_pickle=False)

if __name__ == "__main__":
    TranscriptionToPPG(OPENCPOP_TRANSCRIPTION_PATH, OPENCPOP_PPG_OUTPUT, "opencpop")
    TranscriptionToPPG(OPENCPOPBETA_TRANSCRIPTION_PATH, OPENCPOPBETA_PPG_OUTPUT, "opencpopbeta")
    
