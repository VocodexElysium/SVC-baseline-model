import numpy as np
import math

from utils import *

TRANSCRIPTION_PATH = get_config("default_transcription_args")["transcription_path"]
FRAME_SIZE = get_config("default_transcription_args")["frame_size"]
PPG_OUTPUT = get_config("default_transcription_args")["PPG_output"]

Lookup_Table_Phoneme = {
    'SP': 0,
    'AP': 1,
    'a': 2,
    'ai': 3,
    'an': 4,
    'ang': 5,
    'ao': 6,
    'b': 7,
    'c': 8,
    'ch': 9,
    'd': 10,
    'e': 11,
    'ei': 12,
    'en': 13,
    'eng': 14,
    'er': 15,
    'f': 16,
    'g': 17,
    'h': 18,
    'i': 19,
    'ia': 20,
    'ian': 21,
    'iang': 22,
    'iao': 23,
    'ie': 24,
    'in': 25,
    'ing': 26,
    'iong': 27,
    'iu': 28,
    'j': 29,
    'k': 30,
    'l': 31,
    'm': 32,
    'n': 33,
    'ng': 34,
    'o': 35,
    'ong': 36,
    'ou': 37,
    'p': 38,
    'q': 39,
    'r': 40,
    's': 41,
    'sh': 42,
    't': 43,
    'u': 44,
    'ua': 45,
    'uai': 46,
    'uan': 47,
    'uang': 48,
    'ui': 49,
    'un': 50,
    'uo': 51,
    'v': 52,
    'van': 53,
    've': 54,
    'vn': 55,
    'w': 56,
    'x': 57,
    'y': 58,
    'z': 59,
    'zh': 60
}

if __name__ == "__main__":
    path = TRANSCRIPTION_PATH
    with open(path, "r") as f:
        for tmp_data in f.readlines():
            tmp_data = tmp_data.strip('\n')
            tmp_data = tmp_data.split('|')
            tmp_name = tmp_data[0]
            tmp_phonemes = tmp_data[2].split(" ")
            tmp_durations = tmp_data[4].split(" ")
            for i in range(len(tmp_durations)):
                tmp_durations[i] = float(tmp_durations[i])
                tmp_durations[i] = int(tmp_durations[i] / FRAME_SIZE + 0.5)
            tmp_total_frames = sum(tmp_durations)
            tmp_res = np.zeros(shape=(tmp_total_frames, 1))
            tmp_sum = 0
            for i in range(len(tmp_durations)):
                tmp_res[tmp_sum: tmp_sum + tmp_durations[i]] = Lookup_Table_Phoneme[tmp_phonemes[i]]
                tmp_sum = tmp_sum + tmp_durations[i]
            tmp_dir = f"{PPG_OUTPUT}/{tmp_name}.npy"
            np.save(tmp_dir, tmp_res, allow_pickle=False)
