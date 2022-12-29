import tqdm
import textgrid
import math

from utils import *
import numpy as np

TEXTGRIDPATH = get_config("default_grid_args")["textgrids"]
FRAMESIZE = get_config("default_grid_args")["frame_size"]
PPG_OUTPUT = get_config("default_grid_args")["PPG_output"]

Lookup_Table_i = {
    0: "sentence",
    1: "character",
    2: "syllable",
    3: "pitch",
    4: "duration",
    5: "phoneme",
    6: "tuplet",
    }

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


def grid_process(tg, mode, name):
    if mode == 0:
        return
    elif mode == 1:
        return
    elif mode == 2:
        return
    elif mode == 3:
        return
    elif mode == 4:
        return
    elif mode == 5:
        res = np.zeros(shape=(1, int(math.ceil(tg.maxTime / FRAMESIZE))))
        phoneme_iter = 0
        phoneme_time = 0
        grid_iter = 0
        while grid_iter < len(tg.intervals):
            if grid_iter == len(tg.intervals) - 1:
                # print()
                res[0][phoneme_iter: -1] = Lookup_Table_Phoneme[tg[grid_iter].mark.strip()]
                break
            elif phoneme_time + FRAMESIZE<= tg[grid_iter].maxTime:
                res[0][phoneme_iter] = Lookup_Table_Phoneme[tg[grid_iter].mark.strip()]
                phoneme_iter += 1
                phoneme_time += FRAMESIZE
                continue
            else:
                if abs(phoneme_time + FRAMESIZE - tg[grid_iter].maxTime) <= 0.5000000:
                    res[0][phoneme_iter] = Lookup_Table_Phoneme[tg[grid_iter].mark.strip()]
                    phoneme_iter += 1
                    grid_iter += 1
                    phoneme_time += FRAMESIZE
                    continue
                else:
                    grid_iter += 1
                    res[0][phoneme_iter] = Lookup_Table_Phoneme[tg[grid_iter].mark.strip()]
                    phoneme_iter += 1
                    phoneme_time += FRAMESIZE
                    continue
        # for i in range(res.shape[1]):
        #     print(res[0][i], end=" ")
        res_fname = f"{PPG_OUTPUT}/{name}.npy"
        np.save(res_fname, res, allow_pickle=False)
    elif mode == 6:
        return

if __name__ == "__main__":
    path = TEXTGRIDPATH
    data = ListFiles(path, ordered=True)
    files_path = [(pjoin(path, file), file.split('.')[0]) for file in data]
    tmp = 0
    for file in files_path:
        # if file[1] != "2001":
        #     continue
        tg = textgrid.TextGrid.fromFile(file[0])
        for i in range(7):
            res = grid_process(tg[i], i, file[1])
            
