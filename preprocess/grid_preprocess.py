import tqdm
import textgrid

from utils import *
import numpy as np

TEXTGRIDPATH = get_config("default_gt_ppg_ars")["textgrids"]

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


def grid_process(tg, mode):
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
        return
    elif mode == 6:
        return

if __name__ == "__main__":
    path = TEXTGRIDPATH
    data = ListFiles(path, ordered=True)
    files_path = [(pjoin(path, file), file.split('.')[0]) for file in data]
    tmp = 0
    for file in files_path:
        tg = textgrid.TextGrid.fromFile(file[0])
        for i in range(7):
            if i == 5:
                tmp = max(tmp, tg[i])
            res = grid_process(tg[i], i)
            
