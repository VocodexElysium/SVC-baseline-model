lst = []

if __name__ == "__main__":
    with open("phoneme_table.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('|')
            line = line[1].split(' ')
            for phoneme in line:
                if not phoneme in lst:
                    lst.append(phoneme)
    lst.sort()
    print(lst)