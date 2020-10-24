import csv
import glob
import re

import numpy as np


def dynamicTimeWarping(directory, file):
    values = {}
    base = fetchAQA(directory, file)
    baseSize = base.shape[0] - 1
    compareKeys = glob.glob(directory + "/*.wrd")
    compareKeys.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])

    for key in compareKeys:
        compare = fetchAQA(directory, key.replace(directory + "\\", ""))
        compareSize = compare.shape[0] - 1
        values[key] = DTWDist(base, baseSize, compare, compareSize)

    rankings = {k: v for k, v in sorted(values.items(), key=lambda x: x[1])}
    output = list(rankings.items())
    print()
    print("---------------------------------")
    print("  File   |  DTW Distance  ")
    print("---------------------------------")
    for i in range(10):
        print(output[i][0].split('\\')[-1] + "\t|" + "\t" + str(output[i][1]))


def fetchAQA(directory, file):
    values = []
    file = open(directory + "/" + file, "r")
    rows = csv.DictReader(file, ["c", "s", "w", "t", "a", "d", "x"])
    for row in rows:
        values.append(row['x'])
    return np.asfarray(values)


def DTWDist(seq1, seq1Index, seq2, seq2Index):
    if (seq1Index == 0 and seq2Index == 0):
        return 0
    elif (seq1Index == 0):
        return seq2[seq2Index]
    elif (seq2Index == 0):
        return seq1[seq1Index]
    else:
        return abs(seq1[seq1Index] - seq2[seq2Index])
        + min(DTWDist(seq1, seq1Index - 1, seq2, seq2Index), DTWDist(seq1, seq1Index, seq2, seq2Index - 1),
              DTWDist(seq1, seq1Index - 1, seq2, seq2Index - 1))
