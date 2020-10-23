import glob
import os
import re

import numpy as np


def get_all_sub_folders(folder_directory):
    return [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]


def fetchAllWordsFromDictionary(directory):
    words = set()
    all_files = glob.glob(directory + "/*.wrd")
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                word = ' '.join(map(str, row[0:3]))
                words.add(word)
    return dict.fromkeys(sorted(words), 0)


def getAllUniqueWords(directory):
    filename = directory + r'\header.txt'
    file = open(filename, 'r')
    s = file.read()
    return s.split(',')


def getAllVectors(directory, model):
    vectors = dict()
    all_files = glob.glob(directory + "/" + model + "_vectors_*.txt")
    all_files.sort(key=lambda x: int((x.split('\\')[-1]).split('.')[0].split('_')[-1]))
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                vector = [float(i) for i in row]
                vectors[filename.split('\\')[-1].split("_")[-1].split('.')[0]] = vector
    return vectors


def getAVector(file):
    file = open(file, 'r')
    f = file.read()
    vector_string = f.split(',')
    vector = [float(i) for i in vector_string]
    return vector


def fetchWordsFromFile(directory, filename):
    wordDict = {}
    words = []
    file = directory + "/" + filename + ".wrd"
    with open(file, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            word = ' '.join(map(str, row[0:3]))
            key = row[0] + row[1]
            wrd = row[2]
            if key in wordDict:
                wordDict[key].append(wrd)
            else:
                wrdlist = []
                wrdlist.append(wrd)
                wordDict[key] = wrdlist
    return wordDict


def fetchAllWordsFromFile(directory):
    wordArrayDict = []
    all_files = glob.glob(directory + "/*.wrd")
    all_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])
    for filename in all_files:
        wordDict = {}
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                word = ' '.join(map(str, row[0:3]))
                key = row[0] + row[1]
                wrd = row[2]
                if key in wordDict:
                    wordDict[key].append(wrd)
                else:
                    wrdlist = []
                    wrdlist.append(wrd)
                    wordDict[key] = wrdlist
        wordArrayDict.append(wordDict)
    return wordArrayDict


def editDistanceComp(matrix, p, q, P, Q, r, i, d, con):
    replaceCost = 2
    insertCost = 1
    deleteCost = 1

    if (p == 0 or q == 0):
        return 0

    # Commented out the below code so that I can test the cost function. Can optimise the code later.
    # if(i+d>0.3*con):
    #     # print("id",i,d,0.3*con)
    #     return 1000
    #
    # if(r>0.2* con):
    #     # print('r',r)
    #     return 1000

    if (P[p] == Q[q]):
        if (matrix[p - 1, q - 1] == -1):
            matrix[p - 1, q - 1] = editDistanceComp(matrix, p - 1, q - 1, P, Q, r, i, d, con)
        return matrix[p - 1, q - 1]

    if (matrix[p, q - 1] == -1):
        matrix[p, q - 1] = editDistanceComp(matrix, p, q - 1, P, Q, r, i, d + 1, con)

    if (matrix[p - 1, q - 1] == -1):
        matrix[p - 1, q - 1] = editDistanceComp(matrix, p - 1, q - 1, P, Q, r + 1, i, d, con)
    if (matrix[p - 1, q] == -1):
        matrix[p - 1, q] = editDistanceComp(matrix, p - 1, q, P, Q, r, i + 1, d, con)

    # print(r,i,d)
    return min(matrix[p - 1, q] + insertCost, matrix[p, q - 1] + deleteCost, matrix[p - 1, q - 1] + 1)


def editDistanceFunc(P, Q):
    if len(P) < len(Q):
        P, Q, = Q, P
    i = len(P)
    j = len(Q)

    con = (max(i, j))
    matrix = np.ones((i, j)) * -1

    matrix[:, 0] = np.arange(0, i)
    matrix[0, :] = np.arange(0, j)

    matrix[i - 1, j - 1] = editDistanceComp(matrix, i - 1, j - 1, P, Q, 0, 0, 0, con)

    return matrix[i - 1, j - 1]


def store_directory(directory):
    with open("directory.txt", 'w') as f:
        f.write(directory)
        f.close()


def read_directory():
    with open("directory.txt", 'r') as f:
        param = f.read()
        f.close()
    return param


if __name__ == '__main__':
    # getAllUniqueWords(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data')
    # getAllVectors(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data', 'tf')
    getAVector(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data\tf_vectors_1.txt')
