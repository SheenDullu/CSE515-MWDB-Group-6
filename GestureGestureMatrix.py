import glob
import re

import numpy as np
import pandas as pd

import Utilities
import dtw


def dotProduct(directory):
    all_vectors = pd.DataFrame.from_dict(Utilities.getAllVectors(directory, 'tf'), orient='index')
    df = all_vectors.dot(all_vectors.T)
    df_norm = df.subtract(df.min(axis=1), axis=0) \
        .divide(df.max(axis=1) - df.min(axis=1), axis=0) \
        .combine_first(df)
    df_norm = 1 - df_norm
    df_norm.to_csv('dotProductMatrix.csv')
    return df_norm


def DTWMatrix(directory):
    keys = glob.glob(directory + "/*.wrd")
    keys.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])
    keys = [key.replace(directory + "\\", "") for key in keys]
    matrix = np.empty([len(keys), len(keys)])
    for key1 in keys:
        base = dtw.fetchAQA(directory, key1)
        baseSize = base.shape[0] - 1
        for key2 in keys:
            compare = dtw.fetchAQA(directory, key2)
            compareSize = compare.shape[0] - 1
            matrix[keys.index(key1), keys.index(key2)] = dtw.DTWDist(base, baseSize, compare, compareSize)
    df = pd.DataFrame(matrix, index=keys, columns=keys)
    df_norm = df.subtract(df.min(axis=1), axis=0) \
        .divide(df.max(axis=1) - df.min(axis=1), axis=0) \
        .combine_first(df)
    df_norm = 1 - df_norm
    df_norm.to_csv("dtwDistanceMatrix.csv")
    return df


def editDistance(directory):
    allwrdfiles = Utilities.fetchAllWordsFromFile(directory)

    all_files = glob.glob(directory + "/*.wrd")
    all_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])

    important_sensors = set([5, 6, 7, 8, 9, 10, 11, 12])
    editValueMatrix = np.zeros([len(allwrdfiles), len(allwrdfiles)], dtype=float)
    file_names = []
    for i in range(len(allwrdfiles)):
        for j in range(i + 1, len(allwrdfiles)):
            editVal = 0
            for key in allwrdfiles[i]:
                # if (counter >30 and counter<60):
                numArr = []
                numArr2 = []
                for wrd in allwrdfiles[i][key]:
                    numArr.append(wrd)
                for wrd in allwrdfiles[j][key]:
                    numArr2.append(wrd)
                if int(key[1]) in important_sensors:
                    multiplier = 2
                else:
                    multiplier = 0.5
                editVal += multiplier * Utilities.editDistanceFunc(numArr, numArr2)

            editValueMatrix[i][j] = editVal
            editValueMatrix[j][i] = editVal

        file_names.append(all_files[i].split("\\")[-1].split(".")[0] + ".txt")
        print("File " + all_files[i].split("\\")[-1].split(".")[0] + " done")
    df = pd.DataFrame(editValueMatrix, columns=file_names, index=file_names)
    df_norm = df.subtract(df.min(axis=1), axis=0) \
        .divide(df.max(axis=1) - df.min(axis=1), axis=0) \
        .combine_first(df)
    df_norm = 1 - df_norm
    df_norm.to_csv('editDistanceMatrix.csv')
    df.to_csv('editDistanceMatrixOriginal.csv')
    return df_norm


if __name__ == '__main__':
    #dotProduct(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data')
    editDistance(r'C:\Class\CSE515 Multimedia\3_class_gesture_data')
