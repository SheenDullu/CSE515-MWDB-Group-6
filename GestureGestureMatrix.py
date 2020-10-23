import glob
import re

import numpy as np
import pandas as pd

import Utilities


def dotProduct(directory):
    all_vectors = pd.DataFrame.from_dict(Utilities.getAllVectors(directory, 'tf'), orient='index')
    dot_product = all_vectors.dot(all_vectors.T)
    return dot_product


def editDistance(directory):
    allwrdfiles = Utilities.fetchAllWordsFromFile(directory)

    all_files = glob.glob(directory + "/*.wrd")
    all_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])

    important_sensors = set([5, 6, 7, 8, 9, 10, 11, 12])
    editValueMatrix = np.zeros([len(allwrdfiles) - 1, len(allwrdfiles) - 1], dtype=int)
    for i in range(len(allwrdfiles) - 1):
        for j in range(i + 1, len(allwrdfiles) - 1):
            editVal = 0
            for key in allwrdfiles[i]:
                # if (counter >30 and counter<60):

                numArr = []
                numArr2 = []
                for wrd in allwrdfiles[i][key]:
                    numArr.append(wrd)
                for wrd in allwrdfiles[j][key]:
                    numArr2.append(wrd)

                if (int(key[1]) in important_sensors):
                    multiplier = 2


                else:
                    multiplier = 0.5

                editVal += multiplier * Utilities.editDistanceFunc(numArr, numArr2)

            editValueMatrix[i][j] = editVal
            editValueMatrix[j][i] = editVal

        print("File " + all_files[i].split("\\")[-1].split(".")[0] + " done")
    np.save('editDistanceMatrix.npy', editValueMatrix)


if __name__ == '__main__':
    # dotProduct(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data')
    editDistance(r'C:\Class\CSE515 Multimedia\3_class_gesture_data')
