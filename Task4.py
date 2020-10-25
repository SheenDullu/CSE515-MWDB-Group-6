import pickle
import sys

import numpy as np
import pandas as pd

import Utilities
import kmeans
import task4ab_pinaki


def task4a(directory):
    svd = pd.read_csv(directory, header=None, names=['features', 'file', 'score'])

    num_files = len(svd['file'].value_counts())
    num_features = len(svd['features'].value_counts())
    result = np.ones((1, num_files))
    for f in range(1, num_features + 1):
        temp = (svd[svd['features'] == f])
        temp = temp.sort_values(['file'])
        mean = temp['score'].mean()
        std = temp['score'].std()
        temp = temp.to_numpy()

        result += (np.exp(-np.square(temp[:, 2] - mean) / (2 * std * std)))

    result = result.reshape((num_files, 1))
    numbers = np.arange(1, num_files + 1).reshape((num_files, 1))
    result = np.concatenate([result, numbers], axis=1)
    result = (result[result[:, 0].argsort()])

    return result


def task4c(directory):
    data = pd.read_csv(directory, header=None,
                       names=['features', 'file', 'score'])

    num_features = data['features'].max()
    for f in range(1, num_features + 1):
        temp = data[data['features'] == f].sort_values(['file']).drop(['features', 'file'], axis=1).to_numpy()

        if (f == 1):
            result = temp
        else:
            result = np.concatenate([result, temp], axis=1)

    trial = kmeans.Cluster(result, 3, 1)
    k = trial.kmeans()

    print(k)


def task4d(ggMatrix, k):
    threshold = 0.5  # IMPLEMENT SEPARATE THRESHOLD FUNCTION
    cols = list(ggMatrix.columns)  #
    for col in cols:  #
        ggMatrix.loc[ggMatrix[col] > threshold, col] = 0  #

    dMatrix = pd.DataFrame(index=cols, columns=cols)
    dMatrix = dMatrix.fillna(0)

    for col in cols:
        degree = 0
        values = ggMatrix[col]
        for val in values:
            if val > 0:
                degree += 1
        dMatrix.loc[col, col] = degree

    lMatrix = dMatrix.subtract(ggMatrix)
    L = lMatrix.to_numpy()
    eVectors = np.linalg.eig(L)
    values = []
    vectors = np.empty([len(cols), int(k)])
    for i in range(int(k)):
        max = np.amax(eVectors[0])
        values.append(max)
        loc = np.where(eVectors[0] == max)[0][0]
        vectors[:, i] = eVectors[1][:, loc]
        eVectors[0][loc] = -sys.maxsize - 1

    x = kmeans.Cluster(vectors, int(k))
    x.kmeans()
    output = x.cluster_obj

    for i in range(1, int(k) + 1):
        for j in range(len(output[i])):
            output[i][j] = cols[output[i][j]]

    for i in range(1, int(k) + 1):
        print()
        print()
        print("Cluster " + str(i))
        print("----------")
        for j in range(len(output[i])):
            print(output[i][j])
    print()


def main():
    while True:
        print("Latent Gesture Clustering and Analysis Tasks")
        directory = Utilities.read_directory()
        print("Press 1 for Task 4a\nPress 2 from Task 4b\nPress 3 for Task 4c\nPress4 for Task 4d")
        task = int(input("Press a task to work on (Press 0 to exit)"))
        if task == 1:
            svd_reload = pickle.load(open(directory + "\model_svd_task3.pkl", 'rb'))
            task4ab_pinaki.degree_of_membership(svd_reload, directory)
        elif task == 2:
            svd_reload = pickle.load(open(directory + "\model_nmf_task3.pkl", 'rb'))
            task4ab_pinaki.degree_of_membership(svd_reload, directory)
        elif task == 3:
            ggMatrix, k = show()
            task4c(ggMatrix, k)
        elif task == 4:
            ggMatrix, k = show()
            task4d(ggMatrix, k)


def show():
    print("Enter [1] for dot product")
    print("Enter [2] for PCA")
    print("Enter [3] for SVD")
    print("Enter [4] for NMF")
    print("Enter [5] for LDA")
    print("Enter [6] for edit distance")
    print("Enter [7] for DTW")
    gg = input("Which gesture-gesture similarity matrix would you like to use: ")
    k = input("How many clusters would you like to compute: ")
    if gg == "1":
        ggMatrix = pd.read_csv("dotProductMatrix.csv", index_col=0)
    elif gg == "2":
        ggMatrix = pd.read_csv("similarity_matrix_pca.csv", index_col=0)
    elif gg == "3":
        ggMatrix = pd.read_csv("similarity_matrix_svd.csv", index_col=0)
    elif gg == "4":
        ggMatrix = pd.read_csv("similarity_matrix_nmf.csv", index_col=0)
    elif gg == "5":
        ggMatrix = pd.read_csv("similarity_matrix_lda.csv", index_col=0)
    elif gg == "6":
        ggMatrix = pd.read_csv("editDistanceMatrix.csv", index_col=[0])
    elif gg == "7":
        ggMatrix = pd.read_csv("dtwDistanceMatrix.csv", index_col=[0])
    return ggMatrix, k


if __name__ == '__main__':
    main()
