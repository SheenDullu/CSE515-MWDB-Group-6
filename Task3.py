import csv

import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD

from GestureGestureMatrix import dotProduct, DTWMatrix


def performSVD(gesture_gesture_matrix, p, directory):
    svd = TruncatedSVD(n_components=p)
    svd.fit(gesture_gesture_matrix)
    features = getComponents(gesture_gesture_matrix, svd, p)
    writeToFile(directory + '\\task3a_svd.csv', features)
    print("***** Latent Feature, Gesture and its score are in task3b_svd.wrd *****")


def performNMF(gesture_gesture_matrix, p, directory):
    nmf = NMF(n_components=p, max_iter=3000)
    nmf.fit(gesture_gesture_matrix)
    features = getComponents(gesture_gesture_matrix, nmf, p)
    writeToFile(directory + '\\task3b_nmf.csv', features)
    print("***** Latent Feature, Gesture and its score are in task3b_nmf.wrd *****")


def getComponents(gesture_gesture_matrix, model, p):
    files = gesture_gesture_matrix.index.values
    features = list()
    for i in range(0, p):
        loading_scores = pd.DataFrame(columns=model.components_[i])
        loading_scores.loc[0] = files
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        for j in range(len(files)):
            features.append([i + 1, loading_scores.iloc[0, j].split('.')[0].split('_')[-1], sorted_values[j]])
    return features


def writeToFile(file_name, data):
    with open(file_name, 'w', newline="") as f:
        csv.writer(f).writerows(data)
        f.close()


def main():
    while True:
        task = int(input("Press 1 for performing SVD on Gesture Gesture Matrix \n"
                         "Press 2 for performing NMF on Gesture Gesture Matrix \n"))
        p = int(input("Number of principle components to use: "))
        while True:
            print("List of a Gesture Gesture Matrix:")
            print('Enter 1 for Dot Product')
            print('Enter 2 for PCA')
            print('Enter 3 for SVD')
            print('Enter 4 for NMF')
            print('Enter 5 for LDA')
            print('Enter 6 for Edit Distance')
            print('Enter 7 for DTW')
            gesture_model = int(input("Select a Gesture Gesture Matrix:\n"))
            print("Creating Gesture-Gesture matrix")
            if gesture_model == 1:
                gesture_gesture_matrix = dotProduct(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data')
            elif gesture_model == 6:
                # gesture_gesture_matrix = editDistance(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data')
                gesture_gesture_matrix = pd.read_csv('editDistanceMatrix.csv', index_col=0)
            elif gesture_model == 7:
                gesture_gesture_matrix = DTWMatrix(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data')
                # gesture_gesture_matrix = pd.read_csv('dtwDistanceMatrix.csv', index_col=0)
            elif gesture_model == 0:
                print("Thank you.")
                break

            if task == 1:
                print("Performing SVD...")
                performSVD(gesture_gesture_matrix, p, r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data')
            elif task == 2:
                print("Performing NMF...")
                performNMF(gesture_gesture_matrix, p, r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data')


if __name__ == '__main__':
    main()
