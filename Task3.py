import csv
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD

import Utilities
from GestureGestureMatrix import dotProduct, editDistance, DTWMatrix


def performSVD(gesture_gesture_matrix, p, directory):
    svd = TruncatedSVD(n_components=p)
    svd.fit(gesture_gesture_matrix)
    features = getComponents(gesture_gesture_matrix, svd, p)
    writeToFile(directory + '\\task3a_svd.csv', features)
    print("***** Latent Feature, Gesture and its score are in task3b_svd.wrd *****")
    pickle.dump(svd,open(directory + "\model_svd_task3.pkl","wb"))


def performNMF(gesture_gesture_matrix, p, directory):
    nmf = NMF(n_components=p, max_iter=4000)
    nmf.fit(gesture_gesture_matrix)
    features = getComponents(gesture_gesture_matrix, nmf, p)
    writeToFile(directory + '\\task3b_nmf.csv', features)
    print("***** Latent Feature, Gesture and its score are in task3b_nmf.wrd *****")
    pickle.dump(nmf,open(directory + "\model_nmf_task3.pkl","wb"))


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

def similarity(old_data,new_data,gesture_gesture_matrix):
    sim = list()
    for index, row in enumerate(old_data):
        some = np.asarray(old_data[index])
        temp_new_data = np.asarray(new_data)
        dis = np.linalg.norm(some-temp_new_data)
        # cov_data = data = np.stack((old_data[index],new_data), axis=1)
        # cov_matrix = np.cov(cov_data)
        # print(cov_matrix)
        # cov_matrix = np.linalg.inv(cov_matrix)
        # dis = distance.mahalanobis(old_data[index],new_data,cov_matrix)
        sim.append(dis)
    gesture_gesture_matrix.append(sim)


def write_similarity_matrix(gesture_gesture_matrix,datadir,user_option):
    all_files_objects = glob.glob(datadir + "/W/*.csv")
    all_files_objects.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])
    file_names = list()
    for i in all_files_objects:
        file_names.append(i.split("\\")[-1])
    df = pd.DataFrame(gesture_gesture_matrix, columns=file_names, index=file_names)
    df_norm = df.subtract(df.min(axis=1), axis=0) \
        .divide(df.max(axis=1) - df.min(axis=1), axis=0) \
        .combine_first(df)
    df_norm = 1 - df_norm
    if user_option == 2:
        df_norm.to_csv('similarity_matrix_pca.csv')
    elif user_option == 3:
        df_norm.to_csv('similarity_matrix_svd.csv')
    elif user_option == 4:
        df_norm.to_csv('similarity_matrix_nmf.csv')
    elif user_option == 5:
        df_norm.to_csv('similarity_matrix_lda.csv')


def main():
    while True:
        task = int(input("Press 1 for performing SVD on Gesture Gesture Matrix \n"
                         "Press 2 for performing NMF on Gesture Gesture Matrix \n"))
        directory = Utilities.read_directory()
        p = int(input("Number of principle components to use: "))
        while True:
            print("List of a Gesture Gesture Matrix:")
            print('Enter 1 for Dot Product\nEnter 2 for PCA\nEnter 3 for SVD')
            print('Enter 4 for NMF\nEnter 5 for LDA')
            print('Enter 6 for Edit Distance\nEnter 7 for DTW \n')
            gesture_model = int(input("Select a Gesture Gesture Matrix (Enter 0 to exit)\n"))
            print("Creating Gesture-Gesture matrix")
            if gesture_model == 1:
                gesture_gesture_matrix = dotProduct(directory)
            elif gesture_model == 6:
                gesture_gesture_matrix = editDistance(directory)
                # gesture_gesture_matrix = pd.read_csv('editDistanceMatrix.csv', index_col=0)
            elif gesture_model == 7:
                gesture_gesture_matrix = DTWMatrix(directory)
                # gesture_gesture_matrix = pd.read_csv('dtwDistanceMatrix.csv', index_col=0)
            elif gesture_model == 2:
                gesture_gesture_matrix = list()
                old_data = pd.read_csv(os.path.join(directory, "latent_features_pca_task1.txt"), header=None)
                old_data = old_data.values.tolist()
                trans_data = old_data.copy()
                for i in range(len(trans_data)):
                    similarity(old_data,trans_data[i],gesture_gesture_matrix)
                write_similarity_matrix(gesture_gesture_matrix, directory, gesture_model)
                gesture_gesture_matrix = pd.read_csv('similarity_matrix_pca.csv', index_col=0)
            elif gesture_model == 3:
                gesture_gesture_matrix = list()
                old_data = pd.read_csv(os.path.join(directory, "latent_features_svd_task1.txt"), header=None)
                old_data = old_data.values.tolist()
                trans_data = old_data.copy()
                for i in range(len(trans_data)):
                    similarity(old_data,trans_data[i],gesture_gesture_matrix)
                write_similarity_matrix(gesture_gesture_matrix, directory, gesture_model)
                gesture_gesture_matrix = pd.read_csv('similarity_matrix_svd.csv', index_col=0)
            elif gesture_model == 4:
                gesture_gesture_matrix = list()
                old_data = pd.read_csv(os.path.join(directory, "latent_features_nmf_task1.txt"), header=None)
                old_data = old_data.values.tolist()
                trans_data = old_data.copy()
                for i in range(len(trans_data)):
                    similarity(old_data,trans_data[i],gesture_gesture_matrix)
                write_similarity_matrix(gesture_gesture_matrix, directory, gesture_model)
                gesture_gesture_matrix = pd.read_csv('similarity_matrix_nmf.csv', index_col=0)
            elif gesture_model == 5:
                gesture_gesture_matrix = list()
                old_data = pd.read_csv(os.path.join(directory, "latent_features_lda_task1.txt"), header=None)
                old_data = old_data.values.tolist()
                trans_data = old_data.copy()
                for i in range(len(trans_data)):
                    similarity(old_data,trans_data[i],gesture_gesture_matrix)
                write_similarity_matrix(gesture_gesture_matrix, directory, gesture_model)
                gesture_gesture_matrix = pd.read_csv('similarity_matrix_lda.csv', index_col=0)
            elif gesture_model == 0:
                break

            if task == 1:
                print("Performing SVD...")
                performSVD(gesture_gesture_matrix, p, directory)
            elif task == 2:
                print("Performing NMF...")
                performNMF(gesture_gesture_matrix, p, directory)
        out = int(input("Press 1 to continue\nPress 0 to exit Task 3\n"))
        if out == 0:
            break
        elif out == 1:
            continue


if __name__ == '__main__':
    main()
