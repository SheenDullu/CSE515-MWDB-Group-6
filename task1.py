import csv
import glob
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA, NMF, TruncatedSVD

import Utilities


def reading_tf(datadir):
    tf = list()
    all_files = glob.glob(datadir + "/tf_*")
    all_files.sort(key=lambda x: int((x.split('\\')[-1]).split('.')[0].split('_')[-1]))
    for file_ in all_files:
        tf.append(Utilities.getAVector(file_))
    tf = np.asarray(tf)
    return tf


def reading_tfidf(datadir):
    tfidf = list()
    all_files = glob.glob(datadir + "/tfidf_*")
    all_files.sort(key=lambda x: int((x.split('\\')[-1]).split('.')[0].split('_')[-1]))
    for file_ in all_files:
        tfidf.append(Utilities.getAVector(file_))
    tfidf = np.asarray(tfidf)
    return tfidf


def calculating_nmf(k, vector, unique_word_dicts, datadir):
    nmf = NMF(n_components=k, max_iter=1000)
    nmf.fit(vector)
    nmf_data = nmf.transform(vector)

    word_file = open(datadir + '\word_score_nmf.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=nmf.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0, j], sorted_values[j]])
    word_file.close()
    print("Printed word and score file for NMF")
    pickle.dump(nmf, open(datadir + "\model_nmf.pkl", "wb"))
    print("NMF dumped")
    word_file = open(datadir + '\latent_features.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(len(nmf_data)):
        word_writer.writerow(nmf_data[i])
    word_file.close()


def calculating_pca(k, vector, unique_word_dicts, datadir):
    pca = PCA(n_components=k)
    # tf = tf.T
    pca.fit(vector)
    pca_data = pca.transform(vector)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    word_file = open(datadir + '\word_score_pca.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=pca.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        # sorted_loading_scores = list(loading_scores.sort_values(ascending=False))
        for j in range(len(unique_word_dicts)):
            word_writer.writerow(["Latent feature " + str(k + 1), loading_scores.iloc[0, j], sorted_values[j]])
    word_file.close()
    print("Printed word and score file for PCA")
    pickle.dump(pca, open(datadir + "\model_pca.pkl", "wb"))
    print("PCA dumped")
    word_file = open(datadir + '\latent_features.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(len(pca_data)):
        word_writer.writerow(pca_data[i])
    word_file.close()
    # pca_reload = pickle.load(open(datadir + "\pca.pkl",'rb'))


def calculating_svd(k, vector, unique_word_dicts, datadir):
    svd = TruncatedSVD(n_components=k)
    svd.fit(vector)
    svd_data = svd.transform(vector)

    word_file = open(datadir + '\word_score_svd.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=svd.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0, j], sorted_values[j]])
    word_file.close()
    print("Printed word and score file for SVD")
    pickle.dump(svd, open(datadir + "\model_svd.pkl", "wb"))
    print("SVD dumped")
    word_file = open(datadir + '\latent_features.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(len(svd_data)):
        word_writer.writerow(svd_data[i])
    word_file.close()


def calculating_lda(k, vector, unique_word_dicts, datadir):
    lda = LDA(n_components=k)
    lda.fit(vector)
    lda_data = lda.transform(vector)

    word_file = open(datadir + '\word_score_lda.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=lda.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        # sorted_loading_scores = list(loading_scores.sort_values(ascending=False))
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0, j], sorted_values[j]])
    word_file.close()
    print("Printed word and score file for LDA")
    pickle.dump(lda, open(datadir + "\model_lda.pkl", "wb"))
    print("LDA dumped")
    word_file = open(datadir + '\latent_features.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(len(lda_data)):
        word_writer.writerow(lda_data[i])
    word_file.close()


def main():
    while True:
        print("Build the PCA, SVD, NMF, LDA word scores")
        # directory = input("Enter the directory containing all the components, words and vectors: ")
        directory = Utilities.read_directory()
        k = int(input("Enter how many top latent features you want: "))
        vector_model = int(input("Enter the vector model you want to use \nEnter 1 for TF or Enter 2 for TF-IDF: "))
        model = list()
        if vector_model == 1:
            model = reading_tf(directory)
        elif vector_model == 2:
            model = reading_tfidf(directory)

        unique_word_dicts = Utilities.getAllUniqueWords(directory)
        while True:
            user_option = int(input("Enter 1 for PCA\nEnter 2 for SVD \n"
                                    "Enter 3 for NMF \nEnter 4 for LDA \n"
                                    "Enter 0 to exit: \n"))
            if user_option == 1:
                calculating_pca(k, model, unique_word_dicts, directory)
            elif user_option == 2:
                calculating_svd(k, model, unique_word_dicts, directory)
            elif user_option == 3:
                calculating_nmf(k, model, unique_word_dicts, directory)
            elif user_option == 4:
                calculating_lda(k, model, unique_word_dicts, directory)
            elif user_option == 0:
                break
        out = int(input("Press 1 to continue\nPress 0 to exit Task 1\n"))
        if out == 0:
            break
        elif out == 1:
            continue


if __name__ == '__main__':
    main()
