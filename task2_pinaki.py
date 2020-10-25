import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,NMF,TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.spatial import distance
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
import glob
import matplotlib.pyplot as plt
import os
import csv
import pickle
import Utilities
import Task0
import math
import heapq


def getAllWords(directory):
    words = list()
    all_files = glob.glob(directory + "/*.wrd")

    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                word = ' '.join(map(str, row[0:3]))
                if word not in words:
                    words.append(word)
    return words



def parse_and_store_file_data(file_dict, key, value, all_words, vectors):
    file_name = key
    if file_name not in file_dict.keys():
        file_dict[file_name] = all_words.copy()
    for row in value:
        word = ' '.join(map(str, row[0:3]))
        if word in file_dict[file_name].keys():
            file_dict[file_name][word] += 1
    vector = dict()
    vector["file"] = file_name
    vector.update(file_dict[file_name])
    vectors.append(vector)


def fill_word_dictionary(word_dict, all_words):
    file_dict = dict()
    vectors = list()
    for key,value in word_dict.items():
        parse_and_store_file_data(file_dict, key, value, all_words, vectors)
    df = pd.DataFrame(vectors)
    return file_dict, df

def calculations(directory, data_dict, data_df, all_words):
    # to keep track of the unique words

    tf = all_words.copy()
    tf_idf = all_words.copy()
    final_tf = list()
    final_tf_idf = list()

    for file_name, words_dict in data_dict.items():
        tf_vector = list()
        tf_idf_vector = list()
        total_words = sum(words_dict.values())

        num_of_words_in_gesture = data_df.astype(bool).sum(axis=0)
        for word, count in words_dict.items():
            tf[word] = count / total_words

            d_idf = float(num_of_words_in_gesture[word])
            tf_idf[word] = float(tf[word]) * (math.log10(len(data_df.columns) / d_idf)) if d_idf > 0.0 else 0.0

        tf_vector.extend(tf.values())
        tf_idf_vector.extend(tf_idf.values())
        final_tf.append(tf_vector)
        final_tf_idf.append(tf_idf_vector)
    return final_tf,final_tf_idf

def similarity(old_data,new_data,word_dict,all_files_objects):
    heap = []
    for index, row in old_data.iterrows():
        some = list(row)
        some = np.asarray(some)
        dis = np.linalg.norm(some-new_data)
        # cov_data = data = np.array([new_data,some]).T
        # cov_matrix = np.cov(cov_data)
        # print(cov_matrix.shape)
        # cov_matrix = np.linalg.inv(cov_matrix)
        # dis = distance.mahalanobis(some,new_data,cov_matrix)
        current_file = all_files_objects[index].split('\\')[-1]
        heap.append((dis,current_file))
    heapq.heapify(heap)
    similar_objects = list()
    for i in range(10):
        a,b = heapq.heappop(heap)
        similar_objects.append(b)
    print(similar_objects)
    print("--------------")


def main(user_option,model,gesture_file):
    datadir = Utilities.read_directory()
    # gesture_file = input("Enter the gesture object: ")
    window_size = int(input("Enter the window size: "))
    strides = int(input("Enter the strides: "))
    resolution = int(input("Enter the resolution: "))
    # model = input("Enter 1 for TF and 2 for TF-IDF: ")
    # user_option = input(" Enter 1 for PCA\n Enter 2 for SVD \n Enter 3 for NMF \n Enter 4 for LDA \n Enter 0 to exit: \n")
    
    dirs = Utilities.get_all_sub_folders(datadir)
    bands = Task0.gaussian_bands(resolution)
    word_dict = dict()
    for folder in dirs:
        print("Processing for Folder ", folder, "...")
        all_files = glob.glob(datadir + '\\' + folder + "\\" + gesture_file + ".csv")
        file_directory = datadir + '\\' + folder
        Task0.read_gestures_from_csv(all_files, file_directory, strides, window_size, bands, word_dict)
        print("Done!")

    all_words = Utilities.fetchAllWordsFromDictionary(datadir)
    data_dict, data_df = fill_word_dictionary(word_dict, all_words)
    final_tf, final_tf_idf = calculations(datadir, data_dict, data_df, all_words)

    # old_data = pd.read_csv(os.path.join(datadir,"latent_features.txt"),header=None)
    all_files_objects = glob.glob(datadir + "\W" + "/*.csv")
    all_files_objects.sort(key=lambda x:int((x.split("\\")[-1]).split(".")[0]))
    if user_option == 2:
        old_data = pd.read_csv(os.path.join(datadir,"latent_features_pca_task1.txt"),header=None)
        if model == 'tf':
            pca_reload = pickle.load(open(datadir + "\model_pca_task1.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == 'tfidf':
            pca_reload = pickle.load(open(datadir + "\model_pca_task1.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
    elif user_option == 3:
        old_data = pd.read_csv(os.path.join(datadir,"latent_features_svd_task1.txt"),header=None)
        if model == 'tf':
            pca_reload = pickle.load(open(datadir + "\model_svd_task1.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == 'tfidf':
            pca_reload = pickle.load(open(datadir + "\model_svd_task1.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
    elif user_option == 4:
        old_data = pd.read_csv(os.path.join(datadir,"latent_features_nmf_task1.txt"),header=None)
        if model == 'tf':
            pca_reload = pickle.load(open(datadir + "\model_nmf_task1.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == 'tfidf':
            pca_reload = pickle.load(open(datadir + "\model_nmf_task1.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
    elif user_option == 5:
        old_data = pd.read_csv(os.path.join(datadir,"latent_features_lda_task1.txt"),header=None)
        if model == 'tf':
            pca_reload = pickle.load(open(datadir + "\model_lda_task1.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == 'tfidf':
            pca_reload = pickle.load(open(datadir + "\model_lda_task1.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)



if __name__ == '__main__':
    main()