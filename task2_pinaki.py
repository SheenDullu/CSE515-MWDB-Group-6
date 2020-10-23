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

def avg_std(df):
    avg = df.apply(np.mean, axis=0)
    std = df.apply(np.std, axis=0)
    return avg, std

def calculate_avg_std_sensor_wise(df):
    tran_df = pd.DataFrame(df.T)
    avg_temp, std_temp = avg_std(tran_df)
    avg_amplitude = avg_temp.to_frame('avg')
    std_deviation = std_temp.to_frame('std')
    amp_std = pd.concat([avg_amplitude, std_deviation], axis=1)
    return amp_std

def normalizeSensorWise(df):
    df_norm = df.subtract(df.min(axis=1), axis=0).multiply(2) \
        .divide(df.max(axis=1) - df.min(axis=1), axis=0).subtract(1).combine_first(df)
    return df_norm

def quantization(df, bands):
    mid_point = df.copy()
    for band in bands:
        df.mask((df >= band.lower_bound) & (df < band.upper_bound), band.index, inplace=True)
    df.loc[:] = df.astype(int)
    for band in bands:
        mid_point.mask((mid_point >= band.lower_bound) & (mid_point < band.upper_bound), band.mid_point, inplace=True)
    return df, mid_point

def create_word_dictionary(avg_std_df, mid_point_df, quantized_data, directory, window_length, shift_length):
    folder_name = directory.split("\\")[-1]
    word_list = list()
    for index, row in quantized_data.iterrows():
        for i in range(0, quantized_data.shape[1], shift_length):
            if i + window_length < quantized_data.shape[1]:
                avg_q = mid_point_df.loc[index][i:i + window_length].tolist()
                win = row[i:i + window_length].tolist()
                pair = [folder_name, index + 1, ' '.join(map(str, win)), i, avg_std_df.iloc[index]['avg'],
                        avg_std_df.iloc[index]['std'], np.mean(avg_q)]
                word_list.append(pair)
    return word_list

def read_gestures_from_csv(all_files, directory, shift_length, window_length, bands, word_dict):
    for file_ in all_files:
        file_name = file_.split("\\")[-1].split(".")[0]
        if file_name not in word_dict.keys():
            word_dict[file_name] = list()
        df = pd.read_csv(file_, header=None)
        column_names = [x for x in range(1, df.shape[1])]
        df = pd.DataFrame(df, columns=column_names)
        avg_std_df = calculate_avg_std_sensor_wise(df)
        df_norm = normalizeSensorWise(df)
        quantized_data, mid_point_df = quantization(df_norm.copy(), bands)
        word_dict[file_name].extend(create_word_dictionary(avg_std_df, mid_point_df, quantized_data, directory, window_length, shift_length))


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


def main():
    datadir = input("Enter the directory containing all the components, words and vectors: ")
    new_datadir = input("Enter the directory containing the query objects: ")
    window_size = int(input("Enter the window size: "))
    strides = int(input("Enter the strides: "))
    resolution = int(input("Enter the resolution: "))
    model = input("Enter 1 for TF and 2 for TF-IDF: ")
    user_option = input(" Enter 1 for PCA\n Enter 2 for SVD \n Enter 3 for NMF \n Enter 4 for LDA \n Enter 0 to exit: \n")
    dirs = Utilities.get_all_sub_folders(new_datadir)
    bands = Task0.gaussian_bands(resolution)
    word_dict = dict()
    for folder in dirs:
        print("Processing for Folder ", folder, "...")
        all_files = glob.glob(new_datadir + '\\' + folder + "/*.csv")
        file_directory = new_datadir + '\\' + folder
        read_gestures_from_csv(all_files, file_directory, strides, window_size, bands, word_dict)
        print("Done!")
    all_words = Utilities.fetchAllWordsFromDictionary(datadir)
    data_dict, data_df = fill_word_dictionary(word_dict, all_words)
    final_tf, final_tf_idf = calculations(new_datadir, data_dict, data_df, all_words)
    old_data = pd.read_csv(os.path.join(datadir,"latent_features.txt"),header=None)
    all_files_objects = glob.glob(datadir + "\W" + "/*.csv")
    all_files_objects.sort(key=lambda x:int((x.split("\\")[-1]).split(".")[0]))
    if user_option == '1':
        if model == '1':
            pca_reload = pickle.load(open(datadir + "\model_pca.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == '2':
            pca_reload = pickle.load(open(datadir + "\model_pca.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
    elif user_option == '2':
        if model == '1':
            pca_reload = pickle.load(open(datadir + "\model_svd.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == '2':
            pca_reload = pickle.load(open(datadir + "\model_svd.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
    elif user_option == '3':
        if model == '1':
            pca_reload = pickle.load(open(datadir + "\model_nmf.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == '2':
            pca_reload = pickle.load(open(datadir + "\model_nmf.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
    elif user_option == '4':
        if model == '1':
            pca_reload = pickle.load(open(datadir + "\model_lda.pkl",'rb'))
            final_tf = np.asarray(final_tf)
            trans_data = pca_reload.transform(final_tf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)
        elif model == '2':
            pca_reload = pickle.load(open(datadir + "\model_lda.pkl",'rb'))
            final_tf_idf = np.asarray(final_tf_idf)
            trans_data = pca_reload.transform(final_tf_idf)
            for i in range(len(trans_data)):
                similarity(old_data,trans_data[i],word_dict,all_files_objects)



if __name__ == '__main__':
    main()