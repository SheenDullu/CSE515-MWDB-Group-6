import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,NMF,TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
import glob
import matplotlib.pyplot as plt
import os
import csv
# pd.read_csv("F:\mwdb\data")
# datadir = "F:\mwdb\data"


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

def reading_tf(datadir):
    tf = list()
    all_files = glob.glob(datadir + "/tf_*")
    all_files.sort(key=lambda x:int((x.split("_")[2]).split(".")[0]))
    for file_ in all_files:
        data = pd.read_csv(file_,header=None)
        tf.append(list(data.iloc[0]))
    tf = np.asarray(tf)
    return tf



def reading_tfidf(datadir):
    tfidf = list()
    all_files = glob.glob(datadir + "/tfidf_*")
    all_files.sort(key=lambda x:int((x.split("_")[2]).split(".")[0]))
    for file_ in all_files:
        data = pd.read_csv(file_,header=None)
        tfidf.append(list(data.iloc[0]))
    tfidf = np.asarray(tfidf)
    return tfidf
    

def calculating_nmf(k,vector,unique_word_dicts):
    nmf = NMF(n_components=k)
    nmf.fit(vector)
    nmf_data = nmf.transform(vector)


    # per_var = np.round(nmf.explained_variance_ratio_* 100, decimals=1)
    # labels = ["PC" + str(i) for i in range(1,len(per_var)+1)]
    # plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    # plt.ylabel('Percentage of explained vaiance')
    # plt.xlabel('Principal Component')
    # plt.show()


    word_file = open('F:\mwdb\data\word_score_nmf.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=nmf.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        # sorted_loading_scores = list(loading_scores.sort_values(ascending=False))
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0,j],sorted_values[j]])
    print("Printed word and score file for NMF")

def calculating_pca(k,vector,unique_word_dicts):
    pca = PCA(n_components=k)
    # tf = tf.T
    pca.fit(vector)
    pca_data = pca.transform(vector)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)

    word_file = open('F:\mwdb\data\word_score_pca.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=pca.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        # sorted_loading_scores = list(loading_scores.sort_values(ascending=False))
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0,j],sorted_values[j]])
    print("Printed word and score file for PCA")


def calculating_svd(k,vector,unique_word_dicts):
    svd = TruncatedSVD(n_components=k)
    svd.fit(vector)
    svd_data = svd.transform(vector)

    word_file = open('F:\mwdb\data\word_score_svd.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=svd.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        # sorted_loading_scores = list(loading_scores.sort_values(ascending=False))
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0,j],sorted_values[j]])
    print("Printed word and score file for SVD")

def calculating_lda(k,vector,unique_word_dicts):
    lda = LDA(n_components=k)
    lda.fit(vector)
    lda_data = lda.transform(vector)
    
    word_file = open('F:\mwdb\data\word_score_lda.txt', mode='w')
    word_writer = csv.writer(word_file, delimiter=',')
    for i in range(k):
        loading_scores = pd.DataFrame(columns=lda.components_[i])
        loading_scores.loc[0] = unique_word_dicts
        loading_scores.sort_index(axis=1, ascending=False, inplace=True)
        sorted_values = list(loading_scores.columns)
        # sorted_loading_scores = list(loading_scores.sort_values(ascending=False))
        for j in range(len(unique_word_dicts)):
            word_writer.writerow([loading_scores.iloc[0,j],sorted_values[j]])
    print("Printed word and score file for LDA")    



def main():
    flag = 0
    datadir = input("Enter the directory containing all the components, words and vectors: ")
    k = int(input("Enter how many top latent features you want: "))
    vector_model = input("Enter the vector model you want to use \nEnter 1 for TF or Enter 2 for TF-IDF: ")

    while True:
        if vector_model == "1":
            print("Reading TF vectors...")
            tf = reading_tf(datadir)
            flag = 1
            break
        elif vector_model == "2":
            print("Reading TFIDF vectors...")
            tfidf = reading_tfidf(datadir)
            flag = 2
            break
        elif vector_model == "0":
            exit()

    unique_word_dicts = getAllWords(datadir)
    
    user_option = input(" Enter 1 for PCA\n Enter 2 for SVD \n Enter 3 for NMF \n Enter 4 for LDA \n Enter 0 to exit: \n")
    while True:
        if user_option == "1":
            if flag == 1:
                calculating_pca(k,tf,unique_word_dicts)
            else:
                calculating_pca(k,tfidf,unique_word_dicts)
            break
        elif user_option == "2":
            if flag == 1:
                calculating_svd(k,tf,unique_word_dicts)
            else:
                calculating_svd(k,tfidf,unique_word_dicts)
            break
        elif user_option == "3":
            if flag == 1:
                calculating_nmf(k,tf,unique_word_dicts)
            else:
                calculating_nmf(k,tfidf,unique_word_dicts)
            break
        elif user_option == "4":
            if flag == 1:
                calculating_lda(k,tf,unique_word_dicts)
            else:
                calculating_lda(k,tfidf,unique_word_dicts)
            break
        elif user_option == "0":
            exit()
    


if __name__ == '__main__':
    main()


