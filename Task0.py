import csv
import glob
import math
import os.path

import numpy as np
import pandas as pd
import scipy.stats
from scipy.integrate import quad


class Bands:
    def __init__(self, index, lower_bound, upper_bound):
        self.index = index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


def gaussian_bands(resolution):
    mean = 0
    std = 0.25
    x_min = -1.0
    x_max = 1.0
    x = np.linspace(x_min, x_max, 100)

    def normal_distribution_function(x):
        value = scipy.stats.norm.pdf(x, mean, std)
        return value

    total_area, err = quad(normal_distribution_function, x_min, x_max)
    total_area = round(total_area, 5)
    list_bands = list()
    upper_bound = 1
    index = 1
    for i in range(1, 2 * resolution + 1):
        x1 = (i - resolution - 1) / resolution
        x2 = (i - resolution) / resolution
        area, err = quad(normal_distribution_function, x1, x2)
        area = round(area, 5)
        length = round(2.0 * (area / total_area), 5)

        band = Bands(index, round(upper_bound - length, 5), round(upper_bound, 5))
        upper_bound = upper_bound - length
        list_bands.append(band)
        index += 1
    return list_bands


def quantization(df, bands):
    for band in bands:
        df.mask((df >= band.lower_bound) & (df < band.upper_bound), band.index, inplace=True)
    df.loc[:] = df.astype(int)
    return df


def normalizeSensorWise(df):
    df_norm = df.subtract(df.min(axis=1), axis=0).multiply(2) \
        .divide(df.max(axis=1) - df.min(axis=1), axis=0).subtract(1).combine_first(df)
    return df_norm


def write_quantized_data(data, directory, gesture, window_length, shift_length):
    file_name = gesture.split("\\")[-1].split(".")[0]
    new_file = directory + "/" + file_name + ".wrd"
    vector = list()
    with open(new_file, 'w', newline="") as x:
        for index, row in data.iterrows():
            for i in range(0, data.shape[1], shift_length):
                if i + window_length < data.shape[1]:
                    win = row[i:i + window_length].tolist()
                    pair = [file_name, index + 1, i]
                    vector.append(pair + win)
        csv.writer(x, delimiter=' ').writerows(vector)

def Avg_Std(df):

    avg=df.apply(np.mean,axis=0)

    std=df.apply(np.std,axis=0)

    return avg,std


def read_gestures_from_csv(all_files, directory, resolution, shift_length, window_length):
    print("Building Gaussian Bands...")
    bands = gaussian_bands(resolution)
    print('Reading data from the given folder and quantizing...')
    avg_df=pd.DataFrame()
    std_df=pd.DataFrame()
    for filename in all_files:
        df = pd.read_csv(filename, header=None)
        column_names = [x for x in range(1, df.shape[1])]
        df = pd.DataFrame(df, columns=column_names)

        tran_df=pd.DataFrame(df.T)
        avg_temp,std_temp=Avg_Std(tran_df)
        avg_df=pd.concat([avg_df,avg_temp],axis=1)
        std_df=pd.concat([std_df,std_temp],axis=1)

        df_norm = normalizeSensorWise(df)
        quantized_data = quantization(df_norm, bands)
        write_quantized_data(quantized_data, directory, filename, window_length, shift_length)

    column_names = [x for x in range(0, avg_df.shape[1])]
    avg_df.columns=column_names
    std_df.columns=column_names




def task0a(folder_directory, window_length, shift_length, resolution):
    dirs = [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]
    for folder in dirs:
        print("   Processing for Folder ", folder, "...")
        all_files = glob.glob(folder_directory + '\\' + folder + "/*.csv")
        file_directory = folder_directory + '\\' + folder
        read_gestures_from_csv(all_files, file_directory, resolution, shift_length, window_length)
    print("     ****Created .wrd files for all the gestures.****")


def get_all_words_from_directory(directory):
    words = list()
    all_files = glob.glob(directory + "/*.wrd")
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(' ')
                word = ' '.join(row[3:])
                if word not in words:
                    words.append(word)
    return dict.fromkeys(sorted(words), 0)


def create_word_dictionary(directory, all_words):
    all_files = glob.glob(directory + "/*.wrd")
    file_dict = dict()
    vectors = list()
    for file in all_files:
        parse_and_store_file_data(file_dict, file, all_words, vectors)
    df = pd.DataFrame(vectors)
    return file_dict, df


def parse_and_store_file_data(file_dict, file, all_words, vectors):
    file_name = file.split("\\")[-1].split(".")[0]
    with open(file, 'r') as f:
        sensor_dict = dict()
        for line in f:
            row = line.strip().split(' ')
            word = ' '.join(row[3:])
            if row[1] not in sensor_dict.keys():
                sensor_dict[row[1]] = all_words.copy()
            if word in sensor_dict[row[1]].keys():
                sensor_dict[row[1]][word] += 1
        file_dict[file_name] = sensor_dict
        for key, value in sensor_dict.items():
            vector = dict()
            vector["file"] = file_name
            vector["sensor"] = int(key)
            vector.update(value)
            vectors.append(vector)


def calculations(directory, data_dict, data_df, all_words):
    total_gestures = len(data_dict)
    tf = all_words.copy()
    tf_idf = all_words.copy()
    vectors = list()
    for file_name, sensor_dict in data_dict.items():
        vector = list()
        vector.append(file_name)
        tf_vector = list()
        tf_idf_vector = list()
        for sensor, words_dict in sensor_dict.items():
            total_words = sum(words_dict.values())

            compute_tf_idf = data_df.loc[data_df['sensor'] == int(sensor)]
            num_of_docs_with_gesture = compute_tf_idf.astype(bool).sum(axis=0)

            for word, count in words_dict.items():
                tf[word] = count / total_words

                d_idf = float(num_of_docs_with_gesture[word])
                tf_idf[word] = float(tf[word]) * (math.log10(total_gestures / d_idf)) if d_idf > 0.0 else 0.0

            tf_vector.append(convert_vector_to_string(tf))
            tf_idf_vector.append(convert_vector_to_string(tf_idf))

            tf = dict.fromkeys(tf, 0)
            tf_idf = dict.fromkeys(tf_idf, 0)

        vector.append(" ".join(tf_vector))
        vector.append(" ".join(tf_idf_vector))
        vectors.append(vector)

    with open(directory + '/vectors.txt', 'w', newline="") as f:
        csv.writer(f, delimiter=',').writerows(vectors)
        f.close()


def convert_vector_to_string(data):
    data_str = [str(val) for val in list(data.values())]
    return " ".join(data_str)


def task0b(directory):
    all_words = get_all_words_from_directory(directory)
    print("Building all words dictionary")


    data_dict, data_df = create_word_dictionary(directory, all_words)
    print("Performing TF, TF-IDF, TF-IDF2 calculations")
    calculations(directory, data_dict, data_df, all_words)
    print("     ****Created vectors.txt file****")

