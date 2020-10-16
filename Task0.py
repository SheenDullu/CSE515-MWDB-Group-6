import csv
import glob
import math

import numpy as np
import pandas as pd
import scipy.stats
from scipy.integrate import quad

import Utilities


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


def create_word_dictionary(avg_std_df, df_norm, quantized_data, directory, window_length, shift_length):
    folder_name = directory.split("\\")[-1]
    word_list = list()
    for index, row in quantized_data.iterrows():
        for i in range(0, quantized_data.shape[1], shift_length):
            if i + window_length < quantized_data.shape[1]:
                avg_q = df_norm.loc[index][i:i + window_length].tolist()
                win = row[i:i + window_length].tolist()
                pair = [folder_name, index + 1, ' '.join(map(str, win)), i, avg_std_df.iloc[index]['avg'],
                        avg_std_df.iloc[index]['std'], np.mean(avg_q)]
                word_list.append(pair)
    return word_list


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
        quantized_data = quantization(df_norm.copy(), bands)
        word_dict[file_name].extend(create_word_dictionary(avg_std_df, df_norm, quantized_data, directory,
                                                           window_length, shift_length))


def task0a(folder_directory, window_length, shift_length, resolution):
    dirs = Utilities.get_all_sub_folders(folder_directory)
    print("Building Gaussian Bands...")
    bands = gaussian_bands(resolution)
    print("Done!")
    word_dict = dict()
    for folder in dirs:
        print("Processing for Folder ", folder, "...")
        all_files = glob.glob(folder_directory + '\\' + folder + "/*.csv")
        file_directory = folder_directory + '\\' + folder
        read_gestures_from_csv(all_files, file_directory, shift_length, window_length, bands, word_dict)
        print("Done!")

    for key, value in word_dict.items():
        word_file = folder_directory + '\\' + str(key) + '.wrd'
        with open(word_file, 'w', newline="") as f:
            csv.writer(f, delimiter=',').writerows(value)
            f.close()
    print("    ****Created dictionaries(.wrd) for all the gesture files****")


def parse_and_store_file_data(file_dict, file, all_words, vectors):
    file_name = file.split("\\")[-1].split(".")[0]
    with open(file, 'r') as f:
        if file_name not in file_dict.keys():
            file_dict[file_name] = all_words.copy()
        for line in f:
            row = line.strip().split(',')
            word = ' '.join(map(str, row[0:3]))
            if word in file_dict[file_name].keys():
                file_dict[file_name][word] += 1
    vector = dict()
    vector["file"] = file_name
    vector.update(file_dict[file_name])
    vectors.append(vector)


def fill_word_dictionary(directory, all_words):
    all_files = glob.glob(directory + "/*.wrd")
    file_dict = dict()
    vectors = list()
    for file in all_files:
        parse_and_store_file_data(file_dict, file, all_words, vectors)
    df = pd.DataFrame(vectors)
    return file_dict, df


def calculations(directory, data_dict, data_df, all_words):
    # to keep track of the unique words
    with open(directory + '/header.txt', 'w', newline="") as f:
        csv.writer(f).writerow(list(all_words.keys()))
        f.close()

    tf = all_words.copy()
    tf_idf = all_words.copy()

    for file_name, words_dict in data_dict.items():
        tf_vector = list()
        tf_idf_vector = list()
        total_words = sum(words_dict.values())

        num_of_words_in_gesture = data_df.astype(bool).sum(axis=0)
        for word, count in words_dict.items():
            tf[word] = count / total_words

            d_idf = float(num_of_words_in_gesture[word])
            tf_idf[word] = float(tf[word]) * (math.log10(len(data_df.columns) / d_idf)) if d_idf > 0.0 else 0.0

        tf_vector.append(tf.values())
        tf_idf_vector.append(tf_idf.values())

        tf = dict.fromkeys(tf, 0)
        tf_idf = dict.fromkeys(tf_idf, 0)

        writeToFile(directory + '/tf_vectors_' + file_name + '.txt', tf_vector)
        writeToFile(directory + '/tfidf_vectors_' + file_name + '.txt', tf_idf_vector)


def writeToFile(file_name, data):
    with open(file_name, 'w', newline="") as f:
        csv.writer(f).writerows(data)
        f.close()


def task0b(directory):
    all_words = Utilities.getAllWords(directory)
    print("Building all words dictionary")
    data_dict, data_df = fill_word_dictionary(directory, all_words)
    print("Performing TF, TF-IDF calculations")
    calculations(directory, data_dict, data_df, all_words)
    print("     ****Created .txt files****")


if __name__ == '__main__':
    # task0a(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data', 3, 2, 4)
    task0b(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data')
