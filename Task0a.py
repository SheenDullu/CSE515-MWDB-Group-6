import csv
import glob
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


def create_word_dictionary(avg_std_df, df, quantized_data, directory, window_length, shift_length):
    folder_name = directory.split("\\")[-1]
    word_list = list()
    for index, row in quantized_data.iterrows():
        for i in range(0, quantized_data.shape[1], shift_length):
            if i + window_length < quantized_data.shape[1]:
                avg_q = df.loc[index][i:i + window_length].tolist()
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
    for file in all_files:
        file_name = file.split("\\")[-1].split(".")[0]
        if file_name not in word_dict.keys():
            word_dict[file_name] = list()
        df = pd.read_csv(file, header=None)
        column_names = [x for x in range(1, df.shape[1])]
        df = pd.DataFrame(df, columns=column_names)
        avg_std_df = calculate_avg_std_sensor_wise(df)
        df_norm = normalizeSensorWise(df)
        quantized_data = quantization(df_norm.copy(), bands)
        word_dict[file_name].extend(create_word_dictionary(avg_std_df, df_norm, quantized_data, directory,
                                                           window_length, shift_length))


def get_all_sub_folders(folder_directory):
    return [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]


def task0a(folder_directory, window_length, shift_length, resolution):
    dirs = get_all_sub_folders(folder_directory)
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
            csv.writer(f, delimiter=' ').writerows(value)
            f.close()
    print("    ****Created dictionaries(.wrd) for all the gesture files****")
    # wrd = folder_directory + '\\1.wrd'
    # df = pd.read_csv(wrd, sep=' ')
