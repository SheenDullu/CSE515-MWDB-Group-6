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


def write_quantized_data(avg_std_df, quantized_data, directory, gesture, window_length, shift_length):
    file_name = gesture.split("\\")[-1].split(".")[0]
    folder_name = directory.split("\\")[-1]
    new_file = directory + "/" + file_name + ".wrd"
    vector = list()
    with open(new_file, 'w', newline="") as x:
        for index, row in quantized_data.iterrows():
            for i in range(0, quantized_data.shape[1], shift_length):
                if i + window_length < quantized_data.shape[1]:
                    win = row[i:i + window_length].tolist()
                    pair = [folder_name, file_name, index + 1, i, avg_std_df.iloc[index]['avg'],
                            avg_std_df.iloc[index]['std'], np.mean(win)]
                    vector.append(pair + win)
        csv.writer(x, delimiter=' ').writerows(vector)


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


def read_gestures_from_csv(all_files, directory, resolution, shift_length, window_length, bands):
    for filename in all_files:
        df = pd.read_csv(filename, header=None)
        column_names = [x for x in range(1, df.shape[1])]
        df = pd.DataFrame(df, columns=column_names)
        avg_std_df = calculate_avg_std_sensor_wise(df)
        df_norm = normalizeSensorWise(df)
        quantized_data = quantization(df_norm, bands)
        write_quantized_data(avg_std_df, quantized_data, directory, filename, window_length, shift_length)


def get_all_sub_folders(folder_directory):
    return [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]


def task0a(folder_directory, window_length, shift_length, resolution):
    dirs = get_all_sub_folders(folder_directory)
    print("Building Gaussian Bands...")
    bands = gaussian_bands(resolution)
    for folder in dirs:
        print("Processing for Folder ", folder, "...")
        all_files = glob.glob(folder_directory + '\\' + folder + "/*.csv")
        file_directory = folder_directory + '\\' + folder
        read_gestures_from_csv(all_files, file_directory, resolution, shift_length, window_length, bands)
        print("Created .wrd files for Folder ", folder)
    print("    ****Created dictionary for all the folders****")
