import glob
import os

import numpy as np
import pandas as pd


def find_vec(loc, algo):
    vec = pd.DataFrame()

    all_files = glob.glob(os.path.join(loc) + "\\" + algo + "_vectors_*.txt")
    names = [x for x in range(1, 61)]
    for file in all_files:
        data = pd.read_csv(file)

        vec = pd.concat([vec, data], axis=1)
    vec.columns = names
    return vec


def dot(target, source):
    dot_product = list(np.sum(source * target, axis=0))
    sort = sorted(dot_product)
    result = []
    for i in range(len(sort) - 1, len(sort) - 11, -1):
        result.append(dot_product.index(sort[i]) + 1)
    return result


def main(data_loc, loc, algo, option):
    # data_loc=os.path.join(*loc.split('\\')[:-1])
    # print(data_loc)
    source = find_vec(data_loc, algo).to_numpy()

    target = pd.read_csv(loc).to_numpy()

    if (option == 1):
        print(dot(target, source))


if __name__ == '__main__':
    main(r"C:\Users\Vccha\MWDB\CSE515-MWDB-Group-6\data",
         r"C:\Users\Vccha\MWDB\CSE515-MWDB-Group-6\data\tf_vectors_1.txt", "tf", 1)
