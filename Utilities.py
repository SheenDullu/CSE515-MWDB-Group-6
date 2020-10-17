import glob
import os


def get_all_sub_folders(folder_directory):
    return [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]


def fetchAllWordsFromDictionary(directory):
    words = set()
    all_files = glob.glob(directory + "/*.wrd")
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                word = ' '.join(map(str, row[0:3]))
                words.add(word)
    return dict.fromkeys(sorted(words), 0)


def getAllUniqueWords(directory):
    filename = directory + r'\header.txt'
    file = open(filename, 'r')
    s = file.read()
    return s.split(',')


def getAllVectors(directory, model):
    vectors = dict()
    all_files = glob.glob(directory + "/" + model + "_vectors_*.txt")
    all_files.sort(key=lambda x: int((x.split("_")[2]).split(".")[0]))
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                vector = [float(i) for i in row]
                vectors[filename] = vector
    return vectors


def getAVector(file):
    # file_name = file.split("\\")[-1].split(".")[0]
    file = open(file, 'r')
    f = file.read()
    vector_string = f.split(',')
    vector = [float(i) for i in vector_string]
    return vector


if __name__ == '__main__':
    # getAllUniqueWords(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data')
    # getAllVectors(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data', 'tf')
    getAVector(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data\tf_vectors_1.txt')
