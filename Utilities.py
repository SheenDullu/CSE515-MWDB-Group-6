import glob
import os


def get_all_sub_folders(folder_directory):
    return [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]


def getAllWords(directory):
    words = set()
    all_files = glob.glob(directory + "/*.wrd")
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                word = ' '.join(map(str, row[0:3]))
                words.add(word)
    return dict.fromkeys(sorted(words), 0)

ud = getAllWords('F:\mwdb\data')
lists = [k for k in ud.keys()]
print(len(lists))