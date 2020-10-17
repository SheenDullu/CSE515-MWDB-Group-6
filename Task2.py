import pandas as pd

import Utilities


def printTop10Values(similarity_measure):
    i = 1
    for index, row in similarity_measure.iteritems():
        print(i, '. Gesture ', index.split("_")[2].split(".")[0], ", ", row)
        i = i + 1


def dotProduct(directory, file, model):
    gesture = directory + '\\' + model + '_vectors_' + file + '.txt'
    all_vectors = pd.DataFrame.from_dict(Utilities.getAllVectors(directory, model), orient='index')
    gesture_vector = pd.Series(Utilities.getAVector(gesture))
    dot_product = all_vectors.dot(gesture_vector)
    dot_product = dot_product.sort_values(ascending=False)
    printTop10Values(dot_product[:11])


def main():
    while True:
        # print("Top 10 most Similar Gesture")
        # directory = input("Enter the directory containing all the components, words and vectors: \n")
        # gesture = input("Enter gesture number\n")
        # vector_model = input("Enter the vector model you want to use \nEnter tf or tfidf: ")
        # print('Enter 1 for Dot Product')
        # print('Enter 2 for PCA')
        # print('Enter 3 for SVD')
        # print('Enter 4 for NMF')
        # print('Enter 5 for LDA')
        # print('Enter 6 for Edit Distance')
        # print('Enter 7 for DTW')
        task_input = input("What Task  do you want to perform: (enter 0 to exit)\n")
        task = int(task_input)
        if task == 1:
            # dotProduct(directory, gesture, vector_model)
            dotProduct(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data', '1', 'tf')
            print("########## Completed Dot Product ##########")
        if task == 0:
            print("Thank you. Bye")
            break


if __name__ == '__main__':
    main()
