import glob
import re

import pandas as pd

import Utilities
import dtw


def printTop10Values(similarity_measure):
    i = 1
    for index, row in similarity_measure.iteritems():
        print(i, '. Gesture ', index.split(".")[0].split('_')[-1], ", ", row)
        i = i + 1


def dotProduct(directory, file, model):
    gesture = directory + '\\' + model + '_vectors_' + file + '.txt'
    all_vectors = pd.DataFrame.from_dict(Utilities.getAllVectors(directory, model), orient='index')
    gesture_vector = pd.Series(Utilities.getAVector(gesture))
    dot_product = all_vectors.dot(gesture_vector)
    dot_product = dot_product.sort_values(ascending=False)
    printTop10Values(dot_product[:11])

#
# def editDistanceFunc(P, Q):
#     replaceCost = 1
#     insertCost = 1
#     deleteCost = 1
#     if len(P) < len(Q):
#         P, Q, = Q, P
#     i = len(P) + 1
#     j = len(Q) + 1
#     matrix = np.zeros((i, j))
#
#     matrix[:, 0] = np.arange(0,i)
#     matrix[0,:] = np.arange(0,j)
#
#     for x in range(1, i):
#         for y in range(1, j):
#             if P[x-1] == Q[y-1]:
#                 matrix[x, y] = matrix[x-1, y-1]
#             else:
#                 matrix[x, y] = min(matrix[x-1, y] + insertCost, matrix[x, y-1] + deleteCost, matrix[x-1, y-1] +
#                                    replaceCost)
#     return matrix[i - 1, j - 1]
def replace(s1,s2):
    s1=s1.split(' ')
    s2=s2.split(' ')

    num1=0
    num2=0
    for i in range(len(s1)):
        num1 = num1*10 + int(s1[i])
        num2 = num2*10 + int(s2[i])

    dif=abs(num1-num2)

    #Denominator has to be changed
    return (dif)/(666-111)

def editDistance(directory, file):
    wrdfile = Utilities.fetchWordsFromFile(directory, file)
    allwrdfiles = Utilities.fetchAllWordsFromFile(directory)

    all_files = glob.glob(directory + "/*.wrd")
    all_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])


    editValues = []
    counter=0
    important_sensors=set([5,6,7,8,9,10,11,12])
    for file in allwrdfiles:
        editVal = 0

        for key in wrdfile:

            # if (counter >30 and counter<60):

            numArr = []
            numArr2 = []
            for wrd in wrdfile[key]:

                numArr.append(wrd)
            for wrd in file[key]:
                numArr2.append(wrd)

            if (int(key[1]) in important_sensors):
                multiplier=2


            else:
                multiplier=0.5

            editVal += multiplier * Utilities.editDistanceFunc(numArr, numArr2)



        counter+=1
        editValues.append(editVal)

    vals = sorted(range(len(editValues)), key=lambda k: editValues[k])

    for i in range(0, 11):
        print(str(i+1) + " . Gesture " + all_files[vals[i]].split("\\")[-1].split(".")[0] + " , " + str(editValues[vals[i]]))

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
        if task == 6:
            editDistance(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data', '278')
            print("########## Completed Edit Distance ##########")
        if task == 7:
            dtw.dynamicTimeWarping(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data', '278')
            print("########## Completed DTW ##########")
        if task == 1:
            # dotProduct(directory, gesture, vector_model)
            dotProduct(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data', '570', 'tfidf')
            print("########## Completed Dot Product ##########")
        if task == 0:
            print("Thank you. Bye")
            break


if __name__ == '__main__':

    main()







