import glob
import re

import numpy as np
import pandas as pd

import Utilities


import time

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

def editDistanceComp(matrix,p,q,P,Q,r,i,d):
    replaceCost = 1
    insertCost = 1
    deleteCost = 1


    if (p==0 or q==0):
        return  matrix[p,q]

    if(P[p]== Q[q]):
        if (matrix[p-1,q-1] == -1):
            matrix[p-1,q-1]=editDistanceComp(matrix,p-1,q-1,P,Q,r,i,d)
        return matrix[p-1,q-1]

    if(matrix[p,q-1]==-1):
        if(d>15):
            matrix[p,q-1]= 1000
        else :
            matrix[p,q-1]=editDistanceComp(matrix,p,q-1,P,Q,r,i,d+1)
    if(matrix[p-1,q-1]==-1):
        if(r>15):
            matrix[p-1,q-1]= 1000
        else :
            matrix[p-1,q-1]=editDistanceComp(matrix,p-1,q-1,P,Q,r+1,i,d)
    if(matrix[p-1,q]==-1):
        if(i>15):
            matrix[p-1,q]= 1000
        else :
            matrix[p-1,q]=editDistanceComp(matrix,p-1,q,P,Q,r,i+1,d)


    return min(matrix[p-1, q] + insertCost, matrix[p, q-1] + deleteCost, matrix[p-1, q-1] +replaceCost)

def editDistanceFunc(P,Q):

    if len(P) < len(Q):
        P, Q, = Q, P
    i = len(P)
    j = len(Q)

    matrix = np.ones((i, j))*-1

    matrix[:, 0] = np.arange(0,i)
    matrix[0,:] = np.arange(0,j)

    matrix[i-1,j-1] = editDistanceComp(matrix,i-1,j-1,P,Q,0,0,0)


    return matrix[i-1,j-1]

def editDistance(directory, file):
    wrdfile = Utilities.fetchWordsFromFile(directory, file)
    allwrdfiles = Utilities.fetchAllWordsFromFile(directory)

    all_files = glob.glob(directory + "/*.wrd")
    all_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])


    editValues = []
    for file in allwrdfiles:
        editVal = 0
        for key in wrdfile:
            editVal = 0
            numArr = []
            numArr2 = []
            for wrd in wrdfile[key]:
                numArr.extend(list(map(int, wrd.split(" "))))
            for wrd in file[key]:
                numArr2.extend(list(map(int, wrd.split(" "))))
            editVal += editDistanceFunc(numArr, numArr2)
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
            editDistance(r'C:\Class\CSE515 Multimedia\3_class_gesture_data', '1')
            print("########## Completed Edit Distance ##########")
        if task == 1:
            # dotProduct(directory, gesture, vector_model)
            dotProduct(r'D:\ASU\Courses\MWDB\Project\3_class_gesture_data', '570', 'tfidf')
            print("########## Completed Dot Product ##########")
        if task == 0:
            print("Thank you. Bye")
            break


if __name__ == '__main__':

    main()


    # start=time.time()
    # print(editDistance(r'C:\Users\Vccha\MWDB\CSE515-MWDB-Group-6\data','42'))
    # print(time.time()-start)




