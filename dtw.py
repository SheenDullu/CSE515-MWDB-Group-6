import glob
import numpy as np
import re
import csv
import pandas as pd

def dynamicTimeWarping(directory,file):
    values={}
    base=fetchAQA(directory,file)

    compareKeys=glob.glob(directory+"/*.wrd")
    compareKeys.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.split('(\d+)',var)])
    cs_keys=base["cs"].unique()


    for key in compareKeys:


        compare=fetchAQA(directory,key.replace(directory+"\\",""))

        values[key]=0
        for cs_key in cs_keys:
            base_cs=base[base["cs"]==cs_key]["x"].to_numpy()
            compare_cs=compare[compare["cs"]==cs_key]["x"].to_numpy()
            baseSize=len(base_cs)-1
            compareSize=len(compare_cs)-1

            matrix = np.ones((baseSize+2, compareSize+2)) * -1

            matrix[1:, 0] = base_cs
            matrix[0, 1:] = compare_cs
            matrix[0,0] =0
            flag=0
            matrix[baseSize+1,compareSize+1]=DTWDist(base_cs,baseSize,compare_cs,compareSize,matrix,flag)
            values[key] += matrix[baseSize+1,compareSize+1]
        
    return values
    
    rankings={k:v for k,v in sorted(values.items(),key=lambda x:x[1])}
    output=list(rankings.items())
    print()
    print("---------------------------------")
    print("      File      |  DTW Distance  ")
    print("---------------------------------")
    for i in range(10):
        print(output[i][0]+"\t|"+"\t"+str(output[i][1]))

def fetchAQA(directory,file):
    values=[]
    file = open(directory+"/"+file,"r")
    rows = pd.read_csv(file,header=None,names=["c","s","w","t","a","d","x"])

    rows= rows[["c","s","x"]]
    rows["cs"]=rows.apply(lambda x: str(x['c'])+ str(x['s']),axis=1)

    return rows[["cs",'x']]

def DTWDist(seq1, seq1Index, seq2, seq2Index,matrix,flag):


    if(seq1Index==0 or seq2Index==0):
        return 0

    if(seq1[seq1Index]==seq2[seq2Index]):
        if(matrix[seq1Index,seq2Index]==-1):
            matrix[seq1Index,seq2Index]=DTWDist(seq1,seq1Index-1,seq2,seq2Index-1,matrix,1)
        return matrix[seq1Index,seq2Index]

    if(matrix[seq1Index,seq2Index]==-1):
        matrix[seq1Index,seq2Index]=DTWDist(seq1,seq1Index-1,seq2,seq2Index-1,matrix,1)

    if (matrix[seq1Index+1, seq2Index] == -1):
        if(flag==0):
            matrix[seq1Index+1, seq2Index] = DTWDist(seq1, seq1Index , seq2, seq2Index - 1,matrix,0)
        else:
            matrix[seq1Index + 1, seq2Index] = DTWDist(seq1, seq1Index, seq2, seq2Index - 1, matrix, 1)

    if (matrix[seq1Index, seq2Index+1] == -1):
        matrix[seq1Index, seq2Index+1] = DTWDist(seq1, seq1Index-1, seq2, seq2Index,matrix,1)

    if(flag):
        return  abs(seq1[seq1Index]-seq2[seq2Index])+min(matrix[seq1Index, seq2Index+1], matrix[seq1Index+1, seq2Index], matrix[seq1Index,seq2Index])
    else:
        return min(abs(seq1[seq1Index]-seq2[seq2Index])+matrix[seq1Index, seq2Index+1], matrix[seq1Index+1, seq2Index],abs(seq1[seq1Index]-seq2[seq2Index])+ matrix[seq1Index,seq2Index])

# # EXAMPLE RUN
# dynamicTimeWarping("3_class_gesture_data","270.wrd")
