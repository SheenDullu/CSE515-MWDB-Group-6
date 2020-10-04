from math import log
import pandas as pd
import numpy as np
import os
import sys
import glob

global num_files

def find_wrd(loc):

    f_wrd=pd.DataFrame()
    for dir in os.listdir(loc):
        if os.path.isdir(os.path.join(loc,dir)):
            all_files = glob.glob(os.path.join(loc,dir) + "/*.wrd")
            for file in all_files:

                data = pd.read_csv(os.path.join(loc,dir,file), delimiter=" ", header=None, dtype=str)
                f_wrd=pd.concat([f_wrd,data],axis=0)

    names=['component','f', 'sensor_id', 'time','avg','std','avg_quant','word']


    word=(f_wrd.apply(lambda x: ''.join(x[7:]),axis=1))
    # f_wrd['word']=f_wrd.apply(lambda x: str(''.join((x['word1'], x['word2'], x['word3']))), axis=1)
    # f_wrd=f_wrd.drop(['word1','word2','word3'],axis=1)
    f_wrd=f_wrd.drop(f_wrd.columns[7:],axis=1)
    f_wrd=pd.concat([f_wrd,word],axis=1)
    f_wrd.columns=names

    return f_wrd

def tf_f(data):
    # Total number of words recorded by a sensor
    word_sensor = (data['sensor_id'].value_counts()[0])

    # Count the frequency of words in a sensor's time series
    tf = pd.DataFrame(data.groupby(['component','f', 'sensor_id', 'word']).size() / word_sensor).reset_index()

    return tf


def idf_f(data):
    num_files = int(data['f'].max())
    # Check the number of files the word exists in w.r.t. sensor's time series
    count = data.groupby(['component','sensor_id', 'word', 'f']).count().reset_index()
    count['freq'] = count['time'].apply(lambda x: 1 if x > 0 else 0)
    idf = count.groupby(['component','sensor_id', 'word']).sum().reset_index()

    # Apply the idf formula
    idf['idf'] = idf['freq'].apply(lambda x: log(num_files / x))

    return idf


def vectors(input):

    data=input[['component','f', 'sensor_id', 'word','time']]
    # Create tf-idf-idf1
    tf = tf_f(data)
    idf = idf_f(data)

    # Combining tf and idf
    fin = pd.merge(tf, idf, left_on=['component','sensor_id', 'word'], right_on=['component','sensor_id', 'word'])[
        ['component','f', 'sensor_id', 'word', 0, 'idf']]

    # Finding max freq of a word in all time-series in all files
    # Finding the freq of word of a specific sensor in all files
    freq = data.groupby(['component','word', 'sensor_id']).count().reset_index()[['component','word', 'sensor_id', 'time']]
    # Finding max freq of the word in every file
    freq = freq[['word', 'time']].groupby(['word']).max()
    # Combine dfs
    fin = pd.merge(fin, freq, left_on=['word'], right_on=['word'])
    # Apply tf-idf formula
    fin['tf-idf'] = fin.apply(lambda x: (0.5 + 0.5 * (x[0] / x['time']) * x['idf']), axis=1)
    fin = fin.drop(['idf', "time"], axis=1)

    fin = fin[['component','f', 'sensor_id', 'word', 0, 'tf-idf']]
    fin.columns = ['component','f', 'sensor_id', 'word', 'tf', 'tf-idf']
    fin = fin.sort_values(['component','f', 'sensor_id', 'word'])
    return fin





def rearrange(zero, alg,num_files,comp):
    result = np.array([])

    # rearrange df to calculate for each algorithm
    for f in range(1, num_files + 1):
        temp = zero.filter(regex='^' + alg+ '_'+ '[X,Y,W,Z]'+ '_' + str(f) + '_', axis=1)
        temp_arr = temp.to_numpy()
        temp_arr = temp_arr.reshape((temp_arr.shape[0] * temp_arr.shape[1], 1), order='F')

        if (result.shape[0] < 5):
            result = temp_arr
        else:
            result = np.concatenate((result, temp_arr), axis=0)
    return result


def align(vector):
    # Partition based on file and outer join to standardise vector for each file
    comp= vector["component"].unique()

    flag=1
    num_files=vector['f'].astype(int).max()

    comp_df=vector[vector["component"]==comp[0]].drop(["component"],axis=1)

    for c in range(1,len(comp)):
        temp=vector[vector["component"]==c].drop(['component'],axis=1)
        comp_df=pd.merge(comp_df,temp, on=['f','sensor_id', 'word'], how='outer', suffixes=('_' + str(comp[c - 1]), '_' + str(comp[c])))

    df = comp_df[comp_df['f'] == '1'].drop(['f'], axis=1)


    for f in range(2, num_files + 1):
        temp = comp_df[comp_df['f'] == str(f)].drop(['f'], axis=1)
        df = pd.merge(df, temp, on=['sensor_id', 'word'], how='outer', suffixes=('_' + str(f - 1), '_' + str(f)))
    df = df.fillna(-1)

    # Partition file based standardised vector based on sensor and outer join to standardise every sensor vector
    zero = df[df['sensor_id'] == '0'].drop(['sensor_id'], axis=1)

    for i in range(1, 20):
        temp = df[df['sensor_id'] == str(i)].drop(['sensor_id'], axis=1)
        zero = pd.merge(zero, temp, on=['word'], how='outer', suffixes=('_' + str(i - 1), '_' + str(i)))


    zero = zero.fillna(-1)
    zero = zero.sort_values('word')

    for c in comp:
        comp_alg=zero.filter(regex='_' + str(c) + '_' , axis=1)


        tf = rearrange(comp_alg, 'tf',num_files,comp)
        idf = rearrange(comp_alg, 'tf-idf',num_files,comp)

        if(flag):
            ulti_tf=tf
            ulti_idf=idf
            flag=0
        else:

            ulti_tf=np.concatenate((ulti_tf, tf), axis=0)
            ulti_idf = np.concatenate((ulti_idf, idf), axis=0)



    return ulti_tf, ulti_idf





def write(vec,file,loc):
    vec.to_csv(os.path.join('\\'.join(loc.split('\\')[:-1]),file), index=False,header=None)




def Task0b(loc):

    data=find_wrd(loc)



    print("Data Loaded")

    vec = vectors(data)

    print("Values Calculated")


    tf, tf_idf = align(vec)

    print("Vectors Standardized")
    print("writing to file...")

    file_names=["tf_vectors_fi.txt","tf_idf_vectors_fi.txt"]

    write(pd.DataFrame(tf),file_names[0],loc)
    write(pd.DataFrame(tf_idf), file_names[1],loc)


if __name__=='__main__':

    Task0b(r"C:\Users\Vccha\MWDB\CSE515-MWDB-Group-6\data")



