import pandas as pd
import numpy as np
import math
import kmeans

def task4a(directory):
    svd=pd.read_csv(directory,header=None,names=['features','file','score'])

    num_files=len(svd['file'].value_counts())
    num_features=len(svd['features'].value_counts())
    result=np.ones((1,num_files))
    for f in range(1,num_features+1):
        temp=(svd[svd['features']==f])
        temp=temp.sort_values(['file'])
        mean=temp['score'].mean()
        std=temp['score'].std()
        temp=temp.to_numpy()

        result+=(np.exp(-np.square(temp[:,2]-mean)/(2*std*std)))


    result=result.reshape((num_files,1))
    numbers=np.arange(1,num_files+1).reshape((num_files,1))
    result=np.concatenate([result,numbers],axis=1)
    result=(result[result[:,0].argsort()])

    return result

def task4c(directory):

    data = pd.read_csv(directory, header=None,
                       names=['features', 'file', 'score'])

    num_features=data['features'].max()
    for f in range(1, num_features+1):
        temp = data[data['features'] == f].sort_values(['file']).drop(['features', 'file'], axis=1).to_numpy()

        if (f == 1):
            result = temp
        else:
            result = np.concatenate([result, temp], axis=1)

    trial = kmeans.Cluster(result, 3, 1)
    k = trial.kmeans()

    print(k)



