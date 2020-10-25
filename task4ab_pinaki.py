# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import glob
import pickle

import Utilities


def degree_of_membership(model,datadir):
    bins = dict()
    scores = model.components_
    scores = scores.tolist()
    all_files_objects = glob.glob(datadir + "\W" + "/*.csv")
    all_files_objects.sort(key=lambda x:int((x.split("\\")[-1]).split(".")[0]))
    store = list()
    print("----------------")
    for i in range(len(scores[0])):
        store = list()
        for j in range(len(scores)):
            store.append((scores[j][i],all_files_objects[i].split('\\')[-1],j+1))
        store.sort(key= lambda x:x[0])
        a,b,c = store[-1]
        if c not in bins:
            bins[c] = list()
            bins[c].append(b)
        else:
            bins[c].append(b)
    print(bins)

def main():
    bins = dict()
    datadir = Utilities.read_directory()
    user_input = input("Enter 1 for task 4a and 2 for task 4b: ")
    if user_input == '1':
        svd_reload = pickle.load(open(datadir + "\model_svd_task3.pkl",'rb'))
        degree_of_membership(svd_reload,datadir)
    elif user_input == '2':
        nmf_reload = pickle.load(open(datadir + "\model_nmf_task3.pkl",'rb'))
        degree_of_membership(nmf_reload,datadir)




    # old_data = old_data.values.tolist()
    # for i in range(len(old_data)):
    #     maxpos = old_data[i].index(max(old_data[i]))
    #     bin_number = maxpos+1
    #     if bin_number not in bins:
    #         bins[bin_number] = list()
    #         bins[bin_number].append("" + str(i+1) + ".csv")
    #     else:
    #         bins[bin_number].append("" + str(i+1) + ".csv")
    # print(bins)
    



if __name__ == '__main__':
    main()