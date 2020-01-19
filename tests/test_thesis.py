from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import numpy as np
from dtaidistance import dtw, dtw_c, clustering,util,dtw_visualisation
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from sklearn.utils import shuffle
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import single
import array
import pytest
import pandas as pd
import math
import seaborn as sns
import time
from scipy.special import comb

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def effect_warping_window():
    '''
    Experiment to see effect of warping window on clustering using DTW dist
    :return:
    '''

    results_folder = "window/"
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    def frequency_count(x):
        unique, counts = np.unique(x, return_counts=True)

        return np.asarray((unique, counts)).T
    dataset_name = "UMD"
    dataset_folder = "../data/"
    # dataset_folder = "."
    # train_data, train_labels,test_data,test_labels = util.ucr_dataset_loader(dataset_name, dataset_folder, test_flag=True)
    # train_data = np.concatenate((train_data, test_data), axis=0)
    # train_labels = np.concatenate((train_labels, test_labels), axis=0)

    train_data, train_labels = util.ucr_dataset_loader(dataset_name, dataset_folder,test_flag=False)
    train_data,train_labels = shuffle(train_data ,train_labels, random_state=0)
    train_labels = train_labels.astype(np.int64)
    dtw_rand = []
    dtw_adj = []
    sill = []

    windows = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.8,1]
    start_time = time.time()
    for w in windows:
        window = int(np.floor(w*len(train_data[0])))
        m = dtw.distance_matrix_fast(train_data,window=window)
        i_lower = np.tril_indices(train_labels.size, -1)
        m[i_lower] = m.T[i_lower]
        np.fill_diagonal(m, 0)
        clus = len(np.unique(train_labels))
        model = AgglomerativeClustering(n_clusters=clus,linkage="average", affinity="precomputed")
        model.fit(m)
        dtw_rand.append(rand_index_score(model.labels_, train_labels))
        dtw_adj.append(adjusted_rand_score(model.labels_,train_labels))
        sill.append(silhouette_score(m,model.labels_,metric="precomputed"))

    print("time taken: ", (time.time() - start_time), "seconds")

    fig = plt.figure(figsize=(8, 4))
    # plt.gca().set_color_cycle(['blue', 'yellow'])
    print("sill",np.round(sill, 3))
    plt.scatter(windows,sill,c='blue')
    plt.title('Clustering performance DTW varying window\n'+dataset_name + ' Dataset')
    # plt.legend(['DTW','Euclidean'], loc='upper left')
    plt.ylabel('Silhouette Index')
    plt.xlabel('window')
    fig.savefig(results_folder+dataset_name + '_sill_window.png', bbox_inches='tight')
    plt.close(fig)  # close the figure


    print("wind",windows)
    fig = plt.figure(figsize=(8, 4))
    # plt.gca().set_color_cycle(['blue', 'yellow'])
    print("ran index ",np.round(dtw_rand,3))
    plt.scatter(windows,dtw_rand,c='blue')
    plt.title('Clustering performance DTW varying window\n'+dataset_name + ' Dataset')
    # plt.legend(['DTW','Euclidean'], loc='upper left')
    plt.ylabel('Rand Index')
    plt.xlabel('window')
    fig.savefig(results_folder+dataset_name + '_rand_window.png', bbox_inches='tight')
    plt.close(fig)  # close the figure

    fig = plt.figure(figsize=(8, 4))
    # plt.gca().set_color_cycle(['blue', 'yellow'])
    print("Adjusted rand index", np.round(dtw_adj,3))
    plt.scatter(windows, dtw_adj,c='blue')
    plt.ylabel('Adjusted Rand Index')
    plt.xlabel('windows')
    fig.savefig(results_folder+dataset_name + '_adj-rand_window.png', bbox_inches='tight')
    plt.close(fig)  # close the figure

def test_distance_pruned():
    dataset_name = "StarLightCurves"
    dataset_folder = "../data/"
    train_data, train_labels = util.ucr_dataset_loader(dataset_name, dataset_folder, test_flag=False)
    window = 100
    i = 10
    j = 3

    ## PRUNED FAST
    start_time = time.time()

    d= round(dtw_c.pruned_distance_nogil(train_data[i], train_data[j], window=window), 3)
    print("\nPruned Fast: ", (time.time() - start_time), "seconds")
    print("Pruned Fast Distance", d)

    ## DTW FAST
    start_time = time.time()
    d= round(dtw.distance_fast(train_data[i], train_data[j], window=window), 3)
    print("\nDTW Fast: ", (time.time() - start_time), "seconds")
    print("DTW Fast distance:",d)

def test_distance_matrix_pruned():

    dataset_name = "StarLightCurves"
    dataset_folder = "../data/"
    train_data, train_labels = util.ucr_dataset_loader(dataset_name, dataset_folder, test_flag=False)
    window = 100
    runs = 100
    # start_time = time.time()
    # m = dtw.distance_matrix(train_data[:runs],window = window,show_progress=True,use_c=True,use_nogil = True,parallel= False)
    # print("\nDTW Matrix Fast: ", (time.time() - start_time), "seconds")
    #
    n1 = 4
    n2 = 3
    pruned = []
    dtw_d = []
    dtw_fast = []
    pruned_fast = []

    #PRUNED
    # start_time = time.time()
    # for i in range(runs):
    #     for j in range(runs):
    #         pruned.append(round(dtw.pruned_distance(train_data[i], train_data[j],window=window),3))
    # print("\nPruned DTW: ", (time.time()-start_time), "seconds")

    ## PRUNED FAST
    start_time = time.time()
    for i in range(runs):
        for j in range(runs):
            pruned_fast.append(round(dtw_c.pruned_distance_nogil(train_data[i], train_data[j],window=window),3))
    print("\nPruned Fast DTW: ", (time.time() - start_time), "seconds")

    ## DTW FAST
    start_time = time.time()
    for i in range(runs):
        for j in range(runs):
            dtw_fast.append(round(dtw.distance_fast(train_data[i], train_data[j],window=window),3))
    print("\nDTW fast: ", (time.time()-start_time), "seconds")

    count = 0
    for i in range(len(dtw_fast)):
        if dtw_fast[i]!= pruned_fast[i]:
            if dtw_fast[i]-pruned_fast[i]<0:
                print("dtw:",dtw_fast[i],"pruned:",pruned_fast[i])
                count +=1
    print("count:",count)

    ## DTW
    # start_time = time.time()
    # for i in range(runs):
    #     for j in range(runs):
    #         dtw_d.append(round(dtw.distance(train_data[i], train_data[j],window=window),3))
    # print("\nDTW: ", (time.time() - start_time), "seconds")

def complexity_pruned(dataset_name,train_data,train_labels):
    '''
    Compare run time between dtw and pruned dtw for varying series len
    :param dataset_name:
    :param train_data:
    :param train_labels:
    :return:
    '''
    dtw_fast = []
    pruned_fast = []
    w = 0.1
    runs = len(train_data) #no of data points default: len(train_data)
    N_start = 0.1
    N_max = 1.1 # series len default: len(train_data[0])
    N_step = 0.2
    series_len = []
    for n_size in np.arange(N_start,N_max,N_step):
        n = int(n_size * len(train_data[0]))
        series_len.append(n)
        x = train_data[:, 0:n]
        print("COMP: "+dataset_name+" series of len",n," x shape:",x[:runs].shape)
        window = math.floor(len(x[0]) * w)
        ## PRUNED FAST
        start_time = time.time()
        m = dtw.distance_matrix(x[:runs],dist_type="pruned", window=window, show_progress=True, use_c=True, use_nogil=True,
                                parallel=False)
        pruned_fast.append(time.time() - start_time)
        # print(m)


        ## DTW FAST
        start_time = time.time()
        m = dtw.distance_matrix(x[:runs],dist_type="dtw",window=window, show_progress=True, use_c=True, use_nogil=True,
                                parallel=False)
        dtw_fast.append(time.time()-start_time)
    print("pruned:",pruned_fast)
    print("dtw_fast",dtw_fast)
    fig = plt.figure(figsize=(8, 4))
    # plt.gca().set_color_cycle(['blue', 'yellow'])
    plt.plot(series_len,dtw_fast, 'blue')
    plt.plot(series_len,pruned_fast, 'orange')
    plt.title('Runtime DTW vs Pruned DTW\n' + dataset_name + ' Dataset')
    plt.legend(['DTW', 'PrunedDTW'], loc='upper left')
    plt.ylabel('time (seconds)')
    plt.xlabel('series length')
    fig.savefig("comp/"+dataset_name + '_comp.png', bbox_inches='tight')
    plt.close(fig)  # close the figure
    df = pd.DataFrame({"series_len":series_len,"DTW": dtw_fast, "PrunedDTW": pruned_fast,"window%":np.full(shape=len(pruned_fast),fill_value=w,dtype=np.float)})
    df = df.round(3)
    df.to_csv("comp/"+dataset_name + "_comp.csv", index=False)

def perf_pruned_datasets(dataset_name,train_data,train_labels):
    '''
    Plots a time comparison between DTW and Pruned DTW with varying window size
    :param dataset_name:
    :param train_data:
    :param train_labels:
    :return:
    '''
    dataset_folder = "../data/"

    print("PERF: "+dataset_name+" shape of data used: n = ",len(train_data),", l = ",len(train_data[0]))
    dtw_fast = []
    pruned_fast = []
    runs = len(train_data)
    w_start = 0.1
    w_max = 0.6
    w_step = 0.05
    for w in np.arange(w_start,w_max,w_step):
        window = math.floor(len(train_data[0]) * w)
        print("perf window:",window)
        ## PRUNED FAST
        start_time = time.time()
        m = dtw.distance_matrix(train_data[:runs],dist_type="pruned", window=window, show_progress=True, use_c=True, use_nogil=True,
                                parallel=False)
        pruned_fast.append(time.time() - start_time)
        # print(m)


        ## DTW FAST
        start_time = time.time()
        m = dtw.distance_matrix(train_data[:runs],dist_type="dtw",window=window, show_progress=True, use_c=True, use_nogil=True,
                                parallel=False)
        dtw_fast.append(time.time()-start_time)
    print("pruned:",pruned_fast)
    print("dtw_fast",dtw_fast)
    fig = plt.figure(figsize=(8, 4))
    # plt.gca().set_color_cycle(['blue', 'yellow'])
    x = np.arange(w_start,w_max,w_step)
    plt.plot(x,dtw_fast, 'blue')
    plt.plot(x,pruned_fast, 'orange')
    plt.title('Runtime DTW vs Pruned DTW\n' + dataset_name + ' Dataset')
    plt.legend(['DTW', 'PrunedDTW'], loc='upper left')
    plt.ylabel('time (seconds)')
    plt.xlabel('window')
    fig.savefig("perf/"+dataset_name + '_perform.png', bbox_inches='tight')
    plt.close(fig)  # close the figure
    df = pd.DataFrame({"window":x,"DTW": dtw_fast, "PrunedDTW": pruned_fast})
    df = df.round(3)
    df.to_csv("perf/"+dataset_name + "_perform.csv", index=False)

def dtw_fastdtw_pruned_dtw(test_flag=False):
    '''
    Compare run time between dtw, pruned dtw and fastdtw
    :param test_flag:
    :return:
    '''
    dataset_folder = "../data/"
    datasets = ["StarLightCurves","ECG5000","NonInvasiveFetalECGThorax1","Mallat","HouseTwenty","GunPoint","ElectricDevices","Crop","Haptics","Car","Symbols","FreezerRegularTrain","DodgerLoopDay","UMD","FacesUCR","HandOutlines","HouseTwenty","PigAirwayPressure","Haptics","Phoneme","MixedShapesRegularTrain","Worms","ShapeletSim","Fish","Meat","Ham","OSULeaf","Yoga","Symbols","FaceFour","GunPoint","Plane","MedicalImages","ElectricDevices","Chinatown","SmoothSubspace"]

    pruned_fast = []

    dtw_fast = []
    fast_dtw = []
    series_len = []
    distances={"dp":[],"dw":[],"df":[]}
    dataset_folder = "../data/"
    for dataset_name in datasets:
        print(dataset_name)
        train_data, train_labels, test_data, test_labels = util.ucr_dataset_loader(dataset_name, dataset_folder,test_flag=True)
        train_data = np.concatenate((train_data, test_data), axis=0)
        train_labels = np.concatenate((train_labels, test_labels), axis=0)
        train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
        series_len.append(len(train_data[0]))

        temp = []
        n = len(train_data)
        for i in range(n):
            start_time = time.time()
            d = dtw_c.pruned_distance_nogil(train_data[i],train_data[i+1])
            temp.append(time.time() - start_time)
        pruned_fast.append(np.mean(temp))
        distances["dp"].append(d)

        ## DTW FAST
        temp = []
        for i in range(n):
            start_time = time.time()
            d = dtw.distance_fast(train_data[i],train_data[i+1])
            temp.append(time.time() - start_time)
        dtw_fast.append(np.mean(temp))
        distances["dw"].append(d)

        s1 = np.array(train_data[0])
        s2 = np.array(train_data[1])
        start_time=time.time()
        d,p = fastdtw(s1, s2,dist=euclidean)
        fast_dtw.append(time.time()-start_time)
        distances["df"].append(d)

    df = pd.DataFrame({"datasets":datasets,"series_len": series_len, "DTW": dtw_fast, "PrunedDTW": pruned_fast,"FastDTW":fast_dtw,"d_DTW": distances["dw"], "p_DTW":distances["dp"] , "f_DTW": distances["df"]})
    df['DTW'] = df['DTW'].apply(lambda x: x * 1000) #miliseconds
    df['PrunedDTW'] = df['PrunedDTW'].apply(lambda x: x * 1000)
    df['FastDTW'] = df['FastDTW'].apply(lambda x: x * 1000)
    df = df.round(3)
    df.to_csv("DTWvsPrunedvsFastDTW.csv", index=False)

def exp(test_flag = False):
    dataset_folder = "../data/"
    datasets = ["Car","UMD","OliveOil","FreezerRegularTrain","FiftyWords"]
    train_labels = None
    train_data = None
    for dataset_name in datasets:
        if test_flag:
            train_data, train_labels, test_data, test_labels = util.ucr_dataset_loader(dataset_name, dataset_folder,test_flag=True)
            train_data = np.concatenate((train_data, test_data), axis=0)
            train_labels = np.concatenate((train_labels, test_labels), axis=0)
        else:
            train_data, train_labels = util.ucr_dataset_loader(
            dataset_name, dataset_folder, test_flag = False)
        train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
        # perf_pruned_datasets(dataset_name, train_data, train_labels)
        complexity_pruned(dataset_name, train_data, train_labels)

if __name__ == "__main__":
    # exp(test_flag=False)
    effect_warping_window()