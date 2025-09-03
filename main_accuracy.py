# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:33:42 2025

@author: zhw
"""

from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import StandardScaler
from get_data import *
import numpy as np
from get_data import *
from tools import *
import random
import math
from twed_search_function import *
from preprocessing import *
import time
import os
from numba import njit, prange, vectorize
import numpy as np
from MiniROCKET_ import *
import pandas as pd
from datetime import datetime
#from umap import UMAP

random.seed(2025)
np.random.seed(2025)
now = datetime.now()
now_time_ = now.strftime("%Y-%m-%d-%H-%M-%S")
to_file_name = 'twed_similarity_search_' + now_time_ + '.csv'

file_name_save = ['FreezerRegularTrain','FreezerSmallTrain']

#file_name_save = ['CBF','DiatomSizeReduction']
    
average_k = dict()
average_k['k=1'] = [0,0,0,0,0]
average_k['k=5'] = [0,0,0,0,0]
average_k['k=10'] = [0,0,0,0,0]
average_k['k=20'] = [0,0,0,0,0]
average_k['k=50'] = [0,0,0,0,0]
reduced_dim = 8
for file_name in file_name_save:
    #file_name = 'ItalyPowerDemand'
    print('----------------file_name: ------------------',file_name)
    train_data,train_label,test_data,test_label = get_time_series(file_name)
    all_data = np.vstack((train_data,test_data))
    
    query_num = math.ceil(all_data.shape[0] * 0.01)
    
    test_data = all_data[0:query_num,:].astype('float32')
    train_data = all_data[query_num:,:].astype('float32')

    num_features = 10000
    parameters = fit(X=train_data, num_features=num_features)
    transformed_data = transform(X=train_data, parameters=parameters)
    sc = StandardScaler()
    #X_std = sc.fit_transform(transformed_data)
    X_std = sc.fit_transform(transformed_data)
    
    pca = PCA(n_components=reduced_dim).fit(X_std)
    train_rocket_pca = pca.transform(X_std)

    transformed_test_data = transform(X=test_data, parameters=parameters)
    #sc = StandardScaler()
    test_data_std = sc.transform(transformed_test_data)
    test_rocket_pca = pca.transform(test_data_std)

    only_pca_sc = StandardScaler()   
    X_std_only_pca = only_pca_sc.fit_transform(train_data)
    only_pca = PCA(n_components=reduced_dim).fit(X_std_only_pca)
    train_only_pca = only_pca.transform(X_std_only_pca)
 
    test_only_pca_std = only_pca_sc.transform(test_data)
    test_only_pca = only_pca.transform(test_only_pca_std)

    test_data = np.ascontiguousarray(all_data[0:query_num,:])
    train_data = np.ascontiguousarray(all_data[query_num:,:])
    first_index = 1
    
    #furthest_index = generate_with_furthest(train_data,reduced_dim,first_index)
    furthest_index = generate_with_random(train_data.shape[0],reduced_dim)
    furthest_train_vectors = vector_representation(train_data,furthest_index)  #这一步需要花销挺久的
    
    
    furthest_test_vectors = np.zeros([test_data.shape[0],reduced_dim])
    for num_i in range(test_data.shape[0]):
        furthest_test_vectors[num_i,:] = [twe_distance(test_data[num_i,:], train_data[furthest_index[j_dim],:]) for j_dim in range(reduced_dim)]

    print('train_data: ',train_data.shape)
    print('test_data: ',test_data.shape)

    candidate_num = 0.2 
    candidate_num = math.ceil(train_data.shape[0]*candidate_num)
    print('candidate_num: ',candidate_num)

    k=50
    KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(50)
    if os.path.exists(KNN_file_path):
        print('50-NN exists')
        #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
        brute_force_res_50 = np.loadtxt(KNN_file_path)
        brute_force_res_50 = brute_force_res_50.reshape((len(brute_force_res_50),-1))
        brute_force_res_50.astype('int')
    else:
        print('50-NN no exist')
        s = time.time()
        brute_force_res_50 = search_with_twed(train_data,test_data,k)
        brute_force_res_50.astype('int')
        e = time.time()
        print('brute-force time: ',e-s)
        np.savetxt(KNN_file_path,brute_force_res_50)

    kk=[1,5,10,20,50]

    to_csv_data = {file_name:['rocket','only_pca','DSE','random','original']}
    for k in kk:
        print(k)
        
        key_ = 'k=' + str(k)
        to_csv_data[key_] = [0,0,0,0,0]
        average_str = 'k=' + str(k)
        #average_k[average_str] += compute_overlapping(brute_force_res,res_rocket)
        
        
        KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(k)
        if os.path.exists(KNN_file_path):
            print(str(k) + '_exists')
            #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
            brute_force_res = np.loadtxt(KNN_file_path)
            brute_force_res = brute_force_res.reshape((len(brute_force_res),-1))
        else:
            print(str(k) + '_no exist')
            s = time.time()
            brute_force_res = search_with_twed_from_first_K(train_data,test_data,k,brute_force_res_50)
            e = time.time()
            #print('brute-force time: ',e-s)
            np.savetxt(KNN_file_path,brute_force_res)

        s_time = time.time()
        res_rocket = search_with_filter_and_refine_twed_ROCKET(train_data,test_data,k,candidate_num,train_rocket_pca,test_rocket_pca)
        e_time = time.time()
        #print(e_time-s_time)
        print('rocket accuracy: ',compute_overlapping(brute_force_res,res_rocket))
        to_csv_data[key_][0] = compute_overlapping(brute_force_res,res_rocket)
        average_k[average_str][0] += compute_overlapping(brute_force_res,res_rocket)
        
        s_time = time.time()
        res_pca = search_with_filter_and_refine_twed_ROCKET(train_data,test_data,k,candidate_num,train_only_pca,test_only_pca)
        e_time = time.time()
        #print(e_time-s_time)
        print('only pca accuracy: ',compute_overlapping(brute_force_res,res_pca))
        to_csv_data[key_][1] = compute_overlapping(brute_force_res,res_pca)
        average_k[average_str][1] += compute_overlapping(brute_force_res,res_pca)
        
        s_time = time.time()
        res_DSE = search_with_filter_and_refine_twed_ROCKET(train_data,test_data,k,candidate_num,furthest_train_vectors,furthest_test_vectors)
        e_time = time.time()
        #print(e_time-s_time)
        print('DSE accuracy: ',compute_overlapping(brute_force_res,res_DSE))
        to_csv_data[key_][2] = compute_overlapping(brute_force_res,res_DSE)
        average_k[average_str][2] += compute_overlapping(brute_force_res,res_DSE)
        
        s_time = time.time()
        res_random = search_with_random_filter_and_refine_twed(train_data,test_data,k,candidate_num)
        e_time = time.time()
        #print(e_time-s_time)
        print('Random accuracy: ',compute_overlapping(brute_force_res,res_random))
        to_csv_data[key_][3] = compute_overlapping(brute_force_res,res_random)
        average_k[average_str][3] += compute_overlapping(brute_force_res,res_random)
        
        s_time = time.time()
        res_random = search_with_filter_and_refine_twed_ROCKET(train_data,test_data,k,candidate_num,train_data,test_data)
        e_time = time.time()
        #print(e_time-s_time)
        print('Original using Euclidean distance accuracy: ',compute_overlapping(brute_force_res,res_random))
        to_csv_data[key_][4] = compute_overlapping(brute_force_res,res_random)
        average_k[average_str][4] += compute_overlapping(brute_force_res,res_random)
     
    df = pd.DataFrame(to_csv_data)
    
    df.T.to_csv(to_file_name,mode='a')

average_k['k=1'] = np.array(average_k['k=1'])/len(file_name_save)
average_k['k=5'] = np.array(average_k['k=5'])/len(file_name_save)
average_k['k=10'] = np.array(average_k['k=10'])/len(file_name_save)
average_k['k=20'] = np.array(average_k['k=20'])/len(file_name_save)
average_k['k=50'] = np.array(average_k['k=50'])/len(file_name_save)

print(average_k)