# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:51:41 2025

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

random.seed(2025)
np.random.seed(2025)
now = datetime.now()
now_time_ = now.strftime("%Y-%m-%d-%H-%M-%S")
to_efficiency_file_name = 'efficiency_' + now_time_ + '.csv'
to_preprocessing_file_name = 'preprocessing_time_' + now_time_ + '.csv'

file_name_save = ['FreezerRegularTrain','FreezerSmallTrain']


reduced_dim = 8

for file_name in file_name_save:
    
    to_csv_efficiency_data = {file_name:['BFTWE','FastTWE','EucTWE','DSETWE','GLBTWE','FastTWE_no_LB']}
    to_csv_preprocessing_data = {file_name:['FastTWE','DSETWE']}
    to_csv_efficiency_data['search_time'] = [0,0,0,0,0,0]
    to_csv_preprocessing_data['preprocessing_time'] = [0,0]
    #file_name = 'ItalyPowerDemand'
    print('----------------file_name: ------------------',file_name)
    train_data,train_label,test_data,test_label = get_time_series(file_name)
    all_data = np.vstack((train_data,test_data))
    
    query_num = math.ceil(all_data.shape[0] * 0.01)
    
    test_data = all_data[0:query_num,:].astype('float32')
    train_data = all_data[query_num:,:].astype('float32')
    
    
    s_time= time.time()
    num_features = 10000
    parameters = fit(X=train_data, num_features=num_features)
    transformed_data = transform(X=train_data, parameters=parameters)
    sc = StandardScaler()
    #X_std = sc.fit_transform(transformed_data)
    X_std = sc.fit_transform(transformed_data)
    
    pca = PCA(n_components=reduced_dim).fit(X_std)
    train_rocket_pca = pca.transform(X_std)
    e_time = time.time()
    to_csv_preprocessing_data['preprocessing_time'][0] = e_time - s_time   
    
    
    s_time= time.time()
    transformed_test_data = transform(X=test_data, parameters=parameters)
    #sc = StandardScaler()
    test_data_std = sc.transform(transformed_test_data)
    test_rocket_pca = pca.transform(test_data_std)
    e_time = time.time()
    to_csv_efficiency_data['search_time'][1] = e_time - s_time   
    
    test_data = np.ascontiguousarray(all_data[0:query_num,:])
    train_data = np.ascontiguousarray(all_data[query_num:,:])
    first_index = 1
   
    #furthest_index = generate_with_furthest(train_data,reduced_dim,first_index)
    s_time = time.time()
    furthest_index = generate_with_random(train_data.shape[0],reduced_dim)
    furthest_train_vectors = vector_representation(train_data,furthest_index)  
    e_time = time.time()
    to_csv_preprocessing_data['preprocessing_time'][1] = e_time - s_time   
    
   
    s_time = time.time()
    furthest_test_vectors = np.zeros([test_data.shape[0],reduced_dim])
    for num_i in range(test_data.shape[0]):
        furthest_test_vectors[num_i,:] = [twe_distance(test_data[num_i,:], train_data[furthest_index[j_dim],:]) for j_dim in range(reduced_dim)]
    e_time = time.time()
    to_csv_efficiency_data['search_time'][3] = e_time - s_time
  
    print('train_data: ',train_data.shape)
    print('test_data: ',test_data.shape)
    
    candidate_num = 0.2 
    candidate_num = math.ceil(train_data.shape[0]*candidate_num)
    print('candidate_num: ',candidate_num)
    
    kk = [1]  
    for k in kk:
        print(k)
        
        s_time = time.time()
        brute_force_res = search_with_twed(train_data,test_data,k)
        e_time = time.time()
        to_csv_efficiency_data['search_time'][0] = e_time - s_time   
        
        s_time = time.time()
        lower_res = search_with_twed_lower_bound(train_data,test_data,k)
        e_time = time.time()
        to_csv_efficiency_data['search_time'][4] = e_time - s_time    
        
        s_time = time.time()
        res_rocket_lower_bound = search_with_filter_and_refine_twed_ROCKET_with_lower_bound(train_data,test_data,k,candidate_num,train_rocket_pca,test_rocket_pca)
        e_time = time.time()
        temp_time = to_csv_efficiency_data['search_time'][1]  
        to_csv_efficiency_data['search_time'][1] = to_csv_efficiency_data['search_time'][1] + e_time - s_time  
        
        s_time = time.time()
        res_rocket = search_with_filter_and_refine_twed_ROCKET(train_data,test_data,k,candidate_num,train_rocket_pca,test_rocket_pca)
        e_time = time.time()
        to_csv_efficiency_data['search_time'][5] = temp_time + e_time - s_time  
        
        s_time = time.time()
        res_pca = search_with_filter_and_refine_twed_ROCKET_with_lower_bound(train_data,test_data,k,candidate_num,train_data,test_data)
        e_time = time.time()
        to_csv_efficiency_data['search_time'][2] = to_csv_efficiency_data['search_time'][2] + e_time - s_time   #
        
        s_time = time.time()
        res_DSE = search_with_filter_and_refine_twed_ROCKET_with_lower_bound(train_data,test_data,k,candidate_num,furthest_train_vectors,furthest_test_vectors)
        e_time = time.time()
        to_csv_efficiency_data['search_time'][3] = to_csv_efficiency_data['search_time'][3] + e_time - s_time
        
        
    
  
        
    
    
