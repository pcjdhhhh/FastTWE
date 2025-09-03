# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 17:42:40 2025

@author: zhw
"""

from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from get_data import *
from tools import *
import random
from numba import njit, prange, vectorize
import numpy as np
from MiniROCKET_ import *
from aeon.distances import twe_distance
from lower_bound_function import *


def search_with_random_filter_and_refine_twed(train_data,test_data,k,candidate_num):
    
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    for i in range(n_test):
        query = test_data[i,:]
        #candidate_index
        candidate_index = random.sample(range(0,n_train),candidate_num)
        #refine
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            temp = twe_distance(query,train_data[candidate_index[j],:])
            min_ = max(min_k)
            if temp<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = candidate_index[j]   
                min_k[location] = temp
    return res

def search_with_twed(train_data,test_data,k):
    
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf   
        for j in range(n_train):
            temp = twe_distance(query,train_data[j,:])
            min_ = max(min_k)
            if temp<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = j   #
                min_k[location] = temp
    return res


def search_with_twed_from_first_K(train_data,test_data,k,first_K):
    
    n_test = test_data.shape[0]
    len_first_K = first_K.shape[1]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf    
        for j in range(len_first_K):
            temp_MSM = twe_distance(query,train_data[int(first_K[i,j]),:])
            min_ = max(min_k)
            if temp_MSM<min_:
                location = np.where(min_k==min_)[0][0]  
                res[i,location] = int(first_K[i,j])   
                min_k[location] = temp_MSM
    return res


def search_with_twed_lower_bound(train_data,test_data,k):
    
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf    
        for j in range(n_train):
            min_ = max(min_k)
            #首先计算lower_bound
            lower_bound = glb_twed(query,train_data[j,:])
            if lower_bound<min_:
                temp = twe_distance(query,train_data[j,:])
                if temp<min_:
                    location = np.where(min_k==min_)[0][0]   
                    res[i,location] = j   #
                    min_k[location] = temp
    return res

def search_with_filter_and_refine_twed_ROCKET(train_data,test_data,k,candidate_num,train_vectors,test_vectors):
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        query = test_data[i,:]
        vector_query = test_vectors[i,:]
        vector_dis = np.array([compute_ED(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        
        
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            temp = twe_distance(query,train_data[candidate_index[j],:])
            min_ = max(min_k)
            if temp<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = candidate_index[j]   
                min_k[location] = temp
    return res


def search_with_filter_and_refine_twed_ROCKET_with_lower_bound(train_data,test_data,k,candidate_num,train_vectors,test_vectors):
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        query = test_data[i,:]
        vector_query = test_vectors[i,:]
        vector_dis = np.array([compute_ED(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        
        #
        min_k = np.ones(k) * np.inf    #
        for j in range(candidate_num):
            min_ = max(min_k)
            
            lower_bound = glb_twed(query,train_data[candidate_index[j],:])
            if lower_bound<min_:
                
                temp = twe_distance(query,train_data[candidate_index[j],:])
                if temp<min_:
                    location = np.where(min_k==min_)[0][0]   
                    res[i,location] = candidate_index[j]   
                    min_k[location] = temp
    return res