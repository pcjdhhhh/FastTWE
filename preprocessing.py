# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 08:31:06 2023

@author: zhw
"""

import numpy as np
from get_data import *
from tools import *
import random
import math
from search_function import *
from aeon.distances import twe_distance

def vector_representation(train_data,index):
    #
    #reference_data = train_data[index,:]
    [n,original_dim] = train_data.shape
    reduced_dim = len(index)
    res_vector = np.zeros((n,reduced_dim))
    
    for i in range(n):
        for j in range(reduced_dim):
            res_vector[i,j] = twe_distance(train_data[i,:], train_data[index[j],:])
    return res_vector


def generate_with_random(n,reduced_dim):
    
    #
    
    numbers = np.arange(0, n)
    np.random.shuffle(numbers)
    random_numbers = numbers[0:reduced_dim]
    return random_numbers



    


def generate_with_furthest(train_data,reduced_dim,first_index):
    [n,dim] = train_data.shape
    pair_min = np.ones(n) * np.inf
    res_index = list()
    for i in range(reduced_dim):
        res_index.append(first_index)
        reference = train_data[first_index,:]
        for j in range(n):
            temp_dis = twe_distance(reference,train_data[j,:])
            if temp_dis<pair_min[j]:
                pair_min[j] = temp_dis
        #
        first_index = np.argsort(pair_min)[-1]
    return np.array(res_index)
        
            
            
            
            
        
    






    
    
    
    
    
    
    
    
    
    
    
    
    

    