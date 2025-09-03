# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:13:43 2024

@author: Haowen Zhang
"""
from math import floor, sqrt
import numpy as np
from numba import jit


def glb_msm(x, y, c=0.5):
    leny = len(y)
    lenx = len(x)
    XUE = max(x)
    XLE = min(x)
    YUE = max(y)
    YLE = min(y)

    if y[leny-2]>=y[leny-1]>=x[lenx-1] or y[leny-2]<=y[leny-1]<=x[lenx-1] or x[lenx-2]<=x[lenx-1]<=y[leny-1] or x[lenx-2]>=x[lenx-1]>=y[leny-1]:
        fixed_dist = abs(x[0]-y[0]) + min(abs(x[lenx-1]-y[leny-1]), c)
    else:
        fixed_dist = abs(x[0]-y[0]) + min(
                                        abs(x[lenx-1]-y[leny-1]),
                                        c + abs(y[leny-1] - y[leny-2]),
                                        c + abs(x[lenx-1] - x[lenx-2]))

    y_dist = 0
    for i in range(1, leny-1):

        if y[i] > XUE:
            y_dist += min(abs(y[i]-XUE), c)
        if y[i] < XLE:
            y_dist += min(abs(y[i]-XLE), c)
        
    x_dist = 0
    for i in range(1, lenx-1):

        if x[i] > YUE:
            x_dist += min(abs(x[i]-YUE), c)
        if x[i] < YLE:
            x_dist += min(abs(x[i]-YLE), c)

    lb_dist = fixed_dist + max(y_dist, x_dist)

    return lb_dist

def edr_subcost(x, y, epsilon=0.1):
    if abs(x-y) <= epsilon:
        cost = 0
    else:
        cost = 1
    return cost

def lcss_subcost(x, y, epsilon=0.2):
    if abs(x-y) <= epsilon: 
        r = 1
    else:
        r = 0
    return r

def glb_erp(y, x, m=0): # GLB_ERP
    #glb_erp(x,y)==glb_erp(y,x)  
    lenx = len(x)
    leny = len(y)
    XUE = max(x)
    XLE = min(x)
    YUE = max(y)
    YLE = min(y)
    
    fixed_dist = min((x[lenx-1] - y[leny-1])**2, (x[lenx-1]-m)**2, (y[leny-1]- m)**2)

    y_dist = 0
    for i in range(1, leny-1):

    
        if y[i] > XUE:
            y_dist += min((y[i]-XUE)**2, (y[i]-m)**2)
        elif y[i] < XLE:
            y_dist += min((y[i]-XLE)**2, (y[i]-m)**2)
    
    x_dist = 0
    for i in range(1, lenx-1):

        if x[i] > YUE:
            x_dist += min((x[i]-YUE)**2, (x[i]-m)**2)
        elif x[i] < YLE:
            x_dist += min((x[i]-YLE)**2, (x[i]-m)**2)

    lb_dist = fixed_dist + max(x_dist, y_dist)
    
    return sqrt(lb_dist)

def glb_dtw(y, x):
    lenx = len(x)
    leny = len(y)
    XUE = max(x)
    XLE = min(x)
    YUE = max(y)
    YLE = min(y)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE:
            y_dist += (y[i] - XUE) **2
        if y[i] < XLE:
            y_dist += (y[i] - XLE) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE:
            x_dist += (x[i] - YUE) **2
        if x[i] < YLE:
            x_dist += (x[i] - YLE) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)


def glb_lcss(y, x, epsilon=0.2):
    lenx = len(x)
    leny = len(y)
    XUE = max(x)
    XLE = min(x)
    YUE = max(y)
    YLE = min(y)
    fixed_sum = lcss_subcost(x[0], y[0], epsilon) + lcss_subcost(x[lenx-1], y[leny-1], epsilon)
    
    XLE_lower = XLE-epsilon
    XUE_higher = XUE+epsilon
    
    y_reward = 0
    
    for i in range(1, leny-1):
    
        if y[i] >= XLE_lower and y[i] <= XUE_higher:
            y_reward += 1
    
    YLE_lower = YLE-epsilon
    YUE_higher = YUE+epsilon
    x_reward = 0
   
    for i in range(1, lenx-1):
       
        if x[i] >= YLE_lower and x[i] <= YUE_higher:
            x_reward += 1
    
    sum = fixed_sum + min(y_reward, x_reward)
    lb_dist = 1 - (sum/(min(len(x),len(y))))
    
    return lb_dist

def glb_edr(x, y, epsilon=0.1):
    lenx = len(x)
    leny = len(y)
    XUE = max(x)
    XLE = min(x)
    YUE = max(y)
    YLE = min(y)
    fixed_cost = 0 + min(edr_subcost(x[lenx-1], y[leny-1], epsilon), 1)
    y_dist = 0
    for i in range(1, leny-1):
        if y[i] > XUE:
            y_dist += edr_subcost(y[i], XUE, epsilon)
        if y[i] < XLE:
            y_dist += edr_subcost(y[i], XLE, epsilon)
    x_dist = 0
    for i in range(1, lenx-1):
        if x[i] > YUE:
            x_dist += edr_subcost(x[i], YUE, epsilon)
        if x[i] < YLE:
            x_dist += edr_subcost(x[i], YLE, epsilon)

    lb_dist = fixed_cost + max(x_dist, y_dist)

    return lb_dist

@jit(nopython=True)
def glb_twed(x, y, lamb=1):
    
    XUE = np.array([max(x)] * len(x))
    XLE = np.array([min(x)] * len(x))
    YUE = np.array([max(y)] * len(y))
    YLE = np.array([min(y)] * len(y))

    leny = len(y)
    lenx = len(x)

    fixed_dist = (x[0]-y[0])**2 + min(
                                    (x[lenx-1]-y[leny-1])**2,
                                    (y[lenx-1]-y[lenx-2])**2+lamb,
                                    (x[lenx-1]-x[lenx-2])**2+lamb
                                    )

    
    y_dist = 0
    for i in range(1, leny-1):

        if y[i]>=XUE[i] and y[i-1]>=XUE[i]:
            y_dist += min(((y[i]-XUE[i])**2 + (y[i-1]-XUE[i])**2), (y[i]-y[i-1])**2+lamb)
        if y[i]<=XLE[i] and y[i-1]<=XLE[i]:
            y_dist += min((y[i]-XLE[i])**2 + (y[i-1]-XLE[i])**2, (y[i]-y[i-1])**2+lamb)

    x_dist = 0
    for i in range(1, lenx-1):

        if x[i]>=YUE[i] and x[i-1]>=YUE[i]:
            x_dist += min(((x[i]-YUE[i])**2 + (x[i-1]-YUE[i])**2), ((x[i]-y[i-1])**2+lamb))
        if x[i]<=YLE[i] and x[i-1]<=YLE[i]:
            x_dist += min((x[i]-YLE[i])**2 + (x[i-1]-YLE[i])**2, (x[i]-x[i-1])**2+lamb)

    lb_dist = fixed_dist + max(y_dist, x_dist)

    return lb_dist