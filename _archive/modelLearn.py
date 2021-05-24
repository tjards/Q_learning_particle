#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:51:07 2021

@author: tjards


form:
    
    state, x_k = [x 
                  xdot]
    
    dynamics, x_k = F*x_{k-1} + G*a_k 
        where a_k is acceleration
    
    where, ideally:
        
        F = [1  dt
             0  1]
        G = [1/2dt^2
             dt]
        
    subject to uncertainty:
    
        x_k = F*x_{k-1} + w_k (modelling error)
        
        measurement, z_k = H*x_k + v_k (measurement noise)
            where H = [1 0]


"""

#%% Import stuff

import numpy as np
import matplotlib.pyplot as plt

#%% Data 


x = np.array([[0, 1, 2, 3],
             [0.1, 1.1, 2.3, 2.9],
             [-0.2, 0.8, 2.2, 3.3],
             [0.3, 1.2, 2.2, 3.5]])
y = np.array([-1, 
              0.2, 
              0.9, 
              0.4]).T


#%% Reshape for solver
# to y = Ap

A = x

p = np.linalg.lstsq(A, y, rcond=None)[0]


#%% Plot


# _ = plt.plot(x, y, 'o', label='Raw Data', markersize=10)
# _ = plt.plot(x, x*np.array([a,b,c,d]) , 'r', label='Best Fit')
# _ = plt.legend()
# plt.show()