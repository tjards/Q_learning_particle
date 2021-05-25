#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:42:53 2020

@author: tjards
"""

#%% IMPORT packages
import numpy as np
import dnnModule as dnn
import pickle
import pandas as pd

#%% PREPARE the data
# ------------------------

path_train_i = 'datasets/int_inputs.pkl'
path_train_o = 'datasets/int_outputs.pkl'
path_test_i = 'datasets/int_inputs.pkl'
path_test_o = 'datasets/int_outputs.pkl'


start = 0
maxSize_train = 10000
start2 = 10001
maxSize_test = 15000
delay = 0   # to provide delayed inputs (nominally 0)


# training set
train_x = np.array(pd.read_pickle(path_train_i))[start-delay:maxSize_train-delay,0::].transpose()
train_y = np.array(pd.read_pickle(path_train_o))[start:maxSize_train,:].transpose()

# test set
test_x = np.array(pd.read_pickle(path_test_i))[start2-delay:maxSize_test-delay,0::].transpose()
test_y = np.array(pd.read_pickle(path_test_o))[start2:maxSize_test,:].transpose()


#%% normalize (on input only, fixed due to real-time)

#scale_ins_n = np.amax(abs(train_x),axis = 1)
scale_ins_n = np.array([24.7144, 309.206, 31.1158, 256.741, 27.9044, 250.66, 3137.21, 2679.5, 2522.73])
#scale_outs_n = np.amax(train_y,axis = 1)
scale_outs_n = np.array([23.3097,309.206,31.1158,256.741,25.1831,194.058])

# normalize (on entry only)
train_x = train_x/np.reshape(scale_ins_n, (-1,1))
train_y = train_y/np.reshape(scale_outs_n, (-1,1))
test_x = test_x/np.reshape(scale_ins_n, (-1,1))
test_y = test_y/np.reshape(scale_outs_n, (-1,1))



#%% SET hyperparameters
# -----------------------
n_x = train_x.shape[0]              # number of input features
n_y = train_y.shape[0]              # number of outputs
architecture = [n_x,6,n_y]           # model size [input, ..., hidden nodes, ... ,output]
learning_rate = 0.1                # learning rate (< 1.0)
num_iterations = 1500               # number of iterations
#np.random.seed(1) 
fcost = 'mse' #'x-entropy'          # x-entropy or mse

#%% Training a DNN on the training set
# ------------------------------------
parameters = dnn.train(train_x, train_y, architecture, learning_rate, num_iterations, print_cost=True,fcost=fcost)

# save the parameters to file
file_params = open("network_params.pkl","wb")
pickle.dump(parameters,file_params)
file_params.close()


#%% PREDICT on the test set
# -------------------------
print('Prediction on the training set: ')
pred_train = dnn.predict(train_x, train_y, parameters)
print('Prediction on the test set: ')
pred_test = dnn.predict(test_x, test_y, parameters)

