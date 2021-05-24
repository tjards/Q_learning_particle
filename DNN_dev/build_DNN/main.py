#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:42:53 2020

@author: tjards
"""

#%% IMPORT packages
#import time
import numpy as np
#import h5py
#import matplotlib.pyplot as plt
#import scipy
#import imageio
#from PIL import Image
#from scipy import ndimage
import dnnModule as dnn
import pickle
import pandas as pd

#%% PREPARE the data
# ------------------------

#scale_ins = 1   #1/4
#scale_outs = 1  #1000

# define paths
# path_train_i = 'datasets/inputs_train.pkl'
# path_train_o = 'datasets/outputs_train.pkl'
# path_test_i = 'datasets/inputs_test.pkl'
# path_test_o = 'datasets/outputs_test.pkl'


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
##train_x = np.vstack((train_x,np.array(pd.read_pickle(path_train_i))[start-1:maxSize_train-1,:].transpose())) # delays
train_y = np.array(pd.read_pickle(path_train_o))[start:maxSize_train,:]
#train_y = train_y.reshape((6, train_y.shape[0]))
train_y = train_y.transpose()

# test set
test_x = np.array(pd.read_pickle(path_test_i))[start2-delay:maxSize_test-delay,0::].transpose()
##test_x = np.vstack((test_x, np.array(pd.read_pickle(path_test_i))[start-1:maxSize_test-1,:].transpose())) # delays
test_y = np.array(pd.read_pickle(path_test_o))[start2:maxSize_test,:]
#test_y = test_y.reshape((6, test_y.shape[0]))
test_y = test_y.transpose()


#%% normalize
#get scales
scale_ins_n = np.amax(abs(train_x),axis = 1)
scale_outs_n = np.amax(train_y,axis = 1)
# normalize
train_x = train_x/np.reshape(scale_ins_n, (-1,1))
train_y = train_y/np.reshape(scale_outs_n, (-1,1))
test_x = test_x/np.reshape(scale_ins_n, (-1,1))
test_y = test_y/np.reshape(scale_outs_n, (-1,1))


# #%% Feature engineering (discontinued)
# data_in_labels = ['time','ref_x','ref_y','ref_z','ref_psi','x','y','z','dx','dy','dz','phi','theta','psi','p','q','r']
# data_out_labels =['w1','w2','w3','w4']

# # train
# errors_pos = train_x[1:4,:]-train_x[5:8,:]
# error_psi = np.reshape(train_x[4,:]-train_x[13,:],(1,-1))
# train_x_new = np.vstack((errors_pos,error_psi,train_x[8:17,:]))
# train_x = train_x_new

# # test
# errors_pos = test_x[1:4,:]-test_x[5:8,:]
# error_psi = np.reshape(test_x[4,:]-test_x[13,:],(1,-1))
# test_x_new = np.vstack((errors_pos,error_psi,test_x[8:17,:]))
# test_x = test_x_new



# pull out key parameters
#classes = np.array(test_dataset["list_classes"][:])     # list of classes
#num_px = train_x_orig.shape[1]                          # image size
#m_train = train_x_orig.shape[0]                         # number of training samples
#m_test = test_x_orig.shape[0]                           # number of test samples
#my_label_y = [1]                                        # label of my image (1, 0)


#%% SET hyperparameters
# -----------------------
n_x = train_x.shape[0]              # number of input features
n_y = train_y.shape[0]              # number of outputs
layers_dims = [n_x,6,n_y]  # model size [input, ..., hidden nodes, ... ,output]
#layers_dims = [n_x,16,8,16,4,n_y]
learning_rate=0.3                # learning rate (< 1.0)
num_iterations = 2000             # number of iterations
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1) 
fcost = 'mse' #'x-entropy'                 # x-entropy or mse

#%% Training a DNN on the training set
# ------------------------------------
parameters = dnn.train(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost=True,fcost=fcost)

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
#dnn.print_mislabeled_images(classes, test_x, test_y, pred_test) # show me the mis-predicted stuff


#%% identify a single image
# ------------------------

# # prepare the image
# image_arr = np.array(imageio.imread(path_my_image))
# image = Image.fromarray(image_arr)
# image = image.resize(size=(num_px,num_px))
# image = np.array(image) 
# my_image=image.reshape((num_px*num_px*3,1))
# my_image = my_image/255.

# # predict
# print('Prediction on manual input: ')
# my_predicted_image = dnn.predict(my_image, my_label_y, parameters)

# # plot
# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.figure()
# plt.imshow(image_arr)
# print ("Model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")







