#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:06:18 2020

@author: tjards

This file implements a deep neural network (for classification or regression)

References:
    
    Andrew Ng et al., "Neural Networks and Deep Learning", course through deeplearning.ai:
        https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning

Key notation:
    
    X:              data
    Y:              labels (1/0 for classification or outputs for regression ) 
    architecture:   node count for input, hidden (in order), output layers
    learning_rate:  learning rate
    num_iterations: number of iterations of the optimization loop
    parameters:     these are the learned parameters 
    Wl:             weight matrix of shape (architecture[l], architecture[l-1])
    bl:             bias vector of shape (architecture[l], 1)

"""

#%% Import stuff
# --------------
import numpy as np
import matplotlib.pyplot as plt
#import h5py

#%% Settings
# ----------

nonlin          = "tanh" # which nonlinear activation function to use (sigmoid, relu, or tanh)
print_progress  = 2      # 1 = yes, 0 = no, 2 = yes but no plots
print_rate      = 100    # rate at which to print results (default 100)
output_act      = 'tanh'


#%% Main training function 
# ------------------------
 
def train(X, Y, architecture, learning_rate, num_iterations, print_cost=True, fcost='x-entropy', initialization = 'random'):

    # initialize
    costs = []
               
    if initialization == 'random':              
        parameters = init_params(architecture)
    else:
        parameters = initialization
    
    # Run gradient descent 
    for i in range(0, num_iterations):

        # Forward propagation
        A, caches = forward_prop(X, parameters)
        
        # Compute cost
        if fcost == 'x-entropy':
            cost = compute_cost_ENT(A, Y)
        elif fcost == 'mse':
            cost = compute_cost_MSE(A, Y)
        
        # Backward propagation
        grads = backward_prop(A, Y, caches, fcost)
        
        # Update 
        parameters = update(parameters, grads, learning_rate)
                
        # Print the cost every "print_rate" training example
        if print_progress == 1 or print_progress == 2 :
            if print_cost and i % print_rate == 0:
                print ("DNN cost after iteration %i: %f" %(i, cost))
            if print_cost and i % print_rate == 0:
                costs.append(cost)
                
    # plot the cost
    if print_progress == 1:
        plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
        plt.figure()
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('Iterations/100')
        plt.title("Learning rate =" + str(learning_rate))
     
    return parameters

#%% Activation Functions (forward)
# --------------------------------

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    
    return A, cache

def activation_tanh(Z):

    A = 2/(1+np.exp(-2*Z))-1
    cache = Z
    
    return A, cache

def activation_lin(Z):
    
    A = Z
    cache = Z
    
    return A, cache

#%% Activation Functions (backward)
# --------------------------------

def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0 
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):

    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ
  
def activation_lin_backward(dA, cache):
    
    #Z = cache
    dZ = 1* dA 
    
    return dZ 

def activation_tanh_backward(dA, cache):
    
    Z = cache
    
    s = 2/(1+np.exp(-2*Z))-1
    dZ = dA * (1-s*s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

#%% Initialization 

def init_params(architecture):

    #np.random.seed(1)
    parameters = {}
    L = len(architecture) # Total Layers (iterate as l)          

    # for each layer
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(architecture[l], architecture[l-1]) / np.sqrt(architecture[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((architecture[l], 1))
        
        # just maker sure the sizes are right 
        assert(parameters['W' + str(l)].shape == (architecture[l], architecture[l-1]))
        assert(parameters['b' + str(l)].shape == (architecture[l], 1))
 
    return parameters


#%% Forward propagation tools
# ---------------------------

def linear_forward(A, W, b):

    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation == "lin":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activation_lin(Z)
        
    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activation_tanh(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def forward_prop(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  
    
    # for each layer 
    for l in range(1, L):
        A_prev = A 
        
        # run through the weights/bias and the activation function 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = nonlin)
        
        # save for later 
        caches.append(cache)
    
    # output layer is linear 
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = output_act)
    
    # save outlayer, too
    caches.append(cache)
    
    # assert the state space size here, just to be sure
    assert(AL.shape == (6,X.shape[1]))   
            
    return AL, caches

#%% Compute costs
# ---------------

# for classification 
def compute_cost_ENT(AL, Y):

    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# for regression 
def compute_cost_MSE(AL, Y):
    
    m = Y.shape[1]

    #cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = (1./m) * np.sum(np.power(np.subtract(AL,Y),2))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


#%% Back propagation tools
# ------------------------

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "lin":
        dZ = activation_lin_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "tanh":
        dZ = activation_tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def backward_prop(AL, Y, caches, fcost):

    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    if fcost == 'x-entropy':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif fcost == 'mse':
        #dAL = (1./m) * np.sum(np.subtract(AL,Y))
        dAL = 2*np.subtract(AL,Y)

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = output_act)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = nonlin)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#%% Updates
# ---------

def update(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

#%% Predictions
# -------------

def predict(X, y, parameters):

    m = X.shape[1]
    #n = len(parameters) // 2 # number of layers in the neural network
    #p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = forward_prop(X, parameters)

    
    # # convert probas to 0/1 predictions
    # for i in range(0, probas.shape[1]):
    #     if probas[0,i] > 0.5:
    #         p[0,i] = 1
    #     else:
    #         p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    #print("Accuracy: "  + str(np.sum((p == y)/m)))
    #print("Avg Error ", (1/m)*np.sum(np.power(probas.flatten()-y.flatten(),2)))
        
    return probas

