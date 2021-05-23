#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Q-Learning module

Created on Sat May 15 20:40:36 2021

@author: tjards
"""


import numpy as np
import random

# Setup
# ------
#nState = 2          # number of states
#nAction = 2         # number of actions
#explore_rate = 0.8  # how often to explore, 0 to 1
learn_rate = 0.5    # how much to accept new observations, alpha/lambda, 0.9
discount = 0.9      # balance immediate/future rewards, (gamma): 0.8 to 0.99


# Initialize Q table
# ------------------
def init(nState,nAction):
    Q = np.zeros((nState, nAction))
    return Q

# Select an Action
# ----------------
def select(Q,state,nAction, explore_rate):
    if random.uniform(0, 1) < explore_rate:
        # Explore (select randomly)        
        action = random.randint(0,nAction-1)  
    else:
        # Exploit (select best)
        action = np.argmax(Q[state,:]) # not complete
    return action

# Get new state, reward
# ---------------------
#next_state = 0 # new_state, reward = environment(action) 
#reward = 0
            
# Update Q table
# ---------------
def update(Q,state,action,next_state,reward):
    next_action = np.argmax(Q[next_state,:])
    Q[state, action] += np.multiply(learn_rate, reward + discount*Q[next_state, next_action] - Q[state, action])
    return Q
