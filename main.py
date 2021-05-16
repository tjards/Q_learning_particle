#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# New Project: A particle that learns how to fly

## Dev notes:

- implement double integrator
- implement PD control 
- implement basic learning learning 
- build a deep Q network eventually
- see where it goes

Created on Sat May 15 19:26:29 2021

@author: tjards
"""
#%% Import stuff
# --------------

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.integrate as integrate
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' #my add - this path needs to be added
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
import qLearnLib as QL
from numpy import linalg as LA

#%% Simulation Setup
# --------------------
 
Ti = 0.0        # initial time
Tf = 300         # final time 
Ts = 0.1        # sample time
Tz = 0.005      # integration step size

state   = np.array([1.9, 0.1, 1, 0.2, 0.3, 0.4])   # format: [x, xdot, y, ydot, z, zdot]
inputs  = np.array([0.21, 0.15, 0.1])              # format: [xddot, yddot, zddot]
target  = np.array([0,0,1])                        # format: [xr, yr, zr]
outputs = np.array([state[0],state[2],state[4]])   # format: [x, y, z]
error   = outputs - target
reward = 1/LA.norm(error)
t       = Ti
i       = 1
counter = 0  
nSteps  = int(Tf/Ts+1)
        
t_all           = np.zeros(nSteps)                  # to store times
t_all[0]        = Ti                                # store initial time
states_all      = np.zeros([nSteps, len(state)])    # to store states
states_all[0,:] = state                             # store initial state
targets_all      = np.zeros([nSteps, len(target)])  # to store targets
targets_all[0,:] = target                           # store initial target
rewards_all      = np.zeros([nSteps, 1])  # to store targets
rewards_all[0,:] = reward

# initialize Q table
nParams     = 2    # number of parameters to tune
nOptions    = 10   # number of options to selection from (ranges between 0 and nOptions)
scale       = 2    # scale the value of the options (i.e. scale*[0:nOptions])
Tl          = 1    # length of trial [s]
trial_counter = 0  # initialize counter 
trial_cost    = 0  # initialze cost 
explore_rate  = 0.95  # how often to explore, 0 to 1 (start high, decrease)
Q = QL.init(nParams,nOptions)

# select initial parameters from Q table 
kp_i = QL.select(Q,0,nOptions,explore_rate)
kp = scale*kp_i
kd_i = QL.select(Q,1,nOptions,explore_rate)
kd = scale*kd_i


#%% Define the agent dynamics
# ---------------------------

def dynamics(state, t, inputs):
    
    state_dot = np.zeros(state.shape[0])
    state_dot[0] = state[1]     # xdot
    state_dot[1] = inputs[0]    # xddot
    state_dot[2] = state[3]     # ydot
    state_dot[3] = inputs[1]    # yddot
    state_dot[4] = state[5]     # zdot
    state_dot[5] = inputs[2]    # zddot
    
    return state_dot


#%% Start the Simulation
# ----------------------

while round(t,3) < Tf:

    # evolve the states through the dynamics
    state = integrate.odeint(dynamics, state, np.arange(t, t+Ts, Tz), args = (inputs,))[-1,:]

    # store results
    t_all[i]            = t
    states_all[i,:]     = state
    targets_all[i,:]    = target
    rewards_all[i,:]    = reward #1/np.maximum(trial_cost,0.00001)                           

    # increment 
    t += Ts
    i += 1

    # move target 
    target = np.array([10*np.sin(i*Ts*3),10*np.cos(0.5*i*Ts*3),1])

    # still working on this
    if round(trial_counter,3) < Tl:
    
        # accumulate the cost
        trial_cost += LA.norm(target-state[0:6:2])
    
        # increment the counter 
        trial_counter += Ts
        
    else: 
        
        trial_cost = trial_cost/Tl
        reward = np.maximum(1/trial_cost,0.00001)
        
        #update the Q table
        Q = QL.update(Q,0,kp_i,0,reward)
        Q = QL.update(Q,1,kd_i,1,reward)
        
        # select new parameter from Q table 
        kp_i = QL.select(Q,0,nOptions,explore_rate)
        kp = scale*kp_i
        kd_i = QL.select(Q,1,nOptions,explore_rate)
        kd = scale*kd_i
        #print(kp,kd,1/trial_cost)
        
        # reset
        trial_counter = 0
        
        # reduce the explore rate
        explore_rate = 0.95*explore_rate
        


    # controller (PD type)
    #kp = 2
    #kd = 1.4
    outputs = np.array([state[0],state[2], state[4]]) 
    derror = (1/Ts)*((outputs-target) - error)
    error = outputs-target
    inputs = - kp*(error) - kd*(derror)
   

# %% Animate
# ---------- 

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.grid()
axis = 10
ax.set_xlim3d([-axis, axis])
ax.set_ylim3d([-axis, axis])
ax.set_zlim3d([-axis, axis])

line, = ax.plot([], [],[], 'bo-',ms=10, lw=2)
line_target, = ax.plot([], [],[], 'ro-', ms=5, lw=2)

time_template = 'Time = %.1fs'
#reward_template = 'Last Reward = %.1f'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
time_text2 = ax.text2D(0.65, 0.95, 'Double Integrator Kinematics', transform=ax.transAxes)
time_text3 = ax.text2D(0.65, 0.90, 'Controller: PD', transform=ax.transAxes)
#text_reward = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)

def update(i):
    line.set_data(states_all[i,0],states_all[i,2])
    line.set_3d_properties(states_all[i,4])
    line_target.set_data(targets_all[i,0],targets_all[i,1])
    line_target.set_3d_properties(targets_all[i,2])
    time_text.set_text(time_template%(i*Ts))
    #text_reward.set_text(reward_template%rewards_all[i])
    return line, time_text

ani = animation.FuncAnimation(fig, update, np.arange(1, len(states_all)),
    interval=15, blit=False)

ani.save('animation.gif', writer=writer)
plt.show()

