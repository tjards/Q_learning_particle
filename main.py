#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implementation of Q-Learning to tune PD-Controller parameters

Model: double-integrator particle (3D)

Created on Sat May 15 19:26:29 2021


dev notes:
    - pass learning rate in as parameter _


@author: tjards
"""
#%% Import stuff
# --------------

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.integrate as integrate
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' #my add 
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
import qLearnLib as QL
from numpy import linalg as LA
import random
#import mpl_toolkits.axes_grid1, mpl_toolkits.axisartist
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pickle
import dnnModule as dnn

#random.seed(0)

#%% Simulation Setup
# --------------------
 
Ti          = 0.0       # initial time
Tf          = 5001      # final time 
Ts          = 0.1       # sample time
Tz          = 0.005     # integration step size
verbose     = 0         # print progress (0 = no, 1 = yes)
plotsave    = 0         # save animation (0 = no, 1 = yes), takes long time

# initialize
# ----------

state       = np.array([1.9, 0.1, 1, 0.2, 0.3, 0.4])   # format: [x, xdot, y, ydot, z, zdot]
inputs      = 0*np.array([0.21, 0.15, 0.1])              # format: [xddot, yddot, zddot]
#target     = np.array([0.0,0.0,1.0])                 # format: [xr, yr, zr]
target      = 5*np.array([random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]) 
outputs     = np.array([state[0],state[2],state[4]])   # format: [x, y, z]
error       = outputs - target
trial_cost  = LA.norm(error)
reward      = 1/LA.norm(error)
t           = Ti
i           = 1
nSteps      = int(Tf/Ts+1)

# constaints
# ----------

xmax = 10       # these would be like, walls/floors (safety)
vmax = 5        # max velocity
umax = vmax/Ts  # max acceleration 


   
# storage
# -------
     
t_all               = np.zeros(nSteps)                 # to store times
t_all[0]            = Ti                                
states_all          = np.zeros([nSteps, len(state)])   # to store states
states_all[0,:]     = state                             
inputs_all          = np.zeros([nSteps, len(inputs)])   # to store states
inputs_all[0,:]     = inputs                            
targets_all         = np.zeros([nSteps, len(target)])  # to store targets
targets_all[0,:]    = target                          
rewards_all         = np.zeros([nSteps, 1])            # to store rewards
rewards_all[0,:]    = reward
costs_all           = np.zeros([nSteps, 1])            # to store costs
costs_all[0,:]      = trial_cost
explore_rates_all   = np.zeros([nSteps, 1])            # to store explore rates                            

# Q learning stuff
# ----------------

nParams         = 2    # number of parameters to tune
nOptions        = 15   # number of options to selection from (ranges between 0 and nOptions)
scale           = 1    # scale the value of the options (i.e. scale*[0:nOptions])
Tl              = 5    # length of trial [s]
trial_counter   = Ts   # initialize counter (in-trial)
trial_cost      = 0    # initialze cost 
trial_counts    = 0    # total number of trials
explore_rate    = 1    # how often to explore, 0 to 1 (start high, decrease)
epsilon         = 0.997                     # explore rate of change (->0 faster)
Q               = QL.init(nParams,nOptions) # this is the Q-table 

# randomly select first parameters
kp_i = QL.select(Q,0,nOptions,explore_rate) # index for first parameter
kp = scale*kp_i                             # actual value used
kd_i = QL.select(Q,1,nOptions,explore_rate)
kd = scale*kd_i
explore_rates_all[0,:]  = explore_rate
target_rand0 = 2*random.uniform(-1, 1)      # used for pseudo-random path gen
target_rand1 = 2*random.uniform(-1, 1)
target_rand2 = 2*random.uniform(-1, 1)

# Deep Neural Network stuff
# -------------------------

mini_batch_size     = int(1000/Ts)       # divide by Ts to define in seconds
mini_batch_counts   = -1 #int(-1000/Ts)
DNN_run_count       = 0
DNN_mode            = 'state'            # model based on state (good) or dstate (not good, for broad state spaces)


# import the initial parameters for the DNN
file_initial_parameters = open("initial_params.pkl","rb")
initial_parameters = pickle.load(file_initial_parameters)
file_initial_parameters.close()
#DNN_parameters = initial_parameters.copy()
DNN_parameters = 'random'

# initialize the DNN data
#DNN_outs = states_all[1::,:]                                    # these are the "next states" (i.e. x_k+1)
#DNN_ins = np.hstack((states_all[0:-1,:],inputs_all[0:-1,:]))    # these are the "current states" and "current inputs"


# scaling - manual
#scale_outs_n = np.array([23.3097,309.206,31.1158,256.741,25.1831,194.058])
#scale_ins_n = np.array([24.7144, 309.206, 31.1158, 256.741, 27.9044, 250.66, 3137.21, 2679.5, 2522.73])

# scaling - none
#scale_outs_n = np.array([1,1,1,1,1,1])
#scale_ins_n = np.array([1,1,1,1,1,1,1,1,1])

# scaling - auto
#scale_ins_n = np.amax(abs(train_x),axis = 1)
#scale_outs_n = np.amax(abs(train_y),axis = 1)

#train_x = np.hstack((states_all[0:-1,:],inputs_all[0:-1,:])).transpose()/np.reshape(scale_ins_n, (-1,1))
#train_y = states_all[1::,:].transpose()/np.reshape(scale_outs_n, (-1,1))

#train_x = np.hstack((states_all[0:-1,:],inputs_all[0:-1,:])).transpose()
#train_y = states_all[1::,:].transpose()

# store DNN predicted states (ghosts)
ghosts_all          = np.zeros([nSteps, len(state)])   # to store ghosts
ghosts_all[0,:]     = state 


#%% Define the agent dynamics
# ---------------------------

def dynamics(state, t, inputs):
    
    # double-integrator    
    state_dot = np.zeros(state.shape[0])
    state_dot[0] = state[1]     # xdot
    state_dot[1] = inputs[0]    # xddot
    state_dot[2] = state[3]     # ydot
    state_dot[3] = inputs[1]    # yddot (acceleration)
    state_dot[4] = state[5]     # zdot
    state_dot[5] = inputs[2]    # zddot (acceleration)
    
    #apply constraints (velo/acc)
    #state_dot[0]=np.maximum(np.minimum(state_dot[0],vmax),-vmax)
    #state_dot[2]=np.maximum(np.minimum(state_dot[2],vmax),-vmax)
    #state_dot[4]=np.maximum(np.minimum(state_dot[4],vmax),-vmax)
    #state_dot[1]=np.maximum(np.minimum(state_dot[1],umax),-umax)
    #state_dot[3]=np.maximum(np.minimum(state_dot[3],umax),-umax)
    #state_dot[5]=np.maximum(np.minimum(state_dot[5],umax),-umax)
    
    return state_dot


#%% Start the Simulation
# ----------------------

while round(t,3) < Tf:

      
    # store results
    # -------------
    t_all[i]                = t
    states_all[i,:]         = state
    inputs_all[i,:]         = inputs 
    targets_all[i,:]        = target
    rewards_all[i,:]        = reward       
    costs_all[i,:]          = trial_cost 
    explore_rates_all[i,:]  = explore_rate 


    # model the system with a DNN (by minibatches)
    # ---------------------------
    
    if mini_batch_counts == mini_batch_size:
        
        print('DNN w/new minibatch at i= ',i-mini_batch_size-1,' of size ',mini_batch_size)
        
        # ranges
        batch_start  =   i-mini_batch_size-1
        batch_end    =   i-1
                
        # build the training set
        #train_x = np.hstack((states_all[batch_start:batch_end,:],inputs_all[batch_start:batch_end,:])).transpose()/np.reshape(scale_ins_n, (-1,1))
        #train_y = states_all[batch_start+1:batch_end+1,:].transpose()/np.reshape(scale_outs_n, (-1,1))
        
        # depending on mode
        if DNN_mode == 'state':
            # model by state
            train_x = np.hstack((states_all[batch_start:batch_end,:],inputs_all[batch_start:batch_end,:])).transpose()
            train_y = states_all[batch_start+1:batch_end+1,:].transpose()
        if DNN_mode == 'dstate':
            train_x = np.hstack((states_all[batch_start:batch_end,:]-states_all[batch_start-1:batch_end-1,:],inputs_all[batch_start:batch_end,:])).transpose()
            train_y = states_all[batch_start+1:batch_end+1,:].transpose() - states_all[batch_start:batch_end,:].transpose()
        
        # if this is the first run
        if DNN_run_count == 0:
            #define the scaling (wag, based on max/min of values)
            scale_ins_n = np.amax(abs(train_x),axis = 1)
            #scale_outs_n = np.amax(abs(train_y),axis = 1)
            scale_outs_n=np.array([xmax,vmax,xmax,vmax,xmax,vmax])
            
            # double-check the constraints make sense
            if scale_ins_n[0:train_y.shape[0]].any() != scale_outs_n.any():
                print('Warning: Possible error with scaling')
                print('... or states/inputs modes not fully excited')
            else:
                print('Note: Scaling done, all modes sufficiently excited in this batch')
            #scale_outs_n = np.array([1,1,1,1,1,1])
            #scale_ins_n = np.array([1,1,1,1,1,1,1,1,1])
            ghosts_all[0,:] = states_all[0,:]/np.reshape(scale_outs_n, (-1,1)).transpose()
            
        DNN_run_count += 1
        
        # now scale them
        train_x = train_x/np.reshape(scale_ins_n, (-1,1))
        train_y = train_y/np.reshape(scale_outs_n, (-1,1))
        
        # scale all ghosts
        #ghosts_all[:,:] = ghosts_all[:,:]/np.reshape(scale_outs_n, (-1,1)).transpose()
            
        # define hyper-parameters
        n_x = train_x.shape[0]              # number of input features
        n_y = train_y.shape[0]              # number of outputs
        architecture = [n_x,6,n_y]           # model size [input, ..., hidden nodes, ... ,output]
        architecture = [n_x,6,n_y]
        learning_rate = 0.1                # learning rate (< 1.0)
        num_iterations = 2000               # number of iterations
        #np.random.seed(1) 
        fcost = 'mse' #'x-entropy'          # x-entropy or mse
        
        # train
        DNN_parameters = dnn.train(train_x, train_y, architecture, learning_rate, num_iterations, print_cost=True, fcost=fcost, initialization = DNN_parameters)
        
        # Run a mini simulation using these parameters
        # --------------------------------------------
        
        ### ~~~~~~~~~~~~~~~~~~ counterfactiual dream land ~~~~~~~~~~~~~~~~ ###
        
        # initialize test-set with actual states (will be replaced with predictions)
        test_x = train_x # already normalized
        test_y = train_y # already normalized
              
        # load the initial condition for the ghosts (actual state)
        ghosts_all[batch_start:batch_start+1,:] = states_all[batch_start:batch_start+1,:]/np.reshape(scale_outs_n, (-1,1)).transpose()
        
        # start a simulated trial counter
        sim_trial_counter = Ts   # initialize counter (in-trial)

        # for each sample in the batch     
        #for k in range(batch_start,batch_end):
        for k in range(0,mini_batch_size-1):
            
            # if a trial resets
            if round(sim_trial_counter,5) > Tl:
                # feed it a new position estimate
                ghosts_all[batch_start+k,:] = states_all[batch_start+k,:]/np.reshape(scale_outs_n, (-1,1)).transpose()
                # reset the counter
                sim_trial_counter = 0
                
            
            # - ignore
            #test_x_k = np.hstack((ghosts_all[k:k+1,:],inputs_all[k:k+1,:])).transpose()/np.reshape(scale_ins_n, (-1,1))
            
            # replace the sample in the test set with a ghost
            #est_x[0:n_y,k] = ghosts_all[k,:].transpose()
            
            if DNN_mode == 'state':
                test_x[0:n_y,k] = ghosts_all[batch_start+k,:].transpose() 
                # - ignore
                #test_y_k = states_all[k+1:k+2,:].transpose()/np.reshape(scale_outs_n, (-1,1))
                #ghosts_all[k+1:k+2,:] = dnn.predict(test_x_k, test_y_k, DNN_parameters).transpose()
                # make a prediction and update next ghost
                #ghosts_all[k+1:k+2,:] = dnn.predict(test_x[:,k], test_y[:,k], DNN_parameters).transpose()
                #ghosts_all[k+1:k+2,:] = dnn.predict(np.reshape(test_x[:,k],(-1,1)), np.reshape(test_y[:,k],(-1,1)), DNN_parameters).transpose()
                ghosts_all[batch_start+k+1:batch_start+k+2,:] = dnn.predict(np.reshape(test_x[:,k],(-1,1)), np.reshape(test_y[:,k],(-1,1)), DNN_parameters).transpose()

            if DNN_mode == 'dstate':
                test_x[0:n_y,k] = ghosts_all[batch_start+k,:].transpose() - ghosts_all[batch_start+k-1,:].transpose() 
                # - ignore
                #test_y_k = states_all[k+1:k+2,:].transpose()/np.reshape(scale_outs_n, (-1,1))
                #ghosts_all[k+1:k+2,:] = dnn.predict(test_x_k, test_y_k, DNN_parameters).transpose()
                # make a prediction and update next ghost
                #ghosts_all[k+1:k+2,:] = dnn.predict(test_x[:,k], test_y[:,k], DNN_parameters).transpose()
                #ghosts_all[k+1:k+2,:] = dnn.predict(np.reshape(test_x[:,k],(-1,1)), np.reshape(test_y[:,k],(-1,1)), DNN_parameters).transpose()
                change = dnn.predict(np.reshape(test_x[:,k],(-1,1)), np.reshape(test_y[:,k],(-1,1)), DNN_parameters).transpose()
                ghosts_all[batch_start+k+1:batch_start+k+2,:] = ghosts_all[batch_start+k:batch_start+k+1,:] + change

            #move the sim counter forward
            sim_trial_counter += Ts

            
        # now unscale
        #ghosts_all[batch_start:batch_end+1,:] = ghosts_all[batch_start:batch_end+1,:]*np.reshape(scale_outs_n, (-1,1)).transpose()
        #ghosts_all[:,:] = ghosts_all[:,:]*np.reshape(scale_outs_n, (-1,1)).transpose()
        # do outside now
        
        # reset batch count
        mini_batch_counts = 0
        
        ### ~~~~~~~~~~~~~~~~~~ counterfactiual dream land ~~~~~~~~~~~~~~~~ ###
        
    mini_batch_counts += 1


    # evolve the states through the dynamics
    state = integrate.odeint(dynamics, state, np.arange(t, t+Ts, Tz), args = (inputs,))[-1,:]  


    #apply constraints (pos)
    state[0]=np.maximum(np.minimum(state[0],xmax),-xmax)
    state[2]=np.maximum(np.minimum(state[2],xmax),-xmax)
    state[4]=np.maximum(np.minimum(state[4],xmax),-xmax)
    state[1]=np.maximum(np.minimum(state[1],vmax),-vmax)
    state[3]=np.maximum(np.minimum(state[3],vmax),-vmax)
    state[5]=np.maximum(np.minimum(state[5],vmax),-vmax)
    
    #state[0]=np.maximum(np.minimum(state[0],xmax),-xmax)
    #state[2]=np.maximum(np.minimum(state[2],xmax),-xmax)
    #state[4]=np.maximum(np.minimum(state[4],xmax),-xmax)

    
                    

    # increment 
    t += Ts
    i += 1

    # accumulate cost over thie trial
    # -------------------------------
    
    if round(trial_counter,5) < Tl:
    
        # accumulate the cost (position + velocity component)
        trial_cost += LA.norm(target-state[0:6:2])/Tl - ((LA.norm(target-state[0:6:2])-LA.norm(error)))/Tl

        
        # increment the counter 
        trial_counter += Ts
        
        #reward = 0
    
    # if trial is over, compute rewards
    # ---------------------------------
    
    else: 
              
        #compute the reward (cost normalized over Tl)
        reward = 1/np.maximum(trial_cost,0.00001) # protect div by zero
        
        #update the Q table
        Q = QL.update(Q,0,kp_i,1,reward)
        Q = QL.update(Q,1,kd_i,0,reward)
        
        # select new parameter from Q table 
        kp_i = QL.select(Q,0,nOptions,explore_rate)
        kp = scale*kp_i
        kd_i = QL.select(Q,1,nOptions,explore_rate)
        kd = scale*kd_i
        
        # reset the counters
        trial_counter = Ts      # in-trial counter
        trial_cost = 0
        trial_counts += 1       # total number of trials
        if verbose == 1:
            print('trial ', trial_counts, 'done @:', round(t,2), 's' )
        
        # select new target (randomly)
        target = 10*np.array([random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]) 
        target_rand0 = 5*random.uniform(-1, 1)
        target_rand1 = 5*random.uniform(-1, 1)
        target_rand2 = 5*random.uniform(-1, 1)
        
        # reduce the explore rate (floors at 0.001)
        explore_rate = np.maximum(epsilon**t,0.001)-0.001
        
    # wander the target (according this this trial's random parameters)
    target += 0.5*np.array([1*np.sin(i*Ts*target_rand0),1*np.cos(0.5*i*Ts*target_rand1),1*np.sin(i*Ts*target_rand2)])

    # controller (PD type)
    outputs = np.array([state[0],state[2], state[4]]) 
    derror = (1/Ts)*(error - (outputs-target))
    error = outputs-target
    inputs = - kp*(error) + kd*(derror)
    
    #apply constraints
    inputs[0]=np.maximum(np.minimum(inputs[0],umax),-umax)
    inputs[1]=np.maximum(np.minimum(inputs[1],umax),-umax)
    inputs[2]=np.maximum(np.minimum(inputs[2],umax),-umax)
    
   
    
# %% Results
# -----------

# rescale the ghosts
ghosts_all[:,:] = ghosts_all[:,:]*np.reshape(scale_outs_n, (-1,1)).transpose()


for k in range(0,nParams):
    print('Best parameter ', k, ' is: ', np.argmax(Q[k,:]),' with ', np.max(Q[k,:]))

print('Final Explore Rate: ', explore_rate)


# %% Save data for DNN

# output in discrete model form for DNN
DNN_outs = states_all[1::,:]                                    # these are the "next states" (i.e. x_k+1)
DNN_ins = np.hstack((states_all[0:-1,:],inputs_all[0:-1,:]))    # these are the "current states" and "current inputs"

with open('Data/states.pkl','wb') as outs:
    pickle.dump(DNN_outs,outs, pickle.HIGHEST_PROTOCOL)
with open('Data/inputs.pkl','wb') as ins:
    pickle.dump(DNN_ins,ins, pickle.HIGHEST_PROTOCOL)


# Plots 
# -----

#%% Trajectory animation

inset = 1

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.grid()
axis = 10
ax.set_xlim3d([-axis, axis])
ax.set_ylim3d([-axis, axis])
ax.set_zlim3d([-axis, axis])

line, = ax.plot([], [],[], 'bo-',ms=10, lw=2)
line_target, = ax.plot([], [],[], 'ro-', ms=5, lw=2)
line_ghost, = ax.plot([], [],[], 'mo-',ms=10, lw=2)

time_template = 'Time = %.1f/%.0fs'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
time_text2 = ax.text2D(0.65, 0.95, 'Q-Learning Control', transform=ax.transAxes)
time_text3 = ax.text2D(0.65, 0.90, 'Controller: PD', transform=ax.transAxes)


if inset == 1:

    ax4 = plt.axes([0,0,1,1])
    # set position manually (x pos, y pos, x len, y len)
    ip = InsetPosition(ax, [0,0.7,0.2,0.2])
    ax4.set_axes_locator(ip)
    # cool, make lines to point (save this for later)
    #mark_inset(ax2, ax3, loc1=2, loc2=4, fc="none", ec='0.5')
    # data
    line2, = ax4.plot([], [], '--', c='b', mew=2, alpha=0.8,label='Cost')
    
    ax5 = ax4.twinx()
    ax5.set_axes_locator(ip)
    # data
    line3, = ax4.plot([], [], '-', c='g', mew=2, alpha=0.8,label='Explore')
    
    ax4.set_ylabel('Cost')
    ax4.set_xlim(0,max(t_all))
    ax4.set_ylim(0,1)
    ax4.tick_params(axis='y', labelcolor='b')
    
    ax5.set_ylabel('Explore Rate')
    ax5.set_xlim(0,max(t_all))
    ax5.set_ylim(0,1)
    ax5.tick_params(axis='y', labelcolor='g')

def update(i):
    line.set_data(states_all[i,0],states_all[i,2])
    line.set_3d_properties(states_all[i,4])
    line_target.set_data(targets_all[i,0],targets_all[i,1])
    line_target.set_3d_properties(targets_all[i,2])
    line_ghost.set_data(ghosts_all[i,0],ghosts_all[i,1])
    line_ghost.set_3d_properties(ghosts_all[i,2])
    time_text.set_text(time_template%(i*Ts,Tf))
    #text_reward.set_text(reward_template%rewards_all[i])
    if inset == 1:
        line2.set_data(t_all[0:i],(1/float(max(costs_all[0:i])))*costs_all[0:i,0])
        ax4.set_xlim(0,t_all[i]+1)
        line3.set_data(t_all[0:i],(1/float(max(explore_rates_all[0:i])))*explore_rates_all[0:i,0])
        ax5.set_xlim(0,t_all[i]+1)
    return line, time_text

#fast
ani = animation.FuncAnimation(fig, update, np.arange(1, len(states_all)),interval=15, blit=False)

#slow (for saving)
if plotsave == 1:
    ani2 = animation.FuncAnimation(fig, update, np.linspace(1, len(states_all)-1, num=500, dtype=int),interval=15, blit=False)
    if verbose == 1:
        ani2.save('animation.gif', writer=writer, progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))
    else:
        ani2.save('animation.gif', writer=writer)

#plt.show()

#%% Costs

starts = 2*int(Tl/Ts)-1
segments = int(Tl/Ts)
polyfit_n = 1

# plot costs
fig2, ax2 = plt.subplots()
plt.title('Q-Learning Control Parameters')
ax2.plot(t_all[starts::segments],costs_all[starts::segments,0],'-', c='b', mew=2, alpha=0.8,label='Cost')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Cost [m]')
ax2.set_xlim(0,max(t_all))
ax2.set_ylim(0,max(costs_all))

# create inset axis
ax3 = plt.axes([0,0,1,1])
# set position manually (x pos, y pos, x len, y len)
ip = InsetPosition(ax2, [0.55,0.55,0.4,0.4])
ax3.set_axes_locator(ip)
# cool, make lines to point (save this for later)
#mark_inset(ax2, ax3, loc1=2, loc2=4, fc="none", ec='0.5')
# data
ax3.plot(t_all[starts::segments],explore_rates_all[starts::segments,0],'--', c='g', mew=2, alpha=0.8,label='Explore Rate')
#ax3.plot(t_all[0::int(Tl/Ts)],rewards_all[0::int(Tl/Ts),0],'--', c='g', mew=2, alpha=0.8,label='Rewards')

ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Explore Rate')
ax3.set_xlim(0,max(t_all))
ax3.set_ylim(0,1)

plt.savefig('cost.png')


#%% Rewards

# plot costs
fig2, ax2 = plt.subplots()
plt.title('Q-Learning Control Parameters')
ax2.plot(t_all[starts::segments],costs_all[starts::segments,0],'-', c='b', mew=2, alpha=0.8,label='Cost')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Cost [m]')
#ax2.legend(loc=0)
ax2.set_xlim(0,max(t_all))
ax2.set_ylim(0,max(costs_all))

#best fit line
cost_fit = np.polyfit(t_all[starts::segments], costs_all[starts::segments,0], polyfit_n)
p_cost_fit = np.poly1d(cost_fit)
ax2.plot(t_all[starts::segments], p_cost_fit(t_all[starts::segments]), 'k-')


# create inset axis
ax3 = plt.axes([0,0,1,1])
# set position manually (x pos, y pos, x len, y len)
ip = InsetPosition(ax2, [0.55,0.55,0.4,0.4])
ax3.set_axes_locator(ip)
# cool, make lines to point (save this for later)
#mark_inset(ax2, ax3, loc1=2, loc2=4, fc="none", ec='0.5')
# data
ax3.plot(t_all[starts::segments],rewards_all[starts::segments,0],'--', c='g', mew=2, alpha=0.8,label='Rewards')
#ax3.plot(t_all[0::int(Tl/Ts)],test,'--', c='k', mew=2, alpha=0.8,label='Rewards')

#best fit line
reward_fit = np.polyfit(t_all[starts::segments], rewards_all[starts::segments,0], polyfit_n)
p_reward_fit = np.poly1d(reward_fit)
ax3.plot(t_all[starts::segments], p_reward_fit(t_all[starts::segments]), 'k-')
#plt.show()

ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Rewards')
ax3.set_xlim(0,max(t_all))
ax3.set_ylim(0,max(rewards_all))

plt.savefig('rewards.png')


# plot costs
fig2, ax2 = plt.subplots()
plt.title('Q-Learning Control Parameters')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Cost [m]')
# data
ax2.plot(t_all[starts::segments],rewards_all[starts::segments,0],'--', c='g', mew=2, alpha=0.8,label='Rewards')
#ax3.plot(t_all[0::int(Tl/Ts)],test,'--', c='k', mew=2, alpha=0.8,label='Rewards')

#best fit line
reward_fit = np.polyfit(t_all[starts::segments], rewards_all[starts::segments,0], polyfit_n)
p_reward_fit = np.poly1d(reward_fit)
ax2.plot(t_all[starts::segments], p_reward_fit(t_all[starts::segments]), 'k-')
#plt.show()

ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Rewards')
ax2.set_xlim(0,max(t_all))
ax2.set_ylim(0,max(rewards_all))

plt.savefig('rewards2.png')

#%%

fig2, ax2 = plt.subplots()
#ax2.set_xlabel('Time [s]')
#ax2.set_ylabel('Pos [m]')
# data
#ax2.plot(t_all[1001:9000],states_all[1001:9000,0],'--', c='k', mew=2, alpha=0.8)
#ax2.plot(t_all[1001:9000],ghosts_all[1001:9000,0],'--', c='r', mew=2, alpha=0.8)

#ax2.plot(states_all[8501:9000,0],states_all[8501:9000,2],'--k',ghosts_all[8501:9000,0],ghosts_all[8501:9000,2],'--r')
#ax2.plot(states_all[:,0],states_all[:,2],'--k',ghosts_all[:,0],ghosts_all[:,2],'--r')
#ax2.plot(t_all[1001:9000],ghosts_all[1001:9000,0],'--', c='r', mew=2, alpha=0.8)
begin = 40001
ending = begin+500
var = 1


#ax2.plot(t_all[begin:ending],states_all[begin:ending,var],'--k',t_all[begin:ending],ghosts_all[begin:ending,var],'--r')
ax2.plot(t_all[begin:ending],states_all[begin:ending,var],'--k',t_all[begin:ending],ghosts_all[begin:ending,var],'--r',t_all[begin:ending],inputs_all[begin:ending,0], '--m')
fig2.legend(['states', 'ghosts', 'inputs'])
#ax2.plot(states_all[10000:2*10000,0],states_all[10000:2*10000,2],'--k',ghosts_all[10000:2*10000,0],ghosts_all[10000:2*10000,2],'--r')

