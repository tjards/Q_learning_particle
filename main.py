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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pickle

#random.seed(0)

#%% Simulation Setup
# --------------------
 
Ti          = 0.0       # initial time
Tf          = 3000      # final time 
Ts          = 0.1       # sample time
Tz          = 0.005     # integration step size
verbose     = 0         # print progress (0 = no, 1 = yes)
plotsave    = 0         # save animation (0 = no, 1 = yes), takes long time

# initialize
# ----------

state       = np.array([1.9, 0.1, 1, 0.2, 0.3, 0.4])   # format: [x, xdot, y, ydot, z, zdot]
inputs      = 0*np.array([0.21, 0.15, 0.1])              # format: [xddot, yddot, zddot]
#target     = np.array([0.0,0.0,1.0])                 # format: [xr, yr, zr]
target      = 10*np.array([random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]) 
outputs     = np.array([state[0],state[2],state[4]])   # format: [x, y, z]
error       = outputs - target
trial_cost  = LA.norm(error)
reward      = 1/LA.norm(error)
t           = Ti
i           = 1
nSteps          = int(Tf/Ts+1)
   
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
Tl              = 3    # length of trial [s]
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
target_rand0 = 5*random.uniform(-1, 1)      # used for pseudo-random path gen
target_rand1 = 5*random.uniform(-1, 1)
target_rand2 = 5*random.uniform(-1, 1)

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
    
    return state_dot


#%% Start the Simulation
# ----------------------

while round(t,3) < Tf:

    
    # store results
    t_all[i]                = t
    states_all[i,:]         = state
    inputs_all[i,:]         = inputs 
    targets_all[i,:]        = target
    rewards_all[i,:]        = reward       
    costs_all[i,:]          = trial_cost 
    explore_rates_all[i,:]  = explore_rate 

    # evolve the states through the dynamics
    state = integrate.odeint(dynamics, state, np.arange(t, t+Ts, Tz), args = (inputs,))[-1,:]

    # # store results
    # t_all[i]                = t
    # states_all[i,:]         = state
    # inputs_all[i,:]         = inputs 
    # targets_all[i,:]        = target
    # rewards_all[i,:]        = reward       
    # costs_all[i,:]          = trial_cost 
    # explore_rates_all[i,:]  = explore_rate                      

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
   
    
# %% Results
# -----------
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

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.grid()
axis = 10
ax.set_xlim3d([-axis, axis])
ax.set_ylim3d([-axis, axis])
ax.set_zlim3d([-axis, axis])

line, = ax.plot([], [],[], 'bo-',ms=10, lw=2)
line_target, = ax.plot([], [],[], 'ro-', ms=5, lw=2)

time_template = 'Time = %.1f/%.0fs'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
time_text2 = ax.text2D(0.65, 0.95, 'Q-Learning Control', transform=ax.transAxes)
time_text3 = ax.text2D(0.65, 0.90, 'Controller: PD', transform=ax.transAxes)

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
    time_text.set_text(time_template%(i*Ts,Tf))
    #text_reward.set_text(reward_template%rewards_all[i])
    line2.set_data(t_all[0:i],(1/float(max(costs_all[0:i])))*costs_all[0:i,0])
    ax4.set_xlim(0,t_all[i]+1)
    line3.set_data(t_all[0:i],(1/float(max(explore_rates_all[0:i])))*explore_rates_all[0:i,0])
    ax5.set_xlim(0,t_all[i]+1)
    return line, line2, line3, time_text

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

