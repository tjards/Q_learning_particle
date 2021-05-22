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
import random
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

#%% Simulation Setup
# --------------------
 
Ti = 0.0        # initial time
Tf = 500         # final time 
Ts = 0.1        # sample time
Tz = 0.01      # integration step size

state   = np.array([1.9, 0.1, 1, 0.2, 0.3, 0.4])   # format: [x, xdot, y, ydot, z, zdot]
inputs  = np.array([0.21, 0.15, 0.1])              # format: [xddot, yddot, zddot]
#target  = np.array([0.0,0.0,1.0])                  # format: [xr, yr, zr]
target  = 9*np.array([random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]) 

outputs = np.array([state[0],state[2],state[4]])   # format: [x, y, z]
error   = outputs - target
trial_cost  = LA.norm(error)
reward      = 1/LA.norm(error)
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
rewards_all      = np.zeros([nSteps, 1])  # to store rewards
rewards_all[0,:] = reward
costs_all      = np.zeros([nSteps, 1])  # to store costs
costs_all[0,:] = trial_cost


# initialize Q table
nParams     = 2    # number of parameters to tune
nOptions    = 10   # number of options to selection from (ranges between 0 and nOptions)
scale       = 1    # scale the value of the options (i.e. scale*[0:nOptions])
Tl          = 2    # length of trial [s]
trial_counter = 0  # initialize counter 
trial_cost    = 0  # initialze cost 
explore_rate  = 1  # how often to explore, 0 to 1 (start high, decrease)
Q = QL.init(nParams,nOptions)

# select initial parameters from Q table 
kp_i = QL.select(Q,0,nOptions,explore_rate)
kp = scale*kp_i
kd_i = QL.select(Q,1,nOptions,explore_rate)
kd = scale*kd_i
explore_rates_all       = np.zeros([nSteps, 1])  # to store explore rates
explore_rates_all[0,:]  = explore_rate


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
    costs_all[i,:]      = trial_cost #1/np.maximum(trial_cost,0.00001)
    explore_rates_all[i,:]  = explore_rate                      

    # increment 
    t += Ts
    i += 1

    # wander the target 
    target = 9*np.array([1*np.sin(i*Ts*3),1*np.cos(0.5*i*Ts*3),1*np.sin(i*Ts*2)])

    # still working on this
    if round(trial_counter,3) < Tl:
    
        # accumulate the cost
        trial_cost += LA.norm(target-state[0:6:2])
    
        # increment the counter 
        trial_counter += Ts
        
    else: 
        
        # normalize the cost
        trial_cost = trial_cost/Tl
        
        #compute the reward
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
        
        # select new target (randomly)
        target  = 10*np.array([random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]) 
        
        # reduce the explore rate
        explore_rate = 0.99**t
        #print(explore_rate)
        


    # controller (PD type)
    #kp = 2
    #kd = 1.4
    outputs = np.array([state[0],state[2], state[4]]) 
    derror = (1/Ts)*((outputs-target) - error)
    error = outputs-target
    inputs = - kp*(error) - kd*(derror)
   

# %% Animate
# ---------- 

#%% trajectory

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
#reward_template = 'Last Reward = %.1f'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
time_text2 = ax.text2D(0.65, 0.95, 'Q-Learning Control', transform=ax.transAxes)
time_text3 = ax.text2D(0.65, 0.90, 'Controller: PD', transform=ax.transAxes)
#text_reward = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)

# create inset axis
#==================
ax4 = plt.axes([0,0,1,1])
# set position manually (x pos, y pos, x len, y len)
#ip = InsetPosition(ax, [0.58,0.58,0.4,0.4])
ip = InsetPosition(ax, [0,0.7,0.2,0.2])
ax4.set_axes_locator(ip)
# cool, make lines to point (save this for later)
#mark_inset(ax2, ax3, loc1=2, loc2=4, fc="none", ec='0.5')
# data
line2, = ax4.plot([], [], '--', c='b', mew=2, alpha=0.8,label='Cost')
#ax4.plot([], [], '-', c='g', mew=2, alpha=0.8,label='Explore rate')


# create inset axis
#==================
ax5 = ax4.twinx()
#ax5 = plt.axes([0,0,1,1])
# set position manually (x pos, y pos, x len, y len)
#ip2 = InsetPosition(ax, [0.18,0.18,0.4,0.4])
ax5.set_axes_locator(ip)
# cool, make lines to point (save this for later)
#mark_inset(ax2, ax3, loc1=2, loc2=4, fc="none", ec='0.5')
# data
line3, = ax4.plot([], [], '-', c='g', mew=2, alpha=0.8,label='Explore')
#ax4.plot([], [], '-', c='g', mew=2, alpha=0.8,label='Explore rate')



# add another plot?
#ax5 = plt.axes([0,0,1,1])
#ax5.set_axes_locator(ip)
#ax5.plot(t_all[1::int(Tl/Ts)],explore_rates_all[1::int(Tl/Ts),0],'--', c='g', mew=2, alpha=0.8,label='Explore Rate')

#ax4.plot(t_all[1::int(Tl/Ts)],(1/float(max(costs_all)))*costs_all[1::int(Tl/Ts),0],'--', c='b', mew=2, alpha=0.8,label='Cost')
#ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Cost')
ax4.set_xlim(0,max(t_all))
ax4.set_ylim(0,1)
ax4.tick_params(axis='y', labelcolor='b')

#ax5.set_xlabel('Time [s]')
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

#slow
#ani2 = animation.FuncAnimation(fig, update, np.linspace(1, len(states_all), num=500, dtype=int),interval=15, blit=False)

ani.save('animation.gif', writer=writer, progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))
#ani.save('animation.gif', writer=writer)


#plt.show()

#%% Costs

# plot costs
fig2, ax2 = plt.subplots()
plt.title('Q-Learning Control Parameters')
ax2.plot(t_all[1::int(Tl/Ts)],costs_all[1::int(Tl/Ts),0],'-', c='b', mew=2, alpha=0.8,label='Cost')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Cost [m]')
#ax2.legend(loc=0)
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
ax3.plot(t_all[1::int(Tl/Ts)],explore_rates_all[1::int(Tl/Ts),0],'--', c='g', mew=2, alpha=0.8,label='Explore Rate')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Explore Rate')
ax3.set_xlim(0,max(t_all))
ax3.set_ylim(0,1)


plt.savefig('cost.png')
# other stuff
#ax2.set_ylim(0,26)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax3.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
#ax3.tick_params(axis='x', which='major', pad=8)