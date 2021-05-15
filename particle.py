#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implementation of classical double-integrator kinematics for a particle in 3D Cartesian space.

A simple PD controller is used to track a moving target

"You've been hit by... a smooth particle"

Created on Sat May  8 13:48:10 2021

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

#%% Simulation Setup
# --------------------
 
Ti = 0.0        # initial time
Tf = 60         # final time 
Ts = 0.1        # sample time
Tz = 0.005      # integration step size

state   = np.array([1.9, 0.1, 1, 0.2, 0.3, 0.4])   # format: [x, xdot, y, ydot, z, zdot]
inputs  = np.array([0.21, 0.15, 0.1])              # format: [xddot, yddot, zddot]
target  = np.array([0,0,1])                        # format: [xr, yr, zr]
outputs = np.array([state[0],state[2],state[4]])   # format: [x, y, z]
error   = outputs - target
t       = Ti
i       = 1
counter = 0  
nSteps  = int(Tf/Ts+1)
        
t_all           = np.zeros(nSteps)                  # to store times
t_all[0]        = Ti                                # store initial time
states_all      = np.zeros([nSteps, len(state)])    # to store states
states_all[0,:] = state                             # store initial state


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

    # increment 
    t += Ts
    i += 1

    # move target 
    target = np.array([5*np.sin(i*Ts*0.2),5*np.cos(0.5*i*Ts*0.2),1])

    # controller (PD type)
    kp = 1
    kd = 0.4
    outputs = np.array([state[0],state[2], state[4]]) 
    derror = (1/Ts)*((outputs-target) - error)
    error = outputs-target
    inputs = - kp*(error) - kd*(derror)
   

# %% Animate
# ---------- 

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.grid()
axis = 5
ax.set_xlim3d([-axis, axis])
ax.set_ylim3d([-axis, axis])
ax.set_zlim3d([-axis, axis])

line, = ax.plot([], [],[], 'o-', lw=2)
time_template = 'Time = %.1fs'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
time_text2 = ax.text2D(0.65, 0.95, 'Double Integrator Kinematics', transform=ax.transAxes)
time_text3 = ax.text2D(0.65, 0.90, 'Controller: PD', transform=ax.transAxes)


def update(i):
    line.set_data(states_all[i,0],states_all[i,2])
    line.set_3d_properties(states_all[i,4])
    time_text.set_text(time_template%(i*Ts))
    return line, time_text

ani = animation.FuncAnimation(fig, update, np.arange(1, len(states_all)),
    interval=15, blit=False)

#ani.save('animation.gif', writer=writer)
#plt.show()


