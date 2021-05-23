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
