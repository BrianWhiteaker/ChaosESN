#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:18:22 2020

@author: brian
"""
import numpy as np
from sklearn import linear_model
import torch
import pdb

#RC functions
def set_vectors(n,l,r):
    ''' Initializes the x0, y0 vectors'''
    x0 = np.random.uniform(-r,r,(n,1)) # Random init first state
    y0 = np.zeros((l,1)) # Current timestep-> yt for feedback
    return x0, y0

def get_weight_matrices(k,n,l,ro,ri,rf):
    win    = np.random.uniform(-ri,ri,(n,k))
    wfb    = np.zeros((n,l))#np.random.uniform(-rf,rf,(n,l))##
    wout   = np.random.uniform(-ro,ro,(l, n+k)) 
    return win, wfb, wout

def get_trained_weights(states,labels, alpha, intercept=False):
    #print(f's {states.shape}    l {labels.shape}')
    minimizer = linear_model.Ridge(alpha=alpha,fit_intercept=intercept)
    minimizer.fit(states, labels) 
    wout = minimizer.coef_
    return wout

def get_patch(pos, cx,cy, offset, data):
    return data[pos,
                cx-offset:cx+offset+1,
                cy-offset:cy+offset+1]

def get_flatten_patch(p):
    return np.ndarray.flatten(p)[:,np.newaxis]

def update_res_state(wnet,xt,uxy,alpha, gamma, amp):
    N = wnet.shape[0]
    z = np.dot(wnet, uxy)
    return (1-alpha)*xt + amp*alpha*np.tanh(z)

def predict_y(wout,xu):
    return np.dot(wout, xu)

def update_res_stateGPU(wnet,xt,uxy,a,g):
    z = torch.matmul(wnet,uxy)
    return torch.mul(xt,1-a) + torch.mul(torch.tanh(z),a*g)

def predict_yGPU(wout,xu):
    return torch.matmul(wout, xu)
