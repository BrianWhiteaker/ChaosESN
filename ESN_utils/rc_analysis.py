#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sept 1 14:26:00 2020

@author: brian
"""
import numpy as np
import pdb
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.special import rel_entr
from scipy.optimize import fsolve
import time
import torch

EPS = np.finfo(np.float).eps

def NRMSE(target, pred, normalizer):
    rmse = mean_squared_error(target, pred, squared=False)
    return rmse/normalizer


def distribution(dataP,dataQ,mn,mx, bins=100):
    emptyFlag = []
    measureP,bnlocs = np.histogram(dataP, bins=bins, range=(mn,mx))
    measureQ,_ = np.histogram(dataQ, bins=bins, range=(mn,mx))
    for i in range(bins):
        if((measureP[i] == 0) or (measureQ[i] == 0)):
            emptyFlag.append(False)
        else:
            emptyFlag.append(True)
    measureP = measureP[emptyFlag]
    measureQ = measureQ[emptyFlag]
    pmfP = (measureP)/np.linalg.norm(measureP)
    pmfQ = (measureQ)/np.linalg.norm(measureQ)
    kl = np.sum(rel_entr(pmfP,pmfQ))
    return (kl,pmfP,pmfQ, bnlocs)


def get_mats(fname, k,n): #nxk, nxn
    #pdb.set_trace()
    wnet = np.load(fname)
    win = wnet[:,:k]
    wres = wnet[:,k:n+k]
    return wres.reshape(n,n), win.reshape(n,k)

def leaky_jacobian(xeq, u, alpha, amp, wi, wr):
    '''
    u is Kx1, x is Nx1
    '''
    n = wr.shape[0]
    z = np.dot(wi,u) + np.dot(wr,xeq)
    return (1-alpha)*np.identity(n)+alpha*amp*np.multiply(wr, (1/np.cosh(z))**2)

def partial_u(xeq, u, alpha, amp, wi, wr):
    #pdb.set_trace()
    z = np.dot(wi,u) + np.dot(wr,xeq)
    return alpha*amp*np.multiply(wi, (1/np.cosh(z))**2)

def reachable_matrix(A, B):
    cols = []
    currentA = A
    n = A.shape[0]
    cols.append(np.dot(np.identity(n),B))
    cols.append(np.dot(A,B))
    for i in range(2,n):
        currentA = np.dot(currentA,A)
        cols.append(np.dot(currentA, B))
    return np.hstack(cols)

def gpu_reachable_matrix(A, B, n=None):
    cols = []
    if(n):
        pass
    else:
        n = A.shape[0]
    
    Atens, Btens = torch.from_numpy(A).float().cuda(),\
                   torch.from_numpy(B).float().cuda()
    currentA = Atens.detach().clone()
    I_n = torch.eye(A.shape[0]).cuda() 
    cols.append(torch.matmul(I_n, Btens).detach().cpu().numpy())
    cols.append(torch.matmul(Atens, Btens).detach().cpu().numpy())
    for i in range(2,n):
        currentA = torch.matmul(currentA,Atens)
        cols.append(torch.matmul(currentA, Btens).detach().cpu().numpy())
    #print(f'time: {time.time()-start}')
    return np.hstack(cols)

def rank_along_trajectory(wr, wi, a, g, forcing, n, k, tolerance):
    T = forcing.shape[0]
    ranks = np.zeros(T)
    def Func(x):
        return np.squeeze(-x.reshape(n,1) + (1-a)*x.reshape(n,1) +\
                          a*g*np.tanh(np.dot(wr,x.reshape(n,1)) +\
                         (np.dot(wi,ueq.reshape(k,1)).reshape(n,1))))
    for i in range(T):
        ueq = forcing[i]
        x0 = np.ones((n,1))*.5
        xeq = (fsolve(Func,x0)).reshape(n,1)
        A = leaky_jacobian(xeq, ueq.reshape(k,1), a, g, wi, wr)
        B = partial_u(xeq, ueq.reshape(k,1), a, g, wi, wr)
        Cplus = gpu_reachable_matrix(A, B)
        ranks[i] = rank(Cplus, tolerance)
        if(i%10==0):
            print(f'Column:{i}')
    return ranks

def propagate_F(f, x0, u0, steps):
    x = x0
    u = u0
    test = []
    for i in range(steps):
        x = f(x,u)
        test.append(x)
    return test

def eig_spectrum(wr):
    return np.linalg.eig(wr)[0]

def rank(w, tol=1e-15):
    return np.linalg.matrix_rank(w, tol)

def cos_sim(target, pred):
    return np.squeeze(cosine_similarity(np.expand_dims(target,0), np.expand_dims(pred,0)))
    
def condition_number(wr):
    return np.linalg.cond(wr,np.infty),np.linalg.cond(wr,-np.infty)

def gains(w):
    return np.linalg.norm(w, np.infty), np.linalg.norm(w, -np.infty) 

def cobweb(f, x0, N, eq, a=-1, b=1):
        # plot the function f being iterated
        plt.figure(figsize=(10,10))
        t = np.linspace(a, b, N)
        plt.plot(t, f(t), 'k', label='tanh(k+wx)')
        plt.plot(t, t, "k:", label='y=x')
        # plot equilibrium point
        plt.plot(eq,eq, 'r', ls='', marker='1', ms=20, label='eq')
        plt.vlines(x0,a,b, 'r')
        # plot the iterates
        x, y = x0, f(x0)
        for _ in range(N):
            fy = f(y)        
            plt.plot([x, y], [y,  y], 'b', linewidth=1)
            plt.plot([y, y], [y, fy], 'b', linewidth=1)
            x, y = y, fy
        plt.xlabel('x(t)')
        plt.ylabel('x(t+1)')
        plt.axes().set_aspect(1)
        plt.legend()
        plt.show()
        plt.close()
        
def bifurcation_plot(x0, u, alpha, amp, w):
    '''
    Example usage:
        T = 13
        signal = 1.5*np.sin((np.arange(100))/T)
        milestone = [x/10 for x in range(-5,5)]
        
        plt.figure(figsize=(10,10))
        for w in np.arange(-5., 5., 0.0001):
            if(w in milestone):
                print(f'Working on {w}')
            xInitial = np.random.uniform(-1.,1.)
            bifurcation_plot(xInitial, signal, 0.8,1., w)
        plt.xlabel('w')
        plt.ylabel('x')
        plt.vlines([-1.0,1.0],-1,1, 'r')
        plt.show()
    
    '''
    
    result = []
    x = x0
    for t in range(100): # first 100 steps are discarded 
        x = (1-alpha)*x + alpha*amp*np.tanh(u[t] + w*x)
    for t in range(100): # second 100 steps are collected 
        x = (1-a)*x + alpha*amp*np.tanh(u[t] + w*x)
        result.append(x)
    c = 'b'
    plt.plot([w] * 100, result,c=c, ls='', marker='.', ms=1, alpha = 0.1)
        
