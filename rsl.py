# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.linalg import norm
def forwardsub(A,b):
    
    x = np.zeros(b.shape)
    x[0] = b[0]/A[0,0]
    
    for i in range(1,A.shape[0]):
        s = 0
        for j in range(i-1):
            s = s + (x[j]*A[i,j])/A[i,i]
            x[i] = b[i]  + s
            
    return x 
 
def backwardsub(A,b):
    
    x = np.zeros(b.shape)
    x[-1] = b[-1]/A[-1,-1]
    
    n = b.shape
    
    for i in np.linspace(n,0,n+1):
        s = 0
        for j in np.linspace(n-1,0,n):
            s = s + (x[j]*A[i,j])/A[i,i]
            x[i] = b[i]  + s            
    return x

def QR(A):
    
    n = A.shape
    
    u = np.zeros(n)
    w = np.zeros(n)
    R = np.zeros(n)
    
    u[:,0] = A[:,0].T
    w[:,0] = u[:,0]/norm(u[:,0])
    
    for k in range(1,n[0]):
        proy = 0
        
        for j in range(k):
            proy += (u[:,j]@(A[:,k]))/((norm(u[:,j]))**2)*u[:,j]     
        
        u[:,k] = A[:,k].T - proy
        w[:,k] = u[:,k]/norm(u[:,k])
        
    for l in range(n[0]):
        for m in range(n[1]):
            if(l <= m):
                R[l,m] = w[:,l]@(A[:,m])
              
    return w,R
            
            
A = np.matrix([[4.0, 12.0, -16.0],[12.0, 37.0, -43.0],[-16.0, -43.0, 98.0]])
b = np.array([0,0,0])

Q,R = QR(A)



    