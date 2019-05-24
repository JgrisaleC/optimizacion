# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:29:06 2019

@author: Administrador
"""
import numpy as np
from numpy.linalg import norm

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
            
def Cholesky(A):    
    
    L = np.copy(A)
    
    for k in range(A.shape[0]):
        for i in range(k):
            s = 0
            for j in range(i):
                s = s + L[i,j]*L[k,j]
                
            L[k,i] = (L[k,i] - s)/L[i,i]
        s = 0        
        for j in range(k):
            s = s + L[k,j]**2.0
        
        L[k,k] = np.sqrt(L[k,k] -s)
    
    L[np.where(L==A)] = 0.0
    
    return L   
  