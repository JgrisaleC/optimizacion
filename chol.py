# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:29:06 2019

@author: Administrador
"""
import numpy as np

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
        
a = np.matrix([[4.0, 12.0, -16.0],[12.0, 37.0, -43.0],[-16.0, -43.0, 98.0]])

L = Cholesky(a)        