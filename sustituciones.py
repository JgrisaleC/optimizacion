# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def forward(A,b):
    
    x = np.zeros(b.shape)
    x[0] = b[0]/A[0,0]
    
    for i in range(1,A.shape[0]):
        s = 0
        for j in range(i-1):
            s = s + (x[j]*A[i,j])/A[i,i]
            x[i] = b[i]  + s
            
    return x 
 
def backward(A,b):
    
    x = np.zeros(b.shape)
    x[-1] = b[-1]/A[-1,-1]
    
    n = b.shape
    
    for i in np.linspace(n,0,n+1):
        s = 0
        for j in np.linspace(n-1,0,n):
            s = s + (x[j]*A[i,j])/A[i,i]
            x[i] = b[i]  + s            
    return x

