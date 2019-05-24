# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:18:35 2019

@author: Administrador
"""

from numpy.linalg import norm,solve
import numpy as np

def derivada_parcial(f,a,i):
    h = 0.0001
    ei = np.zeros(a.shape[0])
    ei[i] = 1
   
    z = (f(a+h*ei) -  f(a-h*ei))/(2*h)
               
    return z        

def gradiente(f,a):
    
    g = [derivada_parcial(f,a,i) for i in range(a.shape[0])]
    
    return np.array(g)  

def hessiano(f,a):
    
    n = len(a)
    H = np.empty((n,n))
    
    for i in range(n):
    
        DfDx = lambda x: derivada_parcial(f,x,i)        
        H[i,:] = gradiente(DfDx,a)
    
    return H

    
#f = lambda x: -x[1] + x[0]**3 
#
#a = np.array([6,3])
#
##DfDx = derivada_parcial(f,a,0)
#
#G = gradiente
#H = hessiano
#
#x = solve(H(f,a),G(f,a))
#Gx = norm(G(f,x)) 
#
#while Gx > 0.01:
#    
#    d = solve(H(f,x),-G(f,x))
#    x += d
#    Gx = norm(G(f,x))
    
   