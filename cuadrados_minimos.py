# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:24:34 2019

@author: Administrador
"""

from descomp_matrices import QR, Cholesky
from sustituciones import forward, backward
import numpy as np


def cuadrados_minimos_con_QR(k,datos):    
    #Resolver AtA*alpha = At*y
    #Donde:
    # A = [x_n^0|x_n^1|x_n^2|...|x_n^k]
    #Datos = {(x_0,y_0),(x_1,y_1),...(x_n,y_n)}
    
    y = datos[:,1]
    
    A = np.zeros((k+1,len(datos)))
    for i in range(k):
        A[:,i] = datos[:,0]^k 
    
    AtA = A.T@A
    At = A.T
    
    #AtA*alpha = At*y 
    #At*y => b
    # alpha => x
    #AtA => A
    #Ax=b <=> Q*Rx=b <=> R*x=Qt*b   
 
    b = At@y
   
    Q,R = QR(AtA) #Descomposicion QR   
    Qtb = Q.T@(b)
    
    alpha = backward(R,Qtb) #Substitucion backward   
    return alpha
    

def cuadrados_minimos_con_Cholesky(k,datos):
    
    y = datos[:,1]
    A = np.zeros((k+1,len(datos)))
    for i in range(k):
        A[:,i] = datos[:,0]^k 
    
    AtA = A.T@A
    At = A.T
    b = At@y
    
    #AtA*alpha = At*y 
    #At*y => b
    # alpha => x
    #AtA => A
    
    L = Cholesky(AtA)
    
    y =  backward(L,b)
    alpha = forward(L.T,y)
    
    return alpha
     