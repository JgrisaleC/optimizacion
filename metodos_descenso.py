# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:14:54 2019

@author: Administrador
"""

import numpy as np
import fx_test
from numpy.linalg import solve,eigvals,norm
from derivadas import gradiente,hessiano

def seccion_aurea(f,x,d,e = 10e-5,r=1):
    
    theta1 = (3 - np.sqrt(5))/2
    theta2 = 1 - theta1
    a = 0
    s = r
    b = 2*r
    phib = f(x + b*d)
    phis = f(x + s*s)
    
    while(phib < phis):
        a = s
        s = b
        b *= 2
        phib = f(x + b*d)
        phis = f(x+ s*d)
    
    u = a + theta1*(b - a)
    v = a + theta2*(b - a)
    
    phiu = f(x + u*d)
    phiv = f(x + v*d)
    
    while((b-a)>e):
        if(phiu < phiv):
            b = u
            v = u
            u = a + theta1*(b - a)
            phiv = phiu
            phiu = f(x + u*d)
        else:
            a = u
            u = v
            v = a + theta2(b - a)
            phiu = phiv
            phiv = f(x + v*d)
    
    t = (u+v)/2
    
    return t

def Armijo(f,x,d,g=0.7,n=0.45):
    
    t = 1
    
    while(f(x + t*d) > f(x) + n*t*gradiente(f,x).T@d):
        
        t = g*t

    return t

def Wolfe(f,x,d,c1=0.5,c2=0.75):
    a = 0
    t = 1
    B = np.inf
    
    while(True):
        
        if(f(x + t*d) > f(x) + c1*t*gradiente(f,x).T@d):
            B = t
            t = 0.5*(a + B)
        
        elif(np.gradient(x + t*d).T@d < c2*gradiente(f,x).T@d):
            
            a = t                       
            t = 2*a if B == np.inf else 0.5*(a+B)
        
        else:
            break
        
    return t

def metodo_del_gradiente(f,x0,e,kmax,metodo):
    
    k= 0
    x = x0
	
    while(gradiente(f,x).all()>e and k<kmax):
        
        d = -gradiente(f,x)
        t = metodo(f,x,d)
        x = x + t*d
        k = k+1
        
    return x
		
def metodo_de_newton(f,x0,e,kmax,metodo):
    
    k = 0
    x = x0
    
    while(gradiente(f,x).all()>e and k<kmax):
       
        d = solve(hessiano(f,x),-gradiente(f,x))
        t = metodo(f,x,d)
        x = x + t*d
        k = k + 1
        
    return x
        
def metodo_de_Levenberg_Marquardt(f,x0,e,kmax,g,metodo):
    
    
    k = 0
    x = x0
    
    while(norm(gradiente(f,x)) > e and k < kmax):
    
        B = hessiano(f,x)
        mu = np.min(eigvals(B))
        
        if(mu<=0):
            B += (-mu + g)*np.eye(B.shape[0])
		
        d = solve(B,-gradiente(f,x))
        t = metodo(f,x,d)
        x = x + t*d
        k = k + 1
        
    return x
    	

f = fx_test.rosenbrock

x_min_gradiente = metodo_del_gradiente(f,np.array([2,2]),0.01,1000,Armijo)
x_min_newton = metodo_de_newton(f,np.array([2,2]),0.01,100,Armijo)
x_min_LV = metodo_de_Levenberg_Marquardt(f,np.array([2,2]),0.0001,0.5,1000,Armijo)