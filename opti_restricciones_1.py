# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:57:34 2019

@author: Administrador
"""

from numpy.linalg import norm
from derivadas import gradiente
import numpy as np


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
            v = a + theta2*(b - a)
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
    k = 0
    
    while(True):
        if(k > 10000):
            break
        k = k + 1
        if(f(x + t*d) > (f(x) + c1*t*gradiente(f,x).T@d)):
            B = t
            t = 0.5*(a + B)
        
        elif(gradiente(f,x + t*d).T@d < c2*gradiente(f,x).T@d):
            
            a = t                       
            t = 2*a if B == np.inf else 0.5*(a+B)
            
        else:
            break
        
    return t

def metodo_del_gradiente(f,x0,e,kmax,metodo = Armijo):
    
    k= 0
    x = x0
	
    while(norm(gradiente(f,x))>e and k<kmax):
        
        d = -gradiente(f,x)
        t = metodo(f,x,d)
        x = x + t*d
        k = k+1
        
    return x

def metodo_de_penalizacion(f,G,H,x0,c0=1.5,a=2,e=10e-3,kmax=1000):
    
    x = x0
    c = c0
    k = 0
    
    nG = range(len(G))
    nH = range(len(H))
    
    A = lambda a:np.array(a)
    Gx_r = lambda x: [np.max([0,G[i](x)]) for i in nG]
    Gx = lambda x: [G[i](x) for i in nG]
    Hx = lambda x: [np.power(H[i](x),2) for i in nH]
    Q = lambda x: f(x) + c*(np.sum(Gx_r(x)) + np.sum(Hx(x)))
     
    x_next = metodo_del_gradiente(Q,x,e,kmax)
    
    if((A(Gx(x))<=0).all() and np.isclose(A(Hx(x)),0).all()):
        return x_next
    else:
        c *= a
        k += 1
   
    while(norm(x - x_next)>e and k<kmax):
        print(k)
        x = np.copy(x_next)
        x_next = metodo_del_gradiente(Q,x,e,kmax)
        
        if((A(Gx(x))<=0).all() and np.isclose(A(Hx(x)),0).all()):
            return x_next
        else:
            c *= a
            k += 1
            
    return x

def metodo_de_barrera(f,G,x0,u0=1.0,a=0.5,e=10e-3,kmax=1000):
    
 
    x = x0
    u = u0
    k = 0
    nG = range(len(G))
    
    Gx = lambda x: np.array([G[i](x) for i in nG])
    B = lambda x: np.sum(1/-Gx(x))
    R = lambda x: f(x) +u*B(x)
    
    x_next = metodo_del_gradiente(R,x,e,kmax)
    
    while(norm(x - x_next)>e and k<kmax):
        print(k)
        x = np.copy(x_next)
        x_next = metodo_del_gradiente(R,x,e,kmax)
        u *= a
        k += 1
        
    return x
        

f = lambda x: 0.1*np.power((x[1]+x[2]-3),2)  
G = (lambda x:np.sum(x**2)-1,)
H = (lambda x: 0,)

x0 = np.array([0.1,0.8,0.8])

xm1 = metodo_de_penalizacion(f,G,H,x0,c0=1.5,a=2,e=10e-5,kmax=100)
xm2 = metodo_de_barrera(f,G,x0,u0=1.0,a=0.5,e=10e-5,kmax=100)

f = lambda x: x[0] + x[1]
G = (lambda x: -x[0], )
H = (lambda x: x[0]**2 + x[1]**2 - 2, )

x0 = np.array([0.0,-1.0])

xm3 = metodo_de_penalizacion(f,G,H,x0,c0=1.5,a=2,e=10e-5,kmax=100)
