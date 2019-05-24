#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:14:58 2019

@author: jp
"""

import numpy as np
from fx_test import langermann
from numpy.linalg import norm
from numpy.random import uniform
from derivadas import gradiente,hessiano
from time import time
from f_tp_1 import *

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
    k = 0
    
    while(True):
        k = k + 1
        if(f(x + t*d) > f(x) + c1*t*gradiente(f,x).T@d):
            B = t
            t = 0.5*(a + B)
        
        elif(np.gradient(x + t*d).T@d < c2*gradiente(f,x).T@d):
            
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
        
    return k,x

def metodo_gradiente_conjugados(f,x0,e,kmax,metodo = Wolfe):
    
    d = -gradiente(f,x0)
    k = 0
    xk = x0
    n = len(xk)
    
    while(norm(gradiente(f,xk)) > e and k < kmax):
        
        tk = metodo(f,xk,d)        
        xk_next = xk + tk*d

        if(np.remainder(k+1,n)!=0):            
            B = gradiente(f,xk_next).T@gradiente(f,xk_next)/(gradiente(f,xk).T@gradiente(f,xk)) #formula de Fletcher y Reeves
            
        else:            
            B = 0
                        
        d = -gradiente(f,xk_next) + B*d
        k = k + 1
        xk = xk_next

    return k,xk

def metodo_cuasi_newton(f,x0,H0,e,kmax,metodo = Armijo):
    
    k = 0
    xk = x0
    H = H0
    
    while(norm(gradiente(f,xk)) > e and k < kmax):
        
        d = -H@gradiente(f,xk)
        tk = metodo(f,xk,d)
        x_next = xk + tk*d
        
        p = x_next - xk
        q = gradiente(f,x_next) - gradiente(f,xk)
        
        if(norm(q) >= 10e-5): #p y q son reales.
            
            H = H + (np.outer(p,p.T))/(p.T@q) - (H@(np.outer(q,q.T))@H)/(q.T@H@q)
        
            k = k+1
        
            xk = x_next
            
        else:
            return xk

    return k,xk

def punto_cauchy(f,x,Dk):
    
    gk = gradiente(f,x)
    Bk = hessiano(f,x)
    
    gkTBkgk = gk.T@Bk@gk
    min_cuadratica = np.power(norm(gk),3)/(Dk*gkTBkgk)
    
    tk = 1 if gkTBkgk <= 0 else np.min([min_cuadratica,1])
    
    dk = -tk*(Dk*gk/norm(gk))

    return dk

def eval_mejora_m(f,xk,dk):
    
    #modelo:
    qk = lambda x: f(xk) + gradiente(f,xk).T@(x-xk) + 0.5*(x-xk).T@hessiano(f,xk)@(x-xk)
    mk = lambda d: qk(xk + d)
    
    ared = f(xk) - f(xk + dk)
    pred = mk(0) - mk(dk)

    return ared/pred

def Region_de_confianza(f,x0,e,kmax,D0=1,n=0.2):
    
    k = 0
    Dk = D0
    xk = x0
    
    while(norm(gradiente(f,xk)) > e and k < kmax):
        
        dk = punto_cauchy(f,xk,Dk)
        rhok = eval_mejora_m(f,xk,dk)
        
        if(rhok > n):
            xk = xk + dk
        
        if(rhok < 0.25):
            Dk *= 0.5
        elif(rhok > 0.75 and norm(dk) == Dk):
            Dk *= 2.0
        
        k = k + 1
        
    return k,xk


f = [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23]
x0 = puntos_iniciales
kmax = 1000
e = 10e-5

kgradiente = np.zeros(24)
xgradiente = np.zeros((24,2))

kgradconj = np.zeros(24)
xgradconj = np.zeros((24,2))

knewton = np.zeros(24)
xnewton = np.zeros((24,2))

kregconf = np.zeros(24)
xregconf = np.zeros((24,2))

H0 = np.eye(2)

for i in range(24):
    
    kgradiente[i], xgradiente[i,:] = metodo_del_gradiente(f[i],x0[i],e,kmax)
    kgradconj[i], xgradconj[i,:] = metodo_gradiente_conjugados(f[i],x0[i],e,kmax)
    knewton[i], xnewton[i,:] =  metodo_cuasi_newton(f[i],x0[i],H0,e,kmax)
    kregconf[i], xregconf[i,:] = Region_de_confianza(f[i],x0[i],e,kmax)
   
#Armar matriz con 24 filas que corresponden a los problemas a solucionar y 
#4 columnas que son el numero de algoritmos empleados
#llenar matriz con el numero de iteraciones, esta matriz es c
#r_sp = cada_k_en_fila_de_c/k_minimo_en_fila_c -> repertir en cada fila
#thao = minimo_k_en_columna_de_c
# (k_en_fila_de_r_sp < thao)/numero_de_filas_de_c -> repetir en cada columna
#-----------------------------2b-----------------------------------------------
kmax = 300
e = 10e-7
       
k0, A0 = Region_de_confianza(f[i],x0[i],e,kmax,D0=1,n=0.2)
k1, A1 = Region_de_confianza(f[i],x0[i],e,kmax,D0=1.55,n=0.3)
k2, A2 = Region_de_confianza(f[i],x0[i],e,kmax,D0=0.55,n=0.15)
k3, A3 = Region_de_confianza(f[i],x0[i],e,kmax,D0=2,n=0.01)
    