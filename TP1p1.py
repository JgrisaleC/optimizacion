#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:14:58 2019

@author: jp
"""

import numpy as np
from numpy.linalg import norm
from derivadas import gradiente,hessiano
#from time import time #linux
from timeit import default_timer as time #windows
from f_tp_1 import *
import pylab as plt

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
        
        if(norm(q) >= e): #p y q son reales.
            
            H = H + (np.outer(p,p.T))/(p.T@q) - (H@((np.outer(q,q.T))@H))/(q.T@(H@q))
        
            k = k+1
        
            xk = x_next
            
        else:
            return 10e8, np.inf

    return k,xk

def punto_cauchy(f,x,Dk):
    
    gk = gradiente(f,x)
    Bk = hessiano(f,x)
    
    gkTBkgk = gk.T@(Bk@gk)
    min_cuadratica = np.power(norm(gk),3)/(Dk*gkTBkgk)
    
    tk = 1 if gkTBkgk <= 0 else np.min([min_cuadratica,1])
    
    dk = -tk*(Dk*gk/norm(gk))

    return dk

def eval_mejora_m(f,xk,dk):
    
    #modelo:
    qk = lambda x: f(xk) + gradiente(f,xk).T@(x-xk) + 0.5*(x-xk).T@(hessiano(f,xk)@(x-xk))
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
        elif(rhok == np.nan or rhok == np.inf):
            return 10e8,np.inf
                    
        k = k + 1
        
    return k,xk

f = [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20]
x0 = puntos_iniciales
kmax = 1000
e = 10e-5

#¡¡¡¡¡¡¡¡¡¡¡¡¡ DESCOMENTAR PARA VOLVER A HALLAR LOS MINIMOS NUEVAMENTE!!!!!!!!!!!!!!!!
#↓            ↓                ↓               ↓              ↓            ↓

#kgradiente = np.zeros(21)
#tgradiente = np.zeros(21)
#xgradiente = np.zeros((21))
##
#kgradconj = np.zeros(21)
#tgradconj = np.zeros(21)
#xgradconj = np.zeros((21))
#
#knewton = np.zeros(21)
#tnewton = np.zeros(21)
#xnewton = np.zeros((21))
#
#kregconf = np.zeros(21)
#tregconf = np.zeros(21)
#xregconf = np.zeros((21))
##
#for i in range(7,21):
    
#    start = time()
#    kgradiente[i], xg = metodo_del_gradiente(f[i],x0[i],e,kmax)
#    xgradiente[i] = norm(xg)
#    tgradiente[i] = time()-start
    
#    start = time()
#    kgradconj[i], xgc = metodo_gradiente_conjugados(f[i],x0[i],10e-8,kmax)
#    xgradconj[i] = norm(xgc)
#    tgradconj[i] = time()-start
    
#    H0 = np.eye(len(x0[i]))
#    start = time()
#    knewton[i], xn =  metodo_cuasi_newton(f[i],x0[i],H0,e,kmax)
#    xnewton[i] = norm(xn)
#    tnewton[i] = time()-start
#    
#    start = time()
#    kregconf[i], xr = Region_de_confianza(f[i],x0[i],e,kmax)
#    xregconf[i] = norm(xr)
#    tregconf[i] = time()-start
##    
#    print(i)


#np.savez('resultados_gradiente.npz',k=kgradiente,x=xgradiente,t=tgradiente)
#np.savez('resultados_gradiente_conjugado.npz',k=kgradconj,x=xgradconj,t=tgradconj)
#np.savez('resultados_newton.npz',k=knewton,x=xnewton,t=tnewton)
#np.savez('resultados_region_confianza.npz',k=kregconf,x=xregconf,t=tregconf)

#resultados_gradiente = np.load('resultados_gradiente.npz')
#tgradiente = resultados_gradiente['t']
#kgradiente = resultados_gradiente['k']
#resultados_gradiente_conjugado = np.load('resultados_gradiente_conjugado.npz')
#tgradconj = resultados_gradiente_conjugado['t']
#kgradconj = resultados_gradiente_conjugado['k']
#resultados_newton = np.load('resultados_newton.npz')
#tnewton = resultados_newton['t']
#knewton = resultados_newton['k']
#resultados_region_confianza = np.load('resultados_newton.npz')
#tregconf = resultados_newton['t']
#kregconf = resultados_newton['k']
#

c = np.matrix([tgradiente,tgradconj,tnewton,tregconf]).T

ck = np.matrix([kgradiente,kgradconj,knewton,kregconf]).T 
    
C = np.where(ck < kmax,c,10e8) # equivalente a Cij = c if ck < kmax else Cij = 10e8 

R = [C[i,:]/np.min(C[i,:]) for i in range(C.shape[0])]

R = np.matrix(R)

nproblemas = C.shape[0]
nalgoritmos = C.shape[1]

tau = 60
rho = np.zeros((tau,4))

for t in range(tau):
    rho[t,:] = np.sum(np.where(R<=t,1,0),axis = 0)

rho /= nproblemas

plt.figure("profile")
plt.plot(np.linspace(1,tau,tau),rho[:,0],label = "Gradiente")
plt.plot(np.linspace(1,tau,tau),rho[:,1],label = "Gradiente Conjugado")
plt.plot(np.linspace(1,tau,tau),rho[:,2],label = "Cuasi Newton")
plt.plot(np.linspace(1,tau,tau),rho[:,3],label = "Región de Confianza")
#
plt.legend(loc = 4, fontsize = 'x-large')

plt.grid()
plt.xticks([1,10,20,30,40,50,60], fontsize = 'xx-large')
plt.yticks(np.linspace(0,1,5), fontsize = 'xx-large')
plt.xlim([1,tau])
