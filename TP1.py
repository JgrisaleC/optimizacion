# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:14:54 2019

@author: Administrador
"""

import numpy as np
from fx_test import langermann
from numpy.linalg import norm
from numpy.random import uniform
from derivadas import gradiente
#from time import time #linux
from timeit import default_timer as time #windows
import pandas as pd

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
	
    while(norm(gradiente(f,x))>e and k<kmax):
        
        d = -gradiente(f,x)
        t = metodo(f,x,d)
        x = x + t*d
        k = k+1
        
    return x

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

    return xk

def metodo_cuasi_newton(f,x0,H0,e,kmax,metodo):
    
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
            
            H = H + (np.outer(p,p.T))/(p.T@q) - (H@(np.outer(q,q.T))@H)/(q.T@H@q)
        
            k = k+1
        
            xk = x_next
            
        else:
            print("precision alcanzada: ")
            print(e)
            return xk

    return xk

f = langermann
e = 10e-5 #precision
kmax = 750 #cantidad maxima de iteraciones
H0 = np.eye(2)

#Separar espacio en memoria
x_min_gradiente = np.empty((150,2))
x_min_gradconjugados = np.empty((150,2))
x_min_cuasi_newton = np.empty((150,2))

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------


#¡¡¡¡¡¡¡¡¡¡¡¡¡ DESCOMENTAR PARA VOLVER A HALLAR LOS MINIMOS!!!!!!!!!!!!!!!!
#↓            ↓                ↓               ↓              ↓            ↓


#Random Reestart----------------------------------------------------------------------------------------
A = uniform(low = -5.0, high = 5.0, size = (150,2)) # 150 puntos iniciales generados de manera aleatoria

inicio_gradiente = time()
for i, x0 in enumerate(A): #enumerate permite iterar sobre algo y agrega un contador automaticamente
    x_min_gradiente[i,:] = metodo_del_gradiente(f,x0,e,kmax,seccion_aurea)
    print(i)
fin_gradiente = time() - inicio_gradiente
print(fin_gradiente)
    
inicio_gconj = time()
for j, x0 in enumerate(A):    
    x_min_gradconjugados[j,:] = metodo_gradiente_conjugados(f,x0,e,kmax,seccion_aurea)
    print(j)
fin_gconj = time() - inicio_gconj 
print(fin_gconj)    

inicio_qnewton = time()
for k, x0 in enumerate(A):
    x_min_cuasi_newton[k,:] = metodo_cuasi_newton(f,x0,H0,e,kmax,seccion_aurea)
fin = time() - inicio_qnewton
print(fin)

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#------------------DESCOMENTAR PARA VOLVER A HALLAR LOS MINIMOS------------------------------
#-------------------------------------------------------------------------------------------------------

#----- Guardo valores de una prueba ----------------
#np.save('metodo_gradiente.npy',x_min_gradiente)
#np.save('metodo_gradiente_conjugado.npy',x_min_gradconjugados)
#np.save('metodo_cuasi_newton.npy',x_min_cuasi_newton)

#---Cargo valores de una prueba anterior para no correr todo otra vez
#x_min_gradiente = np.load('metodo_gradiente.npy')
#x_min_gradconjugados = np.load('metodo_gradiente_conjugado.npy')
#x_min_cuasi_newton = np.load('metodo_cuasi_newton.npy')

#Evaluar los minimos hallados en la funcion de langermann
fx_min_gradiente = np.apply_along_axis(f,1,x_min_gradiente)
fx_min_gradconjugados = np.apply_along_axis(f,1,x_min_gradconjugados)
fx_min_cuasi_newton = np.apply_along_axis(f,1,x_min_cuasi_newton)

#Guardo todo en data frames para mejor visualizacion
resultados_gradiente = pd.DataFrame({'x1* ': x_min_gradiente[:,0],
                                     'x2* ': x_min_gradiente[:,1],
                                     'f(x*)': fx_min_gradiente,
                                        })
print(resultados_gradiente)


resultados_gradiente_conjugado = pd.DataFrame({'x1* ': x_min_gradconjugados[:,0],
                                     'x2* ': x_min_gradconjugados[:,1],
                                     'f(x*)': fx_min_gradconjugados,
                                        })    
print(resultados_gradiente_conjugado)


resultados_cuasi_newton = pd.DataFrame({'x1* ': x_min_cuasi_newton[:,0],
                                     'x2* ': x_min_cuasi_newton[:,1],
                                     'f(x*)': fx_min_cuasi_newton,
                                        })
print(resultados_cuasi_newton)