'''
IMPLEMENTATIONS OF POWERS METHOD (AND VARIATIONS)
TO FIND EIGENVALUES AND EIGENVETORS
Specifically:
    - Standard powers method
    - Powers method for symmetric matices
    - Inverse matrix method variant of power method
    - Aitken applied to standard powers method to accelerate convergence
'''

import numpy as np
import scipy as sp


#ITERATION FUNCTIONS:
    
#Standard powers method iteration
def iterate (A, x):
    p = np.argmax(abs(x))
    y = A @ x
    mu = y[p]
    nuovo_p = np.argmax(abs(y))
    if y[nuovo_p] == 0:
        mu = 0
    else:
        nuovo_x = y/y[nuovo_p]
    return nuovo_x, mu


#Aitken to powers method iteraion
def iterate_ait (A, x):
    p0 = np.argmax(abs(x)) #resituisce l'indice dell'elemento massimo (il primo)
    if x[p0] == 0:
        print("lambda0 è 0")
        return 0, 0, 0, x
    x0 = x/x[p0] #x è un vettore colonna, ma per python è matrice
    y0 = A @ x0
    lambda0 = y0[p0]
    
    p1 = np.argmax(abs(y0))
    if y0[p1] == 0:
        print("lambda1 è 0")
        return lambda0, 0, 0, y0
    x1 = y0/y0[p1]
    y1 = A @ x1
    lambda1 = y1[p1]
    
    p2 = np.argmax(abs(y1))
    if y1[p2] == 0:
        print("lambda2 è 0")
        return lambda0, lambda1, 0, y1
    x2 = y1/y1[p2]
    y2 = A @ x2
    lambda2 = y2[p2]
    
    return lambda0, lambda1, lambda2, y2


#--------------------------------------------------#
#MAIN FUNCTIONS:

#Standard powers method
def eig_powers (A, x0, tol, n_max_iter):
    x1, mu1 = iterate (A, x0) 
    n_iterazioni = 0
    while (np.linalg.norm(x1 - x0) > tol) and (n_iterazioni < n_max_iter):
        nuovo_x, nuovo_mu = iterate(A, x1)
        x0, x1 = x1, nuovo_x
        if nuovo_mu == 0:
            break
        n_iterazioni = n_iterazioni + 1
    return x1, nuovo_mu, n_iterazioni
    

#Powers method: symmetric matrices variant
def eig_powers_sym(A,x,tol,max_it):
    k=1
    y=A@x
    lambda1=x.transpose()@y/(x.transpose()@x)
    p=np.argmax(abs(x))
    x=y/y[p]
    #x0=x
    lambda1_0=lambda1
    while k<=max_it:
        y=A@x
        lambda1=x.transpose()@y/(x.transpose()@x)
        p=np.argmax(abs(x))
        x=y/y[p]
        if lambda1==0:
            break
        if abs(lambda1-lambda1_0)<tol: #
            break 
        #x0=x
        lambda1_0=lambda1
        k=k+1
    return lambda1,x,k


#Matrix method for minumun eigenvalue
def eig_powers_inv(A,x,tol,max_it):
    k=1
    p=np.argmax(abs(x)) #resituisce l'indice dell'elemento massimo (il primo)
    x=x/x[p] #x è un vettore colonna, ma per python è matrice
    while k<=max_it:
        y=sp.linalg.solve(A,x)# y=A^{-1}*x  equivalente a Ay=x
        lambda1=y[p]
        p=np.argmax(abs(y))
        if y[p]==0:
            lambda1=x
            break
        x0=x
        x=y/y[p]
        if np.linalg.norm(x-x0)<tol:
            break 
        k=k+1
    return 1/lambda1,x,k-1

        
#Aitken + standard powers method
def eig_powers_ait (A, x, tol, max_it):
    n_iter = 0
    lambda0, lambda1, lambda2, y = iterate_ait(A,x)
    
    if lambda2 - 2*lambda1 + lambda0 == 0:
        return lambda2, y, n_iter
    
    lambdatilde2 = (lambda2*lambda0-lambda1**2) / (lambda2-2*lambda1+lambda0)
    
    while n_iter <= max_it:
        lambdatilde1 = lambdatilde2
        lamnda0, lambda1, lambda2, y = iterate_ait(A, y)
        
        if lambda2 - 2*lambda1 + lambda0 == 0:
            return lambda2, y, n_iter
        
        lambdatilde2 = (lambda2*lambda0-lambda1**2) / (lambda2 - 2*lambda1 + lambda0)
        n_iter = n_iter+1
        
        if abs(lambdatilde2 - lambdatilde1) < tol:
            break
    return lambdatilde2, y, n_iter
        