import mod_eigen_finding as mef
import numpy as np
import scipy as sp


#Symmetric matrix
A = np.array([[3, 2, -2],
             [2, 1, 2],
             [-2, 2, -5]])

x0 = np.array([-1, -3, 2])
tol = 0.00001
n_max_iter = 5000

print('\n\n---------------REAL VALUES----------------------------- \n')
real_eigenvaluesA, real_eigenvectorsA = sp.linalg.eig(A)
print('Eigenvalues for A: ', real_eigenvaluesA)
print('Eigenvectors for A:\n', real_eigenvectorsA)


print('\n\n\n---------------OUR RESULTS:--------------------------- \n')
print('---------------Powers:-------------------------------- \n')
autovettore, autovalore, n_iterazioni = mef.eig_powers(A, x0, tol, n_max_iter)
print("Eigenvector:", end=" ")
print(autovettore)

print("Eigenvalue:", end=" ")
print(autovalore)

print("Iterations:", end=" ")
print(n_iterazioni)

print('\n\n\n---------------Symmetric:---------------------------- \n')
autovettore, autovalore, n_iterazioni = mef.eig_powers_sym(A, x0, tol, n_max_iter)
print("Eigenvector:", end=" ")
print(autovettore)

print("Eigenvalue:", end=" ")
print(autovalore)

print("Iterations:", end=" ")
print(n_iterazioni)


print('\n\n\n---------------Inverse:----------------------------- \n')
autovettore, autovalore, n_iterazioni = mef.eig_powers_inv(A, x0, tol, n_max_iter)
print("Eigenvector:", end=" ")
print(autovettore)

print("Eigenvalue:", end=" ")
print(autovalore)

print("Iterations:", end=" ")
print(n_iterazioni)

print('\n\n\n---------------Aitken:------------------------------ \n')
autovettore, autovalore, n_iterazioni = mef.eig_powers_ait(A, x0, tol, n_max_iter)
print("Eigenvector:", end=" ")
print(autovettore)

print("Eigenvalue:", end=" ")
print(autovalore)

print("Iterations:", end=" ")
print(n_iterazioni)


