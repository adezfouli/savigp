from DerApproximator import get_d1
from GPy.util.linalg import mdot
from numpy.linalg import inv
from ExtRBF import ExtRBF
import numpy as np

Dim = 5
N = 10
M = 6
kernel = ExtRBF(Dim, variance=2, lengthscale=1, ARD=False)

X = np.random.normal(0, 1, Dim * N) \
            .reshape((N, Dim))

Z = np.random.normal(0, 1, Dim * M) \
            .reshape((M, Dim))


O = np.random.normal(0, 1,  M) \
            .reshape((M, 1))

n = 1

def A_n(Z_f):
    Z_shaped = Z_f.reshape((M, Dim))
    return mdot(kernel.K(X[np.newaxis, n,:], Z_shaped), inv(kernel.K(Z_shaped, Z_shaped)), O)[0,0]

def A_n_analytical(Z_f):
    Z_shaped = Z_f.reshape((M, Dim))
    An = mdot(kernel.K(X[np.newaxis, n,:], Z_shaped), inv(kernel.K(Z_shaped, Z_shaped)))
    # return kernel.gradients_X_(mdot(inv(kernel.K(Z_shaped, Z_shaped)), O), Z_shaped, X[np.newaxis, n,:])- \
    return        kernel.gradients_X(mdot(An.T, mdot(inv(kernel.K(Z_shaped, Z_shaped)), O).T ), Z_shaped)

def A_n_analytical_vec(Z_f):
    Z_shaped = Z_f.reshape((M, Dim))
    A = mdot(kernel.K(X, Z_shaped), inv(kernel.K(Z_shaped, Z_shaped)))
    An = mdot(kernel.K(X[np.newaxis, n,:], Z_shaped), inv(kernel.K(Z_shaped, Z_shaped)))
    # return (kernel.get_gradients_X_AK(mdot(inv(kernel.K(Z_shaped, Z_shaped)), O).repeat(N, 1), Z_shaped, X)) - \
    return            kernel.get_gradients_X_SKD(A, mdot(inv(kernel.K(Z_shaped, Z_shaped)), O).repeat(N, 1), Z_shaped)[1,:,:]


print get_d1(A_n, Z.flatten()).reshape((M, Dim))

print

print A_n_analytical(Z.flatten())

print A_n_analytical_vec(Z.flatten())

O1 = np.random.normal(0, 1,  M) \
            .reshape((M, 1))

O2 = np.random.normal(0, 1,  M) \
            .reshape((M, 1))

OO = np.random.normal(0, 1,  M * M) \
            .reshape((M, M))


print (mdot(O1, O2.T) * OO).sum(axis=1)

print mdot(OO, O2).T * O1.T
