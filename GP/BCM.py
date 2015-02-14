# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from numpy.linalg import inv
from numpy.ma import diag
import pylab as pb
from GPy.core.gp import GP
from GPy.util.linalg import mdot


class BCM_GP(GP):

    def __init__(self, X, likelihood, kernel, normalize_X, sigma, p):
        GPBase.__init__(self, X, likelihood, kernel, normalize_X=normalize_X)
        self.p = p
        self.sigma = sigma

    def get_mu_approx(self, U, D, kern, sigma):
        return self.BCM_mu(U, D, kern, sigma)

    def get_var_approx(self, U, D, kern, sigma):
        return self.BCM_var(U, D, kern, sigma)

    def get_mu_approx_dist(self, U, D, kern, sigma):
        p = [0] * len(U)
        for i, u in enumerate(U):
            p[i] = self.BCM_mu(np.array([u]), D, kern, sigma)[0]
        return np.array(p)

    def get_var_approx_dist(self, U, D, kern, sigma):
        p = [0] * len(U)
        for i, u in enumerate(U):
            p[i] = self.BCM_var(np.array([u]), D, kern, sigma)[0]
        return np.array(p)

    def get_mu(self, Xs, X, Y, kern, sigma):
        return mdot(kern.K(Xs, X), inv(sigma * np.eye(len(X)) + kern.K(X, X)), Y)

    def get_var(self, Xs, X, kern, sigma):
        v = (kern.K(Xs, Xs) - mdot(kern.K(Xs, X), inv(sigma * np.eye(len(X)) + kern.K(X, X)), kern.K(X, Xs)))
        return v

    def get_kernel_approx(self, U, X, kern, sigma):
        return inv(mdot(kern.K(U, X), kern.K(X, U), inv(kern.K(U, U))) + sigma * np.eye(len(U)))

    def parition_data(self, X, y, N):
        D = np.zeros((len(X), 2))
        D[:, 1] = y[:, 0]
        D[:, 0] = X[:, 0]
        return np.split(D, N)

    def BCM_var(self, Xs, D, kern, sigma):
        cov = -(self.p - 1.0) * inv(kern.K(Xs, Xs))
        for Di in D:
            cov += inv(self.get_var(Xs, Di[:, [0]], kern, sigma))

        return inv(cov)

    def BCM_mu(self, Xs, D, kern, sigma):
        var = self.BCM_var(Xs, D, kern, sigma)
        cov = np.zeros((len(Xs), len(Xs)))
        for Di in D:
            cov += mdot(inv(self.get_var(Xs, Di[:, [0]], kern, sigma)), self.get_mu(Xs, Di[:, [0]], Di[:, [1]], kern, sigma))
        return mdot(var, cov)

    def _raw_predict(self, _Xnew, which_parts='all', full_cov=False, stop=False):
        D = self.parition_data(self.X, self.likelihood.Y, self.p)
        var = self.get_var_approx_dist(_Xnew, D, self.kern, self.sigma)
        mu = self.get_mu_approx_dist(_Xnew, D, self.kern, self.sigma)
        return mu, var