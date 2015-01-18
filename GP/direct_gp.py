# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from numpy.linalg import inv
from numpy.ma import diag
import pylab as pb
from GPy.core.gp_base import GPBase
from GPy.util.linalg import mdot
from GPy.kern.kern import kern as GP_kern


class DirectGP(GPBase):

    def __init__(self, X, likelihood, kernel, normalize_X, sigma):
        GPBase.__init__(self, X, likelihood, kernel, normalize_X=normalize_X)
        self.sigma = sigma

    def get_mu_approx(self, U, X, Y, kern, sigma):
        return mdot(self.get_kernel_approx(U, X, kern, sigma), kern.K(U, X), Y)

    def get_var_approx(self, U, X, kern, sigma):
        return kern.K(U, U) - mdot(self.get_kernel_approx(U, X, kern, sigma), kern.K(U, X), kern.K(X, U))

    def get_mu_approx_dist(self, U, X, Y, kern, sigma):
        p = [0] * len(U)
        for i, u in enumerate(U):
            p[i] = self.get_mu_approx(np.array([u]), X, Y, kern, sigma)[0]
        return np.array(p)

    def get_var_approx_dist(self, U, X, kern, sigma):
        p = [0] * len(U)
        for i, u in enumerate(U):
            p[i] = self.get_var_approx(np.array([u]), X, kern, sigma)[0]
        return np.array(p)

    def get_mu(self, Xs, X, Y, kern, sigma):
        return mdot(kern.K(Xs, X), inv(sigma * np.eye(len(X)) + kern.K(X, X)), Y)

    def get_var(self, Xs, X, kern, sigma):
        return np.array([diag(kern.K(Xs, Xs) - mdot(kern.K(Xs, X), inv(sigma * np.eye(len(X)) + kern.K(X, X)), kern.K(X, Xs)))]).T

    def get_kernel_approx(self, U, X, kern, sigma):
        return inv(mdot(kern.K(U, X), kern.K(X, U), inv(kern.K(U, U))) + sigma * np.eye(len(U)))

    def _raw_predict(self, _Xnew, which_parts='all', full_cov=False, stop=False):

        mu = self.get_mu_approx_dist(_Xnew, self.X, self.likelihood.Y, self.kern, self.sigma)
        var = self.get_var_approx_dist(_Xnew, self.X, self.kern, self.sigma)

        return mu, var