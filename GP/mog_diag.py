from util import inv_chol

__author__ = 'AT'

import math
from GPy.util.linalg import mdot
from mog import MoG
import numpy as np

class MoG_Diag(MoG):

    def __init__(self, num_comp, num_process, num_dim):
        MoG.__init__(self, num_comp, num_process, num_dim)
        self.invC_klj_Sk = np.empty((self.num_comp, self.num_comp, self.num_process, self.num_dim))
        self.s = []
        self._fixed_init()
        self._update()
        self.num_free_params = self.parameters.shape[0]

    def get_parameters(self):
        return np.hstack([self.m.flatten(), self.log_s.flatten(), self.pi])

    def num_parameters(self):
        return self.num_free_params

    def _fixed_init(self):
        MoG._fixed_init(self)
        self.s = np.random.uniform(low=0.5, high=0.5, size=(self.num_comp, self.num_process, self.num_dim))
        self.log_s = np.log(self.s)

    def _random_init(self):
        MoG._random_init(self)
        self.s = np.random.uniform(low=1.0, high=3.0, size=(self.num_comp, self.num_process, self.num_dim))
        self.log_s = np.log(self.s)

    def update_covariance(self, j, Sj):
        for k in range(self.num_comp):
            self.s[k,j,:] = np.diagonal(Sj).copy()
            if min(self.s[k,j,:]) < 0:
                self.s[k,j,:] = self.s[k,j,:] - 2 * min(self.s[k,j,:])

        self.log_s = np.log(self.s)
        self._update()

    def transform_S_grad(self, g):
        return g.flatten() * self.s.flatten()

    def get_s_size(self):
        return self.num_comp * self.num_process * self.num_dim

    def full_s_dim(self):
        return (self.num_comp, self.num_process, self.num_dim,)

    def get_sjk_size(self):
        return self.num_dim

    def S_dim(self):
        return (self.num_dim,)

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.s = np.exp(sa).reshape((self.num_comp, self.num_process, self.num_dim))
        self.log_s = sa.reshape((self.num_comp, self.num_process, self.num_dim))

    def tr_Ainv_mult_S(self, L, k, j):
        return np.dot(np.diagonal(inv_chol(L)), self.s[k,j,:])

    def tr_A_mult_S(self, A, k, j):
        return np.dot(np.diagonal(A), self.s[k,j,:])

    def C_m(self, j, k, l):
        return (self.m[k, j, :] - self.m[l, j, :]) / (self.s[l, j, :] + self.s[k, j, :])


    def C_m_C(self, j, k, l):
        return (self.invC_klj_Sk[k, l, j] -
                np.square(self.invC_klj_Sk[k, l, j] * (self.m[k, j, :] - self.m[l, j, :])) / self.s[k,j])

    def aSa(self, a, j):
        return mdot(self.s[:,j,:], (a ** 2))

    def aSkja(self, a, k, j):
        # return mdot(self.s[k, j, :], (a ** 2))
        return np.diagonal(mdot(a, np.diag(self.s[k,j,:]), a.T))

    def mmTS(self, k, j):
        return mdot(self.m[k,j, np.newaxis].T, self.m[k,j, np.newaxis]) + np.diag(self.s[k,j])

    def dAinvS_dS(self, L, k, j):
        return np.diagonal(inv_chol(L)) * self.s[k,j,:].flatten()

    def dAS_dS(self, S, k, j):
        return np.diagonal(S) * self.s[k,j,:].flatten()

    def Sa(self, a, k, j):
        return mdot(np.diag(self.s[k,j]), a)

    def _update(self):
        self.parameters = self.get_parameters()
        for k in range(self.num_comp):
            for l in range(self.num_comp):
                for j in range(self.num_process):
                    self.invC_klj_Sk[k,l,j] = self._s_k_skl(k,l,j)

    def log_sk_sl(self, k, l, j):
        a = np.maximum(self.log_s[k, j, :], self.log_s[l, j, :])
        return a + np.log(np.exp(self.log_s[k, j, :] - a) + np.exp(self.log_s[l, j, :] - a))

    def _s_k_skl(self, k, l, j):
        a = np.maximum(self.log_s[k, j, :], self.log_s[l, j, :])
        return np.exp((-a + self.log_s[k, j, :])) / (np.exp((-a + self.log_s[l, j, :]))  + np.exp((-a + self.log_s[k, j, :])))
