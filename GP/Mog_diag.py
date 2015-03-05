__author__ = 'AT'

import math
from GPy.util.linalg import mdot
from MoG import MoG
import numpy as np

class MoG_Diag(MoG):

    def __init__(self, num_comp, num_process, num_dim):
        MoG.__init__(self, num_comp, num_process, num_dim)
        self.invC_klj_Sk = np.empty((self.num_comp, self.num_comp, self.num_process, self.num_dim))
        self.s = []
        self._random_init()
        self._update()
        self.num_free_params = self.parameters.shape[0]

    def get_parameters(self):
        return np.hstack([self.m.flatten(), self.log_s.flatten(), self.pi])

    def num_parameters(self):
        return self.num_free_params

    def _random_init(self):
        MoG._random_init(self)
        self.s = np.random.uniform(low=1.0, high=1.0, size=(self.num_comp, self.num_process, self.num_dim))
        self.log_s = np.log(self.s)

    def update_covariance(self, j, Sj):
        for k in range(self.num_comp):
            self.s[k,j,:] = np.diagonal(Sj).copy()
            if min(self.s[k,j,:]) < 0:
                self.s[k,j,:] = self.s[k,j,:] - 2 * min(self.s[k,j,:])

        self.log_s = np.log(self.s)
        self._update()

    def fixed_init(self):
        super(MoG_Diag, self).fixed_init()
        self.s = np.ones((self.num_comp, self.num_process, self.num_dim))

    def transform_S_grad(self, g):
        return g.flatten() * self.s.flatten()

    def get_s_size(self):
        return self.num_comp * self.num_process * self.num_dim

    def full_s_dim(self):
        return (self.num_comp, self.num_process, self.num_dim,)

    def S_dim(self):
        return (self.num_dim,)

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.s = np.exp(sa).reshape((self.num_comp, self.num_process, self.num_dim))
        self.log_s = sa.reshape((self.num_comp, self.num_process, self.num_dim))

    def ratio(self, j, k, l1, l2):
        e = np.dot((self.m[k, j, :] - self.m[l1, j, :]) * (1.0 / (self.s[l1, j, :] + self.s[k, j, :])),
                   (self.m[k, j, :] - self.m[l1, j, :]))
        e -= np.dot((self.m[k, j, :] - self.m[l2, j, :]) * (1.0 / (self.s[l2, j, :] + self.s[k, j, :])),
                    (self.m[k, j, :] - self.m[l2, j, :]))
        dets = np.product((self.s[l2, j, :] + self.s[k, j, :]) / (self.s[l1, j, :] + self.s[k, j, :]))
        return math.exp(-0.5 * e) * math.sqrt(dets)

    def log_pdf(self, j, k, l):
        return -0.5 * np.dot(
            (self.m[k, j, :] - self.m[l, j, :]) * (1.0 / (self.s[l, j, :] + self.s[k, j, :])),
            (self.m[k, j, :] - self.m[l, j, :])) - \
            self.s[l,j,:].shape[0] * .5 * math.log(2 * math.pi) - (0.5 * (np.log((self.s[l, j, :] + self.s[k, j, :]))).sum())

    def tr_A_mult_S(self, A, k, j):
        return np.dot(np.diagonal(A), self.s[k,j,:])

    def C_m(self, j, k, l):
        return (self.m[k, j, :] - self.m[l, j, :]) / (self.s[l, j, :] + self.s[k, j, :])


    def C_m_C(self, j, k, l):
        return (self.invC_klj_Sk[k, l, j] -
                self.invC_klj_Sk[k, l, j] * (self.m[k, j, :] - self.m[l, j, :]) *
                (self.m[k, j, :] - self.m[l, j, :]) * self.invC_klj_Sk[k, l, j])

    def aSa(self, a, j):
        return mdot(self.s[:,j,:], (a ** 2))

    def mmTS(self, k, j):
        return mdot(self.m[k,j, np.newaxis].T, self.m[k,j, np.newaxis]) + np.diag(self.s[k,j])

    def dAS_dS(self, A):
        return np.diag(A)

    def Sa(self, a, k, j):
        return a * self.s[k,j]

    def _update(self):
        self.parameters = self.get_parameters()
        for k in range(self.num_comp):
            for l in range(self.num_comp):
                for j in range(self.num_process):
                    self.invC_klj_Sk[k,l,j] = self._s_k_skl(k,l,j)

    def _s_k_skl(self, k, l, j):
        s_k_skl = np.empty((self.num_dim))
        for n in range(self.num_dim):
            if self.log_s[k, j, n] > self.log_s[l, j, n]:
                s_k_skl[n] = 1. / (1. + np.exp((self.log_s[l, j, n] - self.log_s[k, j, n])))
            else:
                s_k_skl[n] = np.exp((-self.log_s[l, j, n] + self.log_s[k, j, n])) / (1.  + np.exp((-self.log_s[l, j, n]   +  self.log_s[k, j, n])))
        return s_k_skl