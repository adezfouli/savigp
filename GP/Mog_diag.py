__author__ = 'AT'

import math
from GPy.util.linalg import mdot
from MoG import MoG
import numpy as np


class MoG_Diag(object, MoG):

    def __init__(self, num_comp, num_process, num_dim):
        MoG.__init__(self, num_comp, num_process, num_dim)
        self.invC_klj = np.empty((self.num_comp, self.num_comp, self.num_process, self.num_dim))
        self.s = []
        self._random_init()
        self.parameters = self.get_parameters()
        self._update()
        self.num_free_params = self.parameters.shape[0]

    def get_parameters(self):
        return np.hstack([self.m.flatten(), self.s.flatten(), self.pi])

    def num_parameters(self):
        return self.num_free_params

    def _random_init(self):
        super(MoG_Diag, self)._random_init()
        self.s = np.random.uniform(low=0.1, high=1.0, size=(self.num_comp, self.num_process, self.num_dim))

    def fixed_init(self):
        super(MoG_Diag, self).fixed_init()
        self.s = np.ones((self.num_comp, self.num_process, self.num_dim))

    def transform_S_grad(self, g):
        return g.flatten() * self.s.flatten()

    def get_s_size(self):
        return self.num_comp * self.num_process * self.num_dim

    def S_dim(self):
        return (self.num_dim,)

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.s = np.exp(sa).reshape((self.num_comp, self.num_process, self.num_dim))

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
               math.log(math.sqrt(np.product(2 * math.pi * (self.s[l, j, :] + self.s[k, j, :]))))

    def inv_cov(self, j, k, l):
        return 1. / (self.s[l, j, :] + self.s[k, j, :])

    def tr_A_mult_S(self, A, k, j):
        return np.dot(np.diagonal(A), self.s[k,j,:])

    def C_m(self, j, k, l):
        return self.invC_klj[k, l, j] * (self.m[k, j, :] - self.m[l, j, :])


    def C_m_C(self, j, k, l):
        return (self.invC_klj[k, l, j] -
                self.invC_klj[k, l, j] * (self.m[k, j, :] - self.m[l, j, :]) *
                (self.m[k, j, :] - self.m[l, j, :]) * self.invC_klj[k, l, j])

    def aSa(self, a, j):
        return mdot(self.s[:,j,:], (a ** 2))

    def mmTS(self, k, j):
        return mdot(self.m[k,j], self.m[k,j].T) + np.diag(self.s[k,j])

    def dAS_dS(self, A):
        return np.diag(A)

    def Sa(self, a, k, j):
        return a * self.s[k,j]

    def _update(self):
        for k in range(self.num_comp):
            for l in range(self.num_comp):
                for j in range(self.num_process):
                    self.invC_klj[k,l,j] = self.inv_cov(j, k, l)