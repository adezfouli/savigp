import math
from GPy.util.linalg import mdot

__author__ = 'AT'

import numpy as np

class MoG_Diag:
    def __init__(self, num_comp, num_process, num_dim):
        self.num_comp = num_comp
        self.num_process = num_process
        self.num_dim = num_dim
        self.n = num_comp * num_process * num_dim
        self.invC_klj = np.empty((self.num_comp, self.num_comp, self.num_process, self.num_dim))

        self.m =[]
        self.s = []
        self.pi = []
        self._random_init()
        # self.fixed_init()

    def _random_init(self):
        self.m = np.random.uniform(low=-1.0, high=1.0, size=(self.num_comp, self.num_process, self.num_dim))
        self.s = np.random.uniform(low=0.1, high=1.0, size=(self.num_comp, self.num_process, self.num_dim))
        self.pi = np.random.uniform(low=1.0, high=10.0, size=self.num_comp)
        self.pi = self.pi / sum(self.pi)

    def fixed_init(self):
        self.m = np.zeros((self.num_comp, self.num_process, self.num_dim))
        self.s = np.ones((self.num_comp, self.num_process, self.num_dim))
        self.pi = [1./self.num_comp] * self.num_comp

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.s = sa.reshape((self.num_comp, self.num_process, self.num_dim))

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

    def __str__(self):
        return 'm:' + str(self.m) + '\n' + 's:' + str(self.s) + '\n' + 'pi:' + str(self.pi)

    def mmTS(self, k, j):
        return mdot(self.m[k,j], self.m[k,j].T) + np.diag(self.s[k,j])

    def update(self):
        for k in range(self.num_comp):
            for l in range(self.num_comp):
                for j in range(self.num_process):
                    self.invC_klj[k,l,j] = self.inv_cov(j, k, l)