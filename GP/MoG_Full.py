from numpy.ma import trace
from scipy.linalg import cho_solve
from util import chol_grad, pddet, inv_chol, jitchol

__author__ = 'AT'

import math
from GPy.util.linalg import mdot, dtrtrs
from numpy.linalg import inv, det
from MoG import MoG
import numpy as np


class MoG_Full(MoG):

    def __init__(self, num_comp, num_process, num_dim):
        MoG.__init__(self, num_comp, num_process, num_dim)
        self.invC_klj = np.empty((self.num_comp, self.num_comp, self.num_process, self.num_dim, self.num_dim))
        self.m = []
        self.pi = []
        self.L_flatten = np.empty((self.num_comp, self.num_process, self.get_sjk_size()))
        self.s = np.empty((self.num_comp, self.num_process, self.num_dim, self.num_dim))
        self.L = np.empty((self.num_comp, self.num_process, self.num_dim, self.num_dim))
        self.log_det = np.empty((self.num_comp, self.num_comp, self.num_process))
        self._random_init()
        self._update()
        self.num_free_params = self.parameters.shape[0]

    def get_parameters(self):
        return np.hstack([self.m.flatten(), self.L_flatten.flatten(), self.pi])

    def update_covariance(self, j, Sj):
        Sj = Sj.copy()
        mm = min(Sj[np.diag_indices_from(Sj)])
        if mm < 0:
            Sj[np.diag_indices_from(Sj)] = Sj[np.diag_indices_from(Sj)] - 1.1 * mm
        for k in range(self.num_comp):
            self.s[k,j] = Sj.copy()
            self.L[k,j] = jitchol(Sj,10)
            tmp = self.L[k,j].copy()
            tmp[np.diag_indices_from(tmp)] = np.log(tmp[np.diag_indices_from(tmp)])
            self.L_flatten[k,j] = tmp[np.tril_indices_from(tmp)]
        self._update()

    def num_parameters(self):
        return self.num_free_params

    def _random_init(self):
        MoG._random_init(self)
        for k in range(self.num_comp):
            for j in range(self.num_process):
                self.L_flatten[k,j,:] = np.random.uniform(low=1.0, high=1.0, size=self.get_sjk_size())

    def fixed_init(self):
        self.m = np.zeros((self.num_comp, self.num_process, self.num_dim))
        for k in range(self.num_comp):
            for j in range(self.num_process):
                self.L_flatten[k,j,:] = np.random.uniform(low=0.1, high=1.0, size=self.get_sjk_size())
        self.pi = [1./self.num_comp] * self.num_comp

    def get_sjk_size(self):
        # return self.num_dim
        return self.num_dim * (self.num_dim + 1) / 2

    def get_s_size(self):
        return self.num_comp * self.num_process * self.get_sjk_size()

    def S_dim(self):
        return self.num_dim, self.num_dim

    def full_s_dim(self):
        return (self.num_comp, self.num_process, self.num_dim, self.num_dim)

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.L_flatten = sa.reshape((self.num_comp, self.num_process, self.get_sjk_size()))

    def ratio(self, j, k, l1, l2):
        e = mdot((self.m[k, j, :] - self.m[l1, j, :]).T, inv((self.s[l1, j, :] + self.s[k, j, :])),
                   (self.m[k, j, :] - self.m[l1, j, :]))
        e -= mdot((self.m[k, j, :] - self.m[l2, j, :]).T, inv((self.s[l2, j, :] + self.s[k, j, :])),
                    (self.m[k, j, :] - self.m[l2, j, :]))
        return math.exp(-0.5 * e + (self.log_det[k,l2,j] - self.log_det[k,l1,j]) / 2)

    def log_pdf(self, j, k, l):
        return -0.5 * mdot(
            mdot(self.m[k, j, :] - self.m[l, j, :]).T, inv ((self.s[l, j, :] + self.s[k, j, :])),
            (self.m[k, j, :] - self.m[l, j, :]))  \
               -((self.s[l, j, :, :].shape[0])/2) * math.log(2 * math.pi)  - \
               0.5 * self.log_det[k,l,j]

    def inv_cov(self, j, k, l):
        return inv(self.s[l, j, :] + self.s[k, j, :])

    def tr_A_mult_S(self, A, k, j):
        return trace(cho_solve((A, True), self.s[k,j]))

    def C_m(self, j, k, l):
        return mdot(self.invC_klj[k, l, j], (self.m[k, j, :] - self.m[l, j, :]))

    def C_m_C(self, j, k, l):
        return (self.invC_klj[k, l, j] -
                mdot(self.invC_klj[k, l, j], (self.m[k, j, :] - self.m[l, j, :]),
                (self.m[k, j, :] - self.m[l, j, :]).T, self.invC_klj[k, l, j]))

    def aSa(self, a, j):
        return mdot(a.T, self.s[:,j,:,:], a)

    def mmTS(self, k, j):
        return mdot(self.m[k,j,:,np.newaxis], self.m[k,j,:,np.newaxis].T) + self.s[k,j]

    def dAS_dS(self, A):
        return A

    def Sa(self, a, k, j):
        return mdot(self.s[k,j], a)


    def transform_S_grad(self, g):
        grad = np.empty((self.num_comp, self.num_process, self.get_sjk_size()))
        for k in range(self.num_comp):
            for j in range(self.num_process):
                tmp = chol_grad(self.L[k,j], g[k,j])
                tmp[np.diag_indices_from(tmp)] *= self.L[k,j][ np.diag_indices_from(tmp)]
                grad[k,j] = tmp[np.tril_indices_from(self.L[k,j])]
        return grad.flatten()

    def _update(self):
        self.parameters = self.get_parameters()
        for k in range(self.num_comp):
            for j in range(self.num_process):
                temp = np.zeros((self.num_dim, self.num_dim))
                temp[np.tril_indices_from(temp)] = self.L_flatten[k,j,:].copy()
                temp[np.diag_indices_from(temp)] = np.exp(temp[np.diag_indices_from(temp)])
                self.L[k,j,:,:] = temp
                self.s[k,j] = mdot(self.L[k,j,:,:], self.L[k,j,:,:].T)

        for k in range(self.num_comp):
            for l in range(self.num_comp):
                for j in range(self.num_process):
                    self.invC_klj[k,l,j] = self.inv_cov(j, k, l)

                    self.log_det[k,l,j] = pddet(self.L[k,j]) + math.log(det(np.eye(self.num_dim) +
                                                                            mdot(self.L[l,j].T, dtrtrs(self.L[k,j],
                                                                                                       dtrtrs(self.L[k,j],
                                                                                                              self.L[l,j])[0])[0].T) ))
