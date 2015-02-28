__author__ = 'AT'

from GPy.util.linalg import mdot
import math
from numpy.ma import trace
from MoG_full import MoG_Full
import numpy as np
from util import chol_grad, pddet, inv_chol, jitchol


class MoG_SingleComponent(MoG_Full):

    def __init__(self, num_process, num_dim):
        MoG_Full.__init__(self, 1, num_process, num_dim)

    def ratio(self, j, k, l1, l2):
        return 1.0

    def log_pdf(self, j, k, l):
        return -((self.s[0, j, :, :].shape[0])/2) * (math.log(2 * math.pi) + math.log(2.0)) - \
               0.5 * pddet(self.L[0,j,:])

    def inv_cov(self, j, k, l):
        return  0.5 * inv_chol(self.L[0,j,:])

    def tr_A_mult_S(self, A, k, j):
        return trace(mdot(A.T, self.s[k,j]))

    def aSa(self, a, j):
        return mdot(a.T, self.s[:,j,:,:], a)

    def mmTS(self, k, j):
        return mdot(self.m[k,j,:,np.newaxis], self.m[k,j,:,np.newaxis].T) + self.s[k,j]

    def dAS_dS(self, A):
        return A

    def Sa(self, a, k, j):
        return mdot(self.s[k,j], a)

    # def update_covariance(self, j, Sj):
    #     Sj = Sj.copy()
    #     mm = min(Sj[np.diag_indices_from(Sj)])
    #     if mm < 0:
    #         Sj[np.diag_indices_from(Sj)] = Sj[np.diag_indices_from(Sj)] - 1.1 * mm
    #     for k in range(self.num_comp):
    #         self.s[k,j] = Sj.copy()
    #         self.L[k,j] = jitchol(Sj,10)
    #         tmp = self.L[k,j].copy()
    #         tmp[np.diag_indices_from(tmp)] = np.sqrt(tmp[np.diag_indices_from(tmp)])
    #         self.L_flatten[k,j] = tmp[np.tril_indices_from(tmp)]
    #     self._update()


    # def transform_S_grad(self, g):
    #     grad = np.empty((self.num_comp, self.num_process, self.get_sjk_size()))
    #     for k in range(self.num_comp):
    #         for j in range(self.num_process):
    #             tmp = chol_grad(self.L[k,j], g[k,j])
    #             tmp[np.diag_indices_from(tmp)] *= self.L[k,j][ np.diag_indices_from(tmp)]
    #             tmp[np.diag_indices_from(tmp)] *= 2 * np.sqrt(self.L[k,j][ np.diag_indices_from(tmp)])
    #             grad[k,j] = tmp[np.tril_indices_from(self.L[k,j])]
    #     return grad.flatten()

    def _update(self):
        self.parameters = self.get_parameters()
        for k in range(self.num_comp):
            for j in range(self.num_process):
                temp = np.zeros((self.num_dim, self.num_dim))
                temp[np.tril_indices_from(temp)] = self.L_flatten[k,j,:].copy()
                temp[np.diag_indices_from(temp)] = np.exp(temp[np.diag_indices_from(temp)])
                # temp[np.diag_indices_from(temp)] = temp[np.diag_indices_from(temp)] ** 2
                self.L[k,j,:,:] = temp
                self.s[k,j] = mdot(self.L[k,j,:,:], self.L[k,j,:,:].T)

        for k in range(self.num_comp):
            for l in range(self.num_comp):
                for j in range(self.num_process):
                    self.invC_klj[k,l,j] = self.inv_cov(j, k, l)
