
__author__ = 'AT'

from GPy.util.linalg import mdot
import math
from numpy.ma import trace
import numpy as np
from util import chol_grad, pddet, inv_chol, jitchol
from mog_full import MoG_Full


class MoG_SingleComponent(MoG_Full):

    def __init__(self, num_process, num_dim):
        MoG_Full.__init__(self, 1, num_process, num_dim)

    def ratio(self, j, k, l1, l2):
        return 1.0

    def log_pdf(self, j, k, l):
        return -((self.s[0, j, :, :].shape[0])/2) * (math.log(2 * math.pi) + math.log(2.0)) - \
               0.5 * pddet(self.L[0,j,:])

    def aSa(self, a, j):
        return mdot(a.T, self.s[:,j,:,:], a)

    def mmTS(self, k, j):
        return mdot(self.m[k,j,:,np.newaxis], self.m[k,j,:,np.newaxis].T) + self.s[k,j]

    def Sa(self, a, k, j):
        return mdot(self.s[k,j], a)

    def transform_eye_grad(self):
        grad = np.empty((self.num_comp, self.num_process, self.get_sjk_size()))
        meye = np.eye((self.num_dim))[np.tril_indices_from(self.L[0,0])]
        for k in range(self.num_comp):
            for j in range(self.num_process):
                grad[k,j] = meye
        return grad.flatten()


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
