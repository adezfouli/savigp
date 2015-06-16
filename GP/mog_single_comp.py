__author__ = 'AT'

from scipy.linalg import cho_solve, solve_triangular
from mog import MoG
from GPy.util.linalg import mdot
import math
import numpy as np
from util import chol_grad, pddet, jitchol, tr_AB


class MoG_SingleComponent(MoG):
    """
    Implementation of posterior distribution when it is a single component Gaussian distribution with full
    covariance matrix.

    Attributes
    ----------
    L : ndarray
     Chlesky decomposition of the covariance matrix
    """

    def __init__(self, num_process, num_dim):
        MoG.__init__(self, 1, num_process, num_dim)
        self.invC_klj = np.empty((self.num_comp, self.num_comp, self.num_process, self.num_dim, self.num_dim))
        self.m = []
        self.pi = []
        self.L_flatten = np.empty((self.num_comp, self.num_process, self.get_sjk_size()))
        self.s = np.empty((self.num_comp, self.num_process, self.num_dim, self.num_dim))
        self.L = np.empty((self.num_comp, self.num_process, self.num_dim, self.num_dim))
        self.log_det = np.empty((self.num_comp, self.num_comp, self.num_process))
        self._fixed_init()
        self._update()
        self.num_free_params = self.parameters.shape[0]

    def log_pdf(self, j, k, l):
        return -((self.s[0, j, :, :].shape[0])/2) * (math.log(2 * math.pi) + math.log(2.0)) - \
               0.5 * pddet(self.L[0,j,:])

    def aSa(self, a, k, j):
        return np.diagonal(mdot(a, self.s[k,j,:,:], a.T))

    def mmTS(self, k, j):
        return mdot(self.m[k,j,:,np.newaxis], self.m[k,j,:,np.newaxis].T) + self.s[k,j]

    def Sa(self, a, k, j):
        return mdot(self.s[k,j], a)

    def transform_eye_grad(self):
        """
        In the case of posterior distribution with one component, gradients of the
        entropy term wrt to the posterior covariance is identity. This function returns flatten lower-triangular terms
        of the identity matrices for all processes.
        """
        grad = np.empty((self.num_comp, self.num_process, self.get_sjk_size()))
        meye = np.eye((self.num_dim))[np.tril_indices_from(self.L[0,0])]
        for k in range(self.num_comp):
            for j in range(self.num_process):
                grad[k,j] = meye
        return grad.flatten()

    def get_parameters(self):
        return np.hstack([self.m.flatten(), self.L_flatten.flatten(), self.pi_untrans])

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

    def _fixed_init(self):
        MoG._fixed_init(self)
        for k in range(self.num_comp):
            for j in range(self.num_process):
                self.L_flatten[k,j,:] = np.random.uniform(low=1.0, high=1.0, size=self.get_sjk_size())

    def _random_init(self):
        MoG._random_init(self)
        # self.m = np.zeros((self.num_comp, self.num_process, self.num_dim))
        for k in range(self.num_comp):
            for j in range(self.num_process):
                self.L_flatten[k,j,:] = np.random.uniform(low=1.1, high=5.0, size=self.get_sjk_size())

    def get_sjk_size(self):
        # return self.num_dim
        return self.num_dim * (self.num_dim + 1) / 2

    def get_s_size(self):
        return self.num_comp * self.num_process * self.get_sjk_size()

    def S_dim(self):
        return self.num_dim, self.num_dim

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.L_flatten = sa.reshape((self.num_comp, self.num_process, self.get_sjk_size()))

    # def tr_A_mult_S(self, A, k, j):
    #     return trace(cho_solve((A, True), self.s[k,j]))
    def get_m_S_params(self):
        return self.m, self.L_flatten

    def tr_AinvS(self, L, k, j):
        a = solve_triangular(L, self.L[k, j, :], lower=True)
        return tr_AB(a.T, a)

    def tr_AS(self, A, k, j):
        return tr_AB(A, self.s[k, j, :])

    def dAinvS_dS(self, L, k, j):
        tmp = 2 * cho_solve((L, True), self.L[k,j])
        tmp[np.diag_indices_from(tmp)] *= self.L[k,j][np.diag_indices_from(tmp)]
        return tmp[np.tril_indices_from(self.L[k,j])]

    def dAS_dS(self, S, k, j):
        tmp = 2 * mdot(S, self.L[k,j])
        tmp[np.diag_indices_from(tmp)] *= self.L[k,j][np.diag_indices_from(tmp)]
        return tmp[np.tril_indices_from(self.L[k,j])]

    def transform_S_grad(self, g):
        r"""
        Assume:

        g = df \\ dS

        then this function returns:
        :returns df \\ dL, where L is the Cholesky decomposition of S
        """

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
                # temp[np.diag_indices_from(temp)] = temp[np.diag_indices_from(temp)] ** 2
                self.L[k,j,:,:] = temp
                self.s[k,j] = mdot(self.L[k,j,:,:], self.L[k,j,:,:].T)
