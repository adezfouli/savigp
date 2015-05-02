__author__ = 'AT'

from GPy.util.linalg import mdot
import numpy as np

class MoG:
    def __init__(self, num_comp, num_process, num_dim):
        self.num_comp = num_comp
        self.num_process = num_process
        self.num_dim = num_dim
        self.m = []
        self.pi = []
        self.parameters = []

    def __str__(self):
        return 'm:' + str(self.m) + '\n' + 's:' + str(self.s) + '\n' + 'pi:' + str(self.pi)

    def update_parameters(self, params):
        self.parameters = params
        self.m_from_array(params[:self.get_m_size()])
        self.s_from_array(params[self.get_m_size():(self.get_m_size() + self.get_s_size())])
        self.pi_from_array(params[(self.get_m_size() + self.get_s_size()):])
        self._update()

    def pi_dim(self):
        return self.num_comp

    def m_dim(self):
        return self.num_comp, self.num_process, self.num_dim

    def _fixed_init(self):
        self.m = np.random.uniform(low=0.0, high=0.0, size=(self.num_comp, self.num_process, self.num_dim))
        self.pi = np.random.uniform(low=1.0, high=5.0, size=self.num_comp)
        self.pi = self.pi / sum(self.pi)

    def transform_S_grad(self, g):
        """ transforms gradients to expose to the optimizer  """
        raise NotImplementedError

    def _random_init(self):
        self.m = np.random.uniform(low=-5.1, high=5.1, size=(self.num_comp, self.num_process, self.num_dim))
        self.pi = np.random.uniform(low=1.0, high=10.0, size=self.num_comp)
        self.pi = self.pi / sum(self.pi)

    def random_init(self):
        self._random_init()
        self._update()

    def pi_from_array(self, p):
        pis = np.exp(p)
        self.pi = pis / sum(pis)

    def dpi_dx(self):
        pit = np.repeat(np.array([self.pi.T]), self.num_comp, 0)
        return pit * (-pit.T + np.eye(self.num_comp))

    def transform_pi_grad(self, p):
        return mdot(p, self.dpi_dx())

    def get_m_size(self):
        return self.num_comp * self.num_process * self.num_dim

    def updata_mean(self, j, mj):
        for k in range(self.num_comp):
            self.m[k, j, :] = mj.copy()
        self._update()

    def update_covariance(self, j, Sj):
        """ updates covariance matrix j using Sj """
        raise NotImplementedError

    def num_parameters(self):
        """ return number of free parameters of a model """
        raise NotImplementedError

    def get_s_size(self):
        """ return size of s when flattened """
        raise NotImplementedError

    def S_dim(self):
        """ dimensionality of covariance matrix exposed """
        raise NotImplementedError

    def m_from_array(self, ma):
        """ initializes the mean from ma"""
        raise NotImplementedError

    def get_sjk_size(self):
        """ returns number of elements in each skj """
        raise NotImplementedError

    def s_from_array(self, sa):
        """ initializes the covariance matrix from sa"""
        raise NotImplementedError

    def log_pdf(self, j, k, l):
        """ :return N_j(m_k|m_l, S_l + S_k)"""
        raise NotImplementedError

    def tr_Ainv_mult_S(self, A, k, j):
        """ :return  trace(A S_kj) """
        raise NotImplementedError

    def C_m(self, j, k, l):
        """ :return  C_kl^-1 (m_kj - m_lj) """
        raise NotImplementedError

    def C_m_C(self, j, k, l):
        """ :return  C_kl^-1 (m_kj - m_lj)(m_kj - m_lj)T C_kl^-1"""
        raise NotImplementedError

    def aSa(self, a, j):
        """ :return  a S_j aT"""
        raise NotImplementedError

    def mmTS(self, k, j):
        """ :return  m_kj m_kj^T s_kj  """
        raise NotImplementedError

    def dAinvS_dS(self, L, k, j):
        """ :return  dA^{-1}S dS  """
        raise NotImplementedError

    def dAS_dS(self, L, k, j):
        """ :return  dA^{-1}S dS  """
        raise NotImplementedError


    def Sa(self, a, k, j):
        """ :return  S_kj a  """
        raise NotImplementedError

    def _update(self):
        """ updates internal variables of the class """
        pass
