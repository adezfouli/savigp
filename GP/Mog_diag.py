__author__ = 'AT'

import numpy as np

class MoG_Diag:
    def __init__(self, num_comp, num_process, num_dim):
        self.num_comp = num_comp
        self.num_process = num_process
        self.num_dim = num_dim
        self.m = np.random.uniform(low=-1.0, high=1.0, size=(num_comp, num_process, num_dim))
        self.s = np.ones((num_comp, num_process, num_dim))
        self.pi = np.array([1. / num_comp] * num_comp)

    def m_from_array(self, ma):
        self.m = ma.reshape((self.num_comp, self.num_process, self.num_dim))

    def s_from_array(self, sa):
        self.s = sa.reshape((self.num_comp, self.num_process, self.num_dim))
