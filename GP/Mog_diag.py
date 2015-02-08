__author__ = 'AT'

import numpy as np

class MoG_Diag:
    def __init__(self, num_comp, num_process, num_dim):
        self.num_comp = num_comp
        self.num_process = num_process
        self.num_dim = num_dim
        self.n = num_comp * num_process * num_dim

        self.m =[]
        self.s = []
        self.pi = []
        # self._random_init()
        self.fixed_init()

        self.pi = np.array([1. / num_comp] * num_comp)

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

    def __str__(self):
        return 'm:' + str(self.m) + '\n' + 's:' + str(self.s) + '\n' + 'pi:' + str(self.pi)
