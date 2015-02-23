__author__ = 'AT'

from MoG_Diag import MoG_Diag
import numpy as np


class MoG_Diag_Squared(object, MoG_Diag):

    def __init__(self, num_comp, num_process, num_dim):
        MoG_Diag.__init__(self, num_comp, num_process, num_dim)

    def get_parameters(self):
        return np.hstack([self.m.flatten(), np.exp((self.s.flatten())/2), self.pi])

    def transform_S_grad(self, g):
        return 2 * g.flatten() * np.exp(-(self.s.flatten()) / 2)

    def s_from_array(self, sa):
        self.s = np.log(np.reshape(sa, (self.num_comp, self.num_process, self.num_dim)) ** 2)

