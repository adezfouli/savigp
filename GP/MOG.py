import numpy as np
class MoG:
    def __init__(self, num_comp, num_process, num_dim):
        self.m = np.zeros((num_comp, num_process, num_dim))
        self.sigma = np.array([np.array([np.eye(num_dim)] * num_process)] * num_comp)
        # self.pi = np.array([1. / num_comp] * (num_comp-1) + [0])
        self.pi = np.array([1. / num_comp] * (num_comp))
