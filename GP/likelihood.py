__author__ = 'AT'

import math
from numpy.core.umath_tests import inner1d
from numpy.linalg import inv, det
import scipy
import numpy as np
from GPy.util.linalg import mdot


class Likelihood:

    def __init__(self):
        pass

    def get_ll(self):
        raise Exception("not implemented yet")

    def get_ll_grad(self):
        raise Exception("gradients not supported for multivariate Gaussian")

    def get_num_params(self):
        raise Exception("gradients not supported for multivariate Gaussian")


class MultivariateGaussian(Likelihood):
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.sigma = sigma

    def get_ll(self):
        sigma_inv = inv(self.sigma)
        const = -1.0 / 2 * np.log(det(self.sigma)) - float(len(self.sigma)) / 2 * np.log(2 * math.pi)

        def ll(f, y):
            return const + -1.0 / 2 * inner1d(mdot((f - y), sigma_inv), (f-y))
        return ll

    def get_ll_grad(self):
        raise Exception("gradients not supported for multivariate Gaussian")


class UnivariateGaussian(Likelihood):
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.sigma = sigma

    def get_ll(self):
        const = -1.0 / 2 * np.log(self.sigma) - float(len(self.sigma)) / 2 * np.log(2 * math.pi)

        def ll(f, y):
            return const + -1.0 / 2 * inner1d(f-y, f-y) / self.sigma
        return ll

    def get_ll_grad(self):
        def grad(f, y):
            return 1.0 / 2 * inner1d(f-y, f-y) / (self.sigma * self.sigma)
        return grad

    def get_num_params(self):
        return 1