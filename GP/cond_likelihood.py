import math
from numpy.core.umath_tests import inner1d
from numpy.linalg import inv, det
import scipy
import numpy as np
from GPy.util.linalg import mdot

__author__ = 'AT'


def multivariate_likelihood(sigma):
    sigma_inv = inv(sigma)
    const = -1.0 / 2 * np.log(det(sigma)) - float(len(sigma)) / 2 * np.log(2 * math.pi)

    def ll(f, y):
        return const + -1.0 / 2 * inner1d(mdot((f - y), sigma_inv), (f-y))
    return ll
