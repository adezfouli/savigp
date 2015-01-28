import math
import numpy as np
from numpy.ma import trace
from scipy.linalg import det, inv
from GPy.util.linalg import mdot


def mdiag_dot(A, B):
    return np.einsum('ij,ji -> i', A, B)


def KL_normal(m1, sigma1, m2, sigma2):
    return 1. / 2. * (math.log(det(sigma2) / det(sigma1)) - len(m1) + trace(mdot(inv(sigma2), sigma1)) + \
    mdot((m2 - m1).T, inv(sigma2) , m2- m1))


def cross_ent_normal(m1, sigma1, m2, sigma2):
    return -KL_normal(m1, sigma1, m2, sigma2) - 1. / 2 * math.log(det(2.0 * math.pi * math.e * sigma1))
