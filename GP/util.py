import math
import numpy as np
from numpy.ma import trace
from scipy import linalg
from scipy.lib import lapack
from scipy.linalg import det, inv, lapack
from GPy.util.linalg import mdot, dpotri


def mdiag_dot(A, B):
    return np.einsum('ij,ji -> i', A, B)


def KL_normal(m1, sigma1, m2, sigma2):
    return 1. / 2. * (math.log(det(sigma2) / det(sigma1)) - len(m1) + trace(mdot(inv(sigma2), sigma1)) + \
    mdot((m2 - m1).T, inv(sigma2) , m2- m1))


def cross_ent_normal(m1, sigma1, m2, sigma2):
    """
    calculates cross entropy between two Gaussian
    """
    return -KL_normal(m1, sigma1, m2, sigma2) - 1. / 2 * math.log(det(2.0 * math.pi * math.e * sigma1))

def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError, "not pd: non-positive diagonal elements"
        jitter = diagA.mean() * 1e-6
        while maxtries > 0 and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise linalg.LinAlgError, "not positive definite, even with jitter."

def pddet(L):
    """
    Determinant of a positive definite matrix, only symmetric matricies though
    """
    logdetA = 2*sum(np.log(np.diag(L)))
    return logdetA


def inv_chol(L):
    Ai, _ = dpotri(L, lower=1)
    return Ai
