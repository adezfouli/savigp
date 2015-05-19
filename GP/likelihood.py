from GPy.util.linalg import mdot
from numpy.ma import argsort, sort
from scipy.linalg import inv, det
from scipy.misc import logsumexp
from scipy.special import erfinv
from scipy.special._ufuncs import gammaln

from util import cross_ent_normal, drange


__author__ = 'AT'

import math
from numpy.core.umath_tests import inner1d
import numpy as np


class Likelihood:
    def __init__(self):
        pass

    def ll(self, f, y):
        raise Exception("not implemented yet")

    def ll_grad(self, f, y):
        raise Exception("not implemented yet")

    def ll_F_Y(self, F, Y):
        raise Exception("not implemented yet")

    def ll_grad_F_Y(self, F, Y):
        raise Exception("not implemented yet")

    def get_num_params(self):
        raise Exception("not implemented yet")

    def set_params(self, p):
        raise Exception("not implemented yet")

    def get_params(self):
        raise Exception("not implemented yet")

    def map_Y_to_f(self, Y):
        """
            Used of init of posterior mean.
            By default return mean of the Ys
        """
        return np.mean(Y, axis=0)


    # def predict(self, mu, sigma):
    #     self.dim = mu.shape[0]
    #     self.n_samples = 100000
    #     self.normal_samples = np.random.normal(0, 1, self.n_samples * self.dim) \
    #         .reshape((self.n_samples, self.dim))
    #     F = self.normal_samples * np.sqrt(sigma) + mu
    #     Y = self._get_y_range()
    #     mean  = (np.exp((self.ll_F_Y(F[:, np.newaxis, :], Y)[0])).mean(0)*Y.T).sum(1)
    #     return mean, None
    #
    # def _get_y_range(self):
    #     raise Exception("not implemented yet")


    def ell(self, mu, sigma, Y):
        raise Exception("not implemented yet")

class MultivariateGaussian(Likelihood):
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.sigma = sigma
        self.sigma_inv = inv(self.sigma)
        self.const = -1.0 / 2 * np.log(det(self.sigma)) - float(len(self.sigma)) / 2 * np.log(2 * math.pi)

    def ll(self, f, y):
        return self.const + -1.0 / 2 * inner1d(mdot((f - y), self.sigma_inv), (f-y))

    def ll_F_Y(self, F, Y):
        c = 1.0 / 2 * (mdot((F-Y), self.sigma_inv) * (F-Y)).sum(axis=2)
        return (self.const + -c), None

    def ll_grad(self, f, y):
        raise Exception("gradients not supported for multivariate Gaussian")

    def get_sigma(self):
        return self.sigma

    def get_params(self):
        return self.sigma.flatten()

    def get_num_params(self):
        return self.sigma.flatten().shape[0]

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, self.sigma)


class UnivariateGaussian(Likelihood):
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.set_params(np.log([sigma]))

    def ll(self, f, y):
        return self.const + -1.0 / 2 * inner1d(f - y, f - y) / self.sigma

    def ll_F_Y(self, F, Y):
        c = 1.0 / 2 * np.square(F - Y) / self.sigma
        return (self.const + -c)[:, :, 0], (self.const_grad * self.sigma + c)[:, :, 0]

    def ll_grad(self, f, y):
        return self.const_grad * self.sigma + 1.0 / 2 * inner1d(f - y, f - y) / self.sigma

    def ll_grad_F_Y(self, F, Y):
        return (self.const_grad * self.sigma + 1.0 / 2 * np.square(F - Y) / self.sigma)[:, :, 0]

    def set_params(self, p):
        self.sigma = math.exp(p[0])
        self.const = -1.0 / 2 * np.log(self.sigma) - 1.0 / 2 * np.log(2 * math.pi)
        self.const_grad = -1.0 / 2 / self.sigma

    def get_sigma(self):
        return np.array([[self.sigma]])

    def get_params(self):
        return np.array(np.log([self.sigma]))

    def get_num_params(self):
        return 1

    def predict(self, mu, sigma, Ys, model=None):
        var = sigma + self.sigma
        lpd = None
        if not (Ys is None):
            lpd = -(np.square(0.5 * (Ys - mu)) / var + np.log(2. * math.pi * var))[:, 0]
        return mu, var, lpd

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, np.array([[self.sigma]]))


class LogGaussianCox(Likelihood):
    """
    Log Gaussian Cox process

    p(y|f) = (lambda)^y exp(-lambda) / y!

    lambda = f + offset
    """

    def __init__(self, offset):
        Likelihood.__init__(self)
        self.offset = offset

    def ll(self, f, y):
        _log_lambda = (f + self.offset)
        return (y * _log_lambda - np.exp(_log_lambda) - gammaln(y + 1))

    def ll_grad(self, f, y):
        return y - np.exp(f + self.offset)

    def ll_F_Y(self, F, Y):
        _log_lambda = (F + self.offset)
        return (Y * _log_lambda - np.exp(_log_lambda) - gammaln(Y + 1))[:, :, 0], (Y - np.exp(F + self.offset))[:, :, 0]

    def ll_grad_F_Y(self, F, Y):
        return (Y - np.exp(F + self.offset))[:, :, 0]

    def set_params(self, p):
        self.offset = p[0]

    def get_params(self):
        return np.array([self.offset])

    def get_num_params(self):
        return 1

    def predict(self, mu, sigma, Ys, model=None):
        meanval = np.exp(mu + sigma / 2) * np.exp(self.offset)
        varval = (np.exp(sigma) - 1) * np.exp(2 * mu + sigma) * np.exp(2 * self.offset)
        return meanval, varval, None

class LogisticLL(object, Likelihood):
    """
    Logistic likelihood

    p(y|f) = 1 / (1 + exp(-f))

    lambda = f + offset
    """

    def __init__(self):
        Likelihood.__init__(self)
        self.n_samples = 20000
        self.normal_samples = np.random.normal(0, 1, self.n_samples).reshape((1, self.n_samples))

    def ll(self, f, y):
        if y == 1:
            return (-(f + np.abs(f)) / 2 - np.log(1 + np.exp(-np.abs(f))))[:, 0]
        if y == -1:
            return (-(-f + np.abs(-f)) / 2 - np.log(1 + np.exp(-np.abs(-f))))[:, 0]

    def ll_F_Y(self, F, Y):
        return -np.log(1 + np.exp(F * Y))[:, :, 0], None

    def set_params(self, p):
        if p.shape[0] != 0:
            raise Exception("Logistic function does not have free parameters")

    def predict(self, mu, sigma, Ys, model=None):
        f = self.normal_samples * np.sqrt(sigma) + mu
        mean = np.exp(self.ll_F_Y(f.T[:, :, np.newaxis], np.array([[1]]))[0]).mean(axis=0)[:, np.newaxis]
        lpd = None
        if not (Ys is None):
            lpd = np.log((-Ys + 1) / 2 + Ys * mean)

        return mean, mean * (1 - mean), lpd[:, 0]

    def _get_y_range(self):
        return np.array([[1, -1]]).T

    def get_params(self):
        return np.array([])

    def get_num_params(self):
        return 0

class SoftmaxLL(Likelihood):
    """
    Softmax likelihood:

    p(y=c|f) = exp(f_c) / (f_1 + ... + f_N)
    """

    def __init__(self, dim):
        Likelihood.__init__(self)
        self.dim = dim
        self.n_samples = 20000
        self.normal_samples = np.random.normal(0, 1, self.n_samples * dim) \
            .reshape((self.dim, self.n_samples))

    def ll(self, f, y):
        u = f.copy()
        k = f[:, y[0]].copy()
        for j in range(u.shape[1]):
            u[:, j] -= k

        return -logsumexp(u, 1)

    def ll_F_Y(self, F, Y):
        return -logsumexp(F - (F * Y).sum(2)[:, :, np.newaxis], 2), None

    def predict(self, mu, sigma, Ys, model=None):
        F = np.empty((self.n_samples, mu.shape[0], self.dim))
        for j in range(self.dim):
            F[:, :, j] = np.outer(self.normal_samples[j, :], np.sqrt(sigma[:, j])) + mu[:, j]
        expF = np.exp(F)
        mean = (expF / expF.sum(2)[:, :, np.newaxis]).mean(axis=0)
        lpd = None
        if not (Ys is None):
            lpd = np.log((Ys * mean).sum(axis=1))
        return mean, None, lpd


    def ll_grad(self, f, y):
        raise Exception("gradients not supported for multivariate Gaussian")

    def set_params(self, p):
        if p.shape[0] != 0:
            raise Exception("Softmax function does not have free parameters")

    def get_params(self):
        return np.array([])

    def get_num_params(self):
        return 0


class WarpLL(object, Likelihood):
    def __init__(self, ea, eb, c, log_s):
        Likelihood.__init__(self)
        self.set_params(np.hstack((ea, eb, c, [log_s])))

    def warp(self, Y):
        ea = np.exp(self.params[0, :])
        eb = np.exp(self.params[1, :])
        c = self.params[2, :]
        tanhcb = np.tanh(np.add.outer(Y, c) * eb)
        t = (tanhcb * ea).sum(axis=2) + Y
        w = ((1. - np.square(tanhcb)) * ea * eb).sum(axis=2) + 1
        return t, w

    def warpinv(self, z, t0, N):
        for n in range(N):
            t1, dt1 = self.warp(t0)
            t0 -= (t1 - z) / dt1
        return t0

    def ll_F_Y(self, F, Y):
        t, w = self.warp(Y)
        sq = 1.0 / 2 * np.square(F - t) / self.sigma
        return (self.const + -sq + np.log(w))[:, :, 0], \
               (self.const_grad * self.sigma + sq)[:, :, 0]

    def set_params(self, p):
        self.sigma = np.exp(p[-1])
        self.const = -1.0 / 2 * np.log(self.sigma) - 1.0 / 2 * np.log(2 * math.pi)
        self.const_grad = -1.0 / 2 / self.sigma
        if p.shape[0] > 1:
            n = (p.shape[0] - 1) / 3
            self.params = p[:-1].reshape(3, n)

    def predict(self, mu, sigma, Ys, model=None):
        #calculating var
        s = sigma + self.sigma
        alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8000, 0.9000])
        q = np.outer(np.sqrt(2 * s), erfinv(2 * alpha - 1)) + mu

        z = self.warp(model.Y)[0]
        I = argsort(z, axis=0)
        sortz = sort(z, axis=0)
        sortt = model.Y[I]

        quant = self.warpinv(q, self._get_initial_points(q, sortz, sortt), 100)
        var = np.square((quant[:, 8] - (quant[:, 0])) / 4)

        #calculating mu
        H = np.array([7.6e-07, 0.0013436, 0.0338744, 0.2401386, 0.6108626, 0.6108626, 0.2401386, 0.0338744, 0.0013436, 7.6e-07])
        quard = np.array([-3.4361591, -2.5327317, -1.7566836, -1.0366108, -0.3429013, 0.3429013, 1.0366108, 1.7566836, 2.5327317, 3.4361591])
        mu_quad = np.outer(np.sqrt(2 * s), quard) + mu
        mean = self.warpinv(mu_quad, self._get_initial_points(mu_quad, sortz, sortt), 100)
        mean = mdot(mean, H[:, np.newaxis]) / np.sqrt(math.pi)
        lpd = None
        if not (Ys is None):
            ts, w = self.warp(Ys)
            lpd = -0.5*np.log(2*math.pi*s) - 0.5 * np.square(ts-mu)/s + np.log(w)
        return mean, var[:, np.newaxis], lpd[:, 0]

    def _get_y_range(self):
        return np.array([xrange(-1000, 2000, 1)]).T / 1000

    def _get_initial_points(self, q, sortz, sortt):
        t0 = np.empty(q.shape)
        for j in range(q.shape[0]):
            for k in range(q.shape[1]):
                if q[j, k] > sortz[-1]:
                    t0[j,k] = sortt[-1]
                elif q[j,k] < sortz[0]:
                    t0[j,k] = sortt[0]
                else:
                    I = np.argmax(sortz > q[j,k])
                    I = np.array([I - 1, I])
                    t0[j,k] = sortt[I].mean()
        return t0


    def test(self):
        mu = np.array([[1.13395340993645e-06, 5.65190424705805e-06, 5.78826209038103e-06, 2.83243484612040e-06, -7.38434570563690e-07]]).T
        sigma = np.array([[ 0.299216202282485, 0.243742513817980, 0.295996476326654, 0.230752860541760, 0.281672812756221
        ]]).T
        Ys = np.array([[-0.200000000000000, -0.150000000000000, -0.100000000000000, -0.150000000000000, -0.250000000000000]]).T
        self.set_params(np.array([-2.0485, 1.7991, 1.5814, 2.7421, 0.9426, 1.7804, 0.1856, 0.7024, -0.7421, -0.0712]))
        self.sigma = 0.8672

    def get_params(self):
        return np.array(np.log([self.sigma]))

    def get_num_params(self):
        return 1
