from copy import deepcopy, copy
import warnings
from numpy.ma import trace
from scipy.linalg import inv, det
from sklearn import preprocessing
import GPy
from matplotlib.pyplot import show
from GPy.util.linalg import mdot
import numpy as np
from gsavigp import GSAVIGP
from gsavigp_single_comp import GSAVIGP_SignleComponenet
from optimizer import *
from savigp import Configuration
from cond_likelihood import multivariate_likelihood
from grad_checker import GradChecker
from plot import plot_fit
from util import chol_grad, jitchol, bcolors


class SAVIGP_Prediction:
    def __init__(self):
        pass

    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=GPy.likelihoods.Gaussian(None, vairiance))
        return m

    @staticmethod
    def normal_generate_samples(n_samples, var):
        num_samples = n_samples
        noise = var
        num_in = 1
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
        # X = preprocessing.scale(X)
        X.sort(axis=0)
        rbf = GPy.kern.RBF(num_in, variance=0.5, lengthscale=np.array((0.2,)))
        white = GPy.kern.White(num_in, variance=noise)
        kernel = rbf + white
        K = kernel.K(X)
        y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
        return X, y, rbf


    @staticmethod
    def prediction():
        np.random.seed(12000)
        num_input_samples = 100
        num_samples = 10000
        gaussian_sigma = 0.2

        X, Y, kernel = SAVIGP_Prediction.normal_generate_samples(num_input_samples, gaussian_sigma)

        try:
            # for diagonal covariance
            s1 = GSAVIGP(X, Y, num_input_samples, 2, multivariate_likelihood(np.array([[gaussian_sigma]])),
                         np.array([[gaussian_sigma]]),
                         [kernel], num_samples, [
                    Configuration.MoG,
                    Configuration.ETNROPY,
                    Configuration.CROSS,
                    Configuration.ELL,
                    # Configuration.HYPER
                ])

            # for full gaussian with single component
            # s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
            # [kernel], num_samples, [
            # Configuration.MoG,
            # Configuration.ETNROPY,
            #                                         Configuration.CROSS,
            #                                         Configuration.ELL,
            #                                         # Configuration.HYPER
            #     ])

            # Optimizer.SGD(s1, 1e-16,  s1._get_params(), 2000, verbose=False, adaptive_alpha=False)
            Optimizer.BFGS(s1, max_fun=100000)
        except KeyboardInterrupt:
            pass
        print 'parameters:', s1._get_params()
        print 'num_input_samples', num_input_samples
        print 'num_samples', num_samples
        print 'gaussian sigma', gaussian_sigma
        print s1.__class__.__name__
        plot_fit(s1, plot_raw=True)

        gp = SAVIGP_Prediction.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp.plot()
        show(block=True)

    @staticmethod
    def test1():
        dim = 3
        A = np.random.uniform(low=3.0, high=10.0, size=dim * dim).reshape(dim, dim)
        print np.diagonal(A)
        A = mdot(np.tril(A), np.tril(A).T)

        def f(L):
            return trace(mdot(inv(A), np.diag(L)))

        def grad_f(L):
            return np.diagonal(inv(A))

        L_len = (dim) * (dim + 1) / 2
        GradChecker.check(f, grad_f, np.random.uniform(low=1.0, high=3.0, size=dim), ["f"] * L_len)


if __name__ == '__main__':
    try:
        SAVIGP_Prediction.prediction()
    finally:
        pass
