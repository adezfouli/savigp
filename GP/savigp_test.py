from data_source import DataSource

__author__ = 'AT'

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


class SAVIGP_Test:
    def __init__(self):
        pass

    @staticmethod
    def test_grad_diag(config, verbose):
        num_input_samples = 4
        num_samples = 100000
        gaussian_sigma = 0.02
        num_process = 4
        cov = np.eye(num_process) * gaussian_sigma
        np.random.seed(1111)
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP(X, Y, num_input_samples - 1, 3, multivariate_likelihood(np.array(cov)), np.array(cov),
                     [deepcopy(kernel) for j in range(num_process)], num_samples, config)

        s1.rand_init_MoG()

        def f(x):
            s1.set_params(x)
            return s1.objective_function()

        def f_grad(x):
            s1.set_params(x)
            return s1.objective_function_gradients()

        return GradChecker.check(f, f_grad, s1.get_params(), s1.get_param_names(), verbose=verbose)


    @staticmethod
    def test_grad_single(config, verbose):
        num_input_samples = 4
        num_samples = 100000
        gaussian_sigma = 0.02
        num_process = 4
        cov = np.eye(num_process) * gaussian_sigma
        np.random.seed(111)
        X, Y, kernel = SAVIGP_Test.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples - 1, multivariate_likelihood(np.array(cov)),
                                      np.array(cov),
                                      [deepcopy(kernel) for j in range(num_process)], num_samples, config)

        s1.rand_init_MoG()

        def f(x):
            s1.set_params(x)
            return s1.objective_function()

        def f_grad(x):
            s1.set_params(x)
            return s1.objective_function_gradients()

        return GradChecker.check(f, f_grad, s1.get_params(), s1.get_param_names(), verbose=verbose)

    @staticmethod
    def test_grad(verbose = False):
        configs = [
            [
                Configuration.MoG,
                Configuration.ETNROPY,
            ],
            [
                Configuration.MoG,
                Configuration.CROSS,
            ],
            [
                Configuration.MoG,
                Configuration.ELL,
            ],
            [
                Configuration.HYPER,
                Configuration.ETNROPY,
            ],
            [
                Configuration.HYPER,
                Configuration.CROSS,
            ],
            [
                Configuration.HYPER,
                Configuration.ELL,
            ]
        ]
        for c in configs:
            e1 = SAVIGP_Test.test_grad_diag(c, verbose)
            if e1 < 0.1:
                print bcolors.OKBLUE, 'passed: diag', c, e1
            else:
                print bcolors.WARNING, 'failed: diag', c, e1
            print bcolors.ENDC

        for c in configs:
            e1 = SAVIGP_Test.test_grad_single(c, verbose)
            if e1 < 0.1:
                print bcolors.OKBLUE, 'passed: full', c, e1
            else:
                print bcolors.WARNING, 'failed: full', c, e1
            print bcolors.ENDC

    @staticmethod
    def init_test():
        np.random.seed(12000)
        num_input_samples = 3
        num_samples = 100
        gaussian_sigma = 0.2

        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)

        s1 = GSAVIGP(X, Y, num_input_samples, 2, multivariate_likelihood(np.array([[gaussian_sigma]])),
                     np.array([[gaussian_sigma]]),
                     [kernel], num_samples, [
                Configuration.MoG,
                Configuration.ETNROPY,
                Configuration.CROSS,
                Configuration.ELL,
                Configuration.HYPER
            ])
        Optimizer.BFGS(s1, max_fun=3)

        s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, multivariate_likelihood(np.array([[gaussian_sigma]])),
                                      np.array([[gaussian_sigma]]),
                                      [kernel], num_samples, [
                Configuration.MoG,
                Configuration.ETNROPY,
                Configuration.CROSS,
                Configuration.ELL,
                Configuration.HYPER
            ])
        Optimizer.BFGS(s1, max_fun=3)


    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=GPy.likelihoods.Gaussian(None, vairiance))
        return m

    @staticmethod
    def test_gp():
        num_input_samples = 10
        num_samples = 10000
        gaussian_sigma = 0.2
        X, Y, kernel = SAVIGP_Test.normal_generate_samples(num_input_samples, gaussian_sigma)
        gp = SAVIGP_Test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp_mean, gp_var = gp.predict(X)
        s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, multivariate_likelihood(np.array([[gaussian_sigma]])),
                                      np.array([[gaussian_sigma]]),
                                      [kernel], num_samples, [
                Configuration.MoG,
                Configuration.ETNROPY,
                Configuration.CROSS,
                Configuration.ELL,
                # Configuration.HYPER
            ])
        try:
            # Optimizer.SGD(s1, 1e-6, s1._get_params(), 10000,  ftol=1e-3, adaptive_alpha=False)
            Optimizer.BFGS(s1, max_fun=100000)
        except KeyboardInterrupt:
            pass
        sa_mean, sa_var = s1._predict(X)
        print 'asvig mean:', sa_mean
        print 'gp_mean:', gp_mean.T
        print 'asvig var:', sa_var
        print 'gp_var:', gp_var.T



if __name__ == '__main__':
    SAVIGP_Test.init_test()
    SAVIGP_Test.test_grad()
