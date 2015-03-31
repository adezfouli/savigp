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
from gsavigp_diag import GSAVIGP_Diag
from gsavigp_single_comp import GSAVIGP_SignleComponenet
from optimizer import *
from savigp import Configuration
from likelihood import MultivariateGaussian, UnivariateGaussian
from grad_checker import GradChecker
from plot import plot_fit
from util import chol_grad, jitchol, bcolors


class SAVIGP_Test:
    def __init__(self):
        pass

    @staticmethod
    def get_cond_ll(likelihood):
        cov = None
        num_process = -1
        ll = None
        gaussian_sigma = None
        if likelihood == 'multi_Gaussian':
            gaussian_sigma = 0.5
            num_process = 3
            cov = np.eye(num_process) * gaussian_sigma
            ll = MultivariateGaussian(np.array(cov))
        if likelihood == 'univariate_Gaussian':
            gaussian_sigma = 0.5
            num_process = 1
            cov = np.eye(num_process) * gaussian_sigma
            ll = UnivariateGaussian(gaussian_sigma)
        return cov, gaussian_sigma, ll, num_process

    @staticmethod
    def test_grad_diag(config, verbose, sparse, likelihood_type):
        num_input_samples = 3
        num_samples = 100000
        cov, gaussian_sigma, ll, num_process = SAVIGP_Test.get_cond_ll(likelihood_type)
        np.random.seed(1212)
        if sparse:
            num_inducing = num_input_samples - 1
        else:
            num_inducing = num_input_samples
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP_Diag(X, Y, num_inducing, 3, ll,
                     [deepcopy(kernel) for j in range(num_process)], num_samples, config)

        s1.rand_init_mog()

        def f(x):
            s1.set_params(x)
            return s1.objective_function()

        def f_grad(x):
            s1.set_params(x)
            return s1.objective_function_gradients()

        return GradChecker.check(f, f_grad, s1.get_params(), s1.get_param_names(), verbose=verbose)


    @staticmethod
    def test_grad_single(config, verbose, sparse, likelihood_type):
        num_input_samples = 3
        num_samples = 100000
        cov, gaussian_sigma, ll, num_process = SAVIGP_Test.get_cond_ll(likelihood_type)
        np.random.seed(111)
        if sparse:
            num_inducing = num_input_samples - 1
        else:
            num_inducing = num_input_samples
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP_SignleComponenet(X, Y, num_inducing, ll,
                                      [deepcopy(kernel) for j in range(num_process)], num_samples, config)

        s1.rand_init_mog()

        def f(x):
            s1.set_params(x)
            return s1.objective_function()

        def f_grad(x):
            s1.set_params(x)
            return s1.objective_function_gradients()

        return GradChecker.check(f, f_grad, s1.get_params(), s1.get_param_names(), verbose=verbose)

    @staticmethod
    def report_output(config, error, model):
        if error < 0.1:
            print bcolors.OKBLUE, 'passed:', model, config, ' error: ', error
        else:
            print bcolors.WARNING, 'failed', model, config, ' error: ', error
        print bcolors.ENDC

    @staticmethod
    def test_grad(verbose=False):
        configs = [
            [
                Configuration.MoG,
                Configuration.ENTROPY,
                Configuration.CROSS,
                Configuration.ELL,
            ],
            [
                Configuration.MoG,
                Configuration.ENTROPY,
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
                Configuration.ENTROPY,
            ],
            [
                Configuration.HYPER,
                Configuration.CROSS,
            ],
            [
                Configuration.HYPER,
                Configuration.ELL,
            ],
            [
                Configuration.LL,
                Configuration.ELL,
            ]
        ]

        sparse = [True, False]
        models = ['diag', 'full']
        ll = ['univariate_Gaussian', 'multi_Gaussian']

        for m in models:
            for s in sparse:
                for l in ll:
                    for c in configs:
                        # for multi_Gaussian gradients of ll are not implemented
                        if not ('multi_Gaussian' == l and Configuration.LL in c):
                            e1 = None
                            if m == 'diag':
                                e1 = SAVIGP_Test.test_grad_diag(c, False, s, l)
                            if m == 'full':
                                e1 = SAVIGP_Test.test_grad_single(c, False, s, l)
                            SAVIGP_Test.report_output(c, e1, 'model: ' + m + ', ' + ' sparse:' + str(s) + ', ' + ', '
                                                      + 'likelihood: ' + l)


    @staticmethod
    def init_test():
        np.random.seed(12000)
        num_input_samples = 3
        num_samples = 100
        gaussian_sigma = 0.2

        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)

        s1 = GSAVIGP_Diag(X, Y, num_input_samples, 2, MultivariateGaussian(np.array([[gaussian_sigma]])),

                     [kernel], num_samples, [
                Configuration.MoG,
                Configuration.ENTROPY,
                Configuration.CROSS,
                Configuration.ELL,
                Configuration.HYPER
            ])
        Optimizer.BFGS(s1, max_fun=3)

        s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, MultivariateGaussian(np.array([[gaussian_sigma]])),
                                      [kernel], num_samples, [
                Configuration.MoG,
                Configuration.ENTROPY,
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
    def test_gp(plot=False):
        np.random.seed(111)
        num_input_samples = 10
        num_samples = 10000
        gaussian_sigma = 1
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)
        kernel = [GPy.kern.RBF(1, variance=.2, lengthscale=np.array((.2,)))]
        m = GSAVIGP_SignleComponenet(X, Y, num_input_samples, UnivariateGaussian(np.array(gaussian_sigma)),
                                      kernel, num_samples, None)
        try:
            Optimizer.optimize_model(m, 10000, False, ['mog'])
        except KeyboardInterrupt:
            pass
        sa_mean, sa_var = m._raw_predict(X)
        gp = SAVIGP_Test.gpy_prediction(X, Y, gaussian_sigma, kernel[0])
        gp_mean, gp_var = gp.predict(X)
        mean_error = (np.abs(sa_mean - gp_mean)).sum() / sa_mean.shape[0]
        var_error = (np.abs(sa_var - gp_var)).sum() / gp_var.T.shape[0]
        if mean_error < 0.1:
            print bcolors.OKBLUE, "passed: mean gp prediction ", mean_error
        else:
            print bcolors.WARNING, "failed: mean gp prediction ", mean_error
        print bcolors.ENDC
        if var_error < 0.1:
            print bcolors.OKBLUE, "passed: var gp prediction ", var_error
        else:
            print bcolors.WARNING, "failed: var gp prediction ", var_error
        print bcolors.ENDC
        if plot:
            plot_fit(m)
            gp.plot()
            show(block=True)


if __name__ == '__main__':
    SAVIGP_Test.test_gp(True)
    # SAVIGP_Test.init_test()
    # SAVIGP_Test.test_grad()
