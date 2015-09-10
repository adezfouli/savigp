__author__ = 'AT'

import logging
from data_source import DataSource
from data_transformation import IdentityTransformation
from model_learn import ModelLearn
from plot_results import PlotOutput
from savigp_diag import SAVIGP_Diag
from savigp_single_comp import SAVIGP_SingleComponent
from copy import deepcopy
import GPy
from matplotlib.pyplot import show
from optimizer import *
from savigp import Configuration
from likelihood import UnivariateGaussian, MultivariateGaussian
from grad_checker import GradChecker
from plot import plot_fit
from util import bcolors


class SAVIGP_Test:
    r"""
    Testing SAVIGP models.
    """

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
            num_process = 2
            cov = np.diag(np.random.uniform(1, 5, num_process))
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
        num_samples = 200000
        cov, gaussian_sigma, ll, num_process = SAVIGP_Test.get_cond_ll(likelihood_type)
        np.random.seed(1212)
        if sparse:
            num_inducing = num_input_samples - 1
        else:
            num_inducing = num_input_samples
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, cov)
        s1 = SAVIGP_Diag(X, Y, num_inducing, 3, ll,
                     [deepcopy(kernel) for j in range(num_process)], num_samples, config, 0, True, True)

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
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, cov)
        s1 = SAVIGP_SingleComponent(X, Y, num_inducing, ll,
                                      [deepcopy(kernel) for j in range(num_process)], num_samples, config, 0, True, True)

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
        """
        Test gradient of the model
        """
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
                Configuration.HYPER,
                Configuration.ELL,
                Configuration.ENTROPY,
                Configuration.CROSS
            ],

            [
                Configuration.LL,
                Configuration.ELL,
            ]
        ]

        sparse = [False, True]
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
                                e1 = SAVIGP_Test.test_grad_diag(c, True, s, l)
                            if m == 'full':
                                e1 = SAVIGP_Test.test_grad_single(c, True, s, l)
                            SAVIGP_Test.report_output(c, e1, 'model: ' + m + ', ' + ' sparse:' + str(s) + ', ' + ', '
                                                      + 'likelihood: ' + l)


    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=GPy.likelihoods.Gaussian(None, vairiance))
        return m

    @staticmethod
    def test_model_learn(config):
        """
        Compares the model output with exact GP
        """
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        names = []
        num_input_samples = 20
        gaussian_sigma = .2

        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)
        train_n = int(0.5 * num_input_samples)

        Xtrain = X[:train_n, :]
        Ytrain = Y[:train_n, :]
        Xtest = X[train_n:, :]
        Ytest = Y[train_n:, :]
        kernel1 = ModelLearn.get_kernels(Xtrain.shape[1], 1, True)
        kernel2 = ModelLearn.get_kernels(Xtrain.shape[1], 1, True)
        gaussian_sigma = 1.0

        #number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 10000
        cond_ll = UnivariateGaussian(np.array(gaussian_sigma))

        n1, _ = ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel1, method,
                                           'test_' + ModelLearn.get_ID(), 'test', num_inducing,
                                     num_samples, sparsify_factor, ['mog', 'll', 'hyp'], IdentityTransformation, True,
                                           logging.DEBUG, True)

        n2, _ =ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel2, 'gp',
                                           'test_' + ModelLearn.get_ID(), 'test', num_inducing,
                                     num_samples, sparsify_factor, ['mog', 'll', 'hyp'], IdentityTransformation)

        PlotOutput.plot_output('test', ModelLearn.get_output_path(), [n1, n2], None, False)


    @staticmethod
    def test_gp(plot=False, method='full'):
        """
        Compares model prediction with an exact GP (without optimisation)
        """
        # note that this test fails without latent noise in the case of full Gaussian
        np.random.seed(111)
        num_input_samples = 10
        num_samples = 10000
        gaussian_sigma = .2
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma, 1)
        kernel = [GPy.kern.RBF(1, variance=1., lengthscale=np.array((1.,)))]

        if method == 'full':
            m = SAVIGP_SingleComponent(X, Y, num_input_samples, UnivariateGaussian(np.array(gaussian_sigma)),
                                          kernel, num_samples, None, 0.001, True, True)

        if method == 'diag':
            m = SAVIGP_Diag(X, Y, num_input_samples, 1, UnivariateGaussian(np.array(gaussian_sigma)),
                                          kernel, num_samples, None, 0.001, True, True)

        # update model using optimal parameters
        # gp = SAVIGP_Test.gpy_prediction(X, Y, gaussian_sigma, kernel[0])
        # gp_mean, gp_var = gp.predict(X, full_cov=True)
        # m.MoG.m[0,0] = gp_mean[:,0]
        # m.MoG.update_covariance(0, gp_var - gaussian_sigma * np.eye(10))

        try:
            folder_name = 'test' + '_' + ModelLearn.get_ID()
            logger = ModelLearn.get_logger(folder_name, logging.DEBUG)

            Optimizer.optimize_model(m, 10000, logger, ['mog'])
        except KeyboardInterrupt:
            pass
        sa_mean, sa_var = m.predict(X)
        gp = SAVIGP_Test.gpy_prediction(X, Y, gaussian_sigma, deepcopy(kernel[0]))
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
    SAVIGP_Test.test_grad()
    # SAVIGP_Test.test_gp(True, method='full')
    # SAVIGP_Test.test_model_learn({'method': 'full', 'sparse_factor': 1.0})
