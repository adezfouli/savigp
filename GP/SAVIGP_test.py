from numpy.ma import trace
from scipy.linalg import inv, det
from sklearn import preprocessing
import GPy
from matplotlib.pyplot import show
from GPy.util.linalg import mdot
import numpy as np
from GSAVIGP import GSAVIGP
from GSAVIGP_full import GSAVIGP_Full
from GSAVIGP_singel_comp import GSAVIGP_SignleComponenet
from Optimizer import *
from SAVIGP import Configuration
from cond_likelihood import multivariate_likelihood
from grad_checker import GradChecker
from plot import plot_fit
from util import chol_grad, jitchol


class SAVIGP_test:
    def __init__(self):
        pass
    @staticmethod
    def test_grad():
        num_input_samples = 4
        num_samples = 10000
        gaussian_sigma = 0.02
        num_process = 4
        cov = np.eye(num_process) * gaussian_sigma
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP(X, Y, num_input_samples, 3, multivariate_likelihood(np.array(cov)), np.array(cov),
                    [kernel] * num_process, num_samples, [
                                                Configuration.MoG,
                                                Configuration.ETNROPY,
                                                Configuration.CROSS,
                                                # Configuration.ELL,
                                                # Configuration.HYPER
            ])

        s1.rand_init_MoG()

        def f(x):
            s1._set_params(x)
            return s1.objective_function()

        def f_grad(x):
            s1._set_params(x)
            return s1.objective_function_gradients()

        GradChecker.check(f, f_grad, s1._get_params(), s1._get_param_names())

    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=GPy.likelihoods.Gaussian(None, vairiance))
        return m

    @staticmethod
    def normal_generate_samples(n_samples, var):
        num_samples=n_samples
        noise=var
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
    def test_gp():
        num_input_samples = 10
        num_samples = 10000
        gaussian_sigma = 0.2
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp_mean, gp_var = gp.predict(X)
        s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
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
        print 'gp_mean:' , gp_mean.T
        print 'asvig var:', sa_var
        print 'gp_var:' , gp_var.T


    @staticmethod
    def prediction():
        np.random.seed(12000)
        num_input_samples = 100
        num_samples = 10000
        gaussian_sigma = 0.2

        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)

        try:
            # for diagonal covariance
            s1 = GSAVIGP(X, Y, num_input_samples, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                        [kernel], num_samples, [
                                                    Configuration.MoG,
                                                    Configuration.ETNROPY,
                                                    Configuration.CROSS,
                                                    Configuration.ELL,
                                                    # Configuration.HYPER
                ])

            # for full gaussian with single component
            # s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
            #             [kernel], num_samples, [
            #                                         Configuration.MoG,
            #                                         Configuration.ETNROPY,
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
        plot_fit(s1, plot_raw= True)

        gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp.plot()
        show(block=True)

    @staticmethod
    def prediction_full_gp():
        num_input_samples = 100
        num_samples = 10000
        gaussian_sigma = 0.02
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP_Full(X, Y, num_input_samples, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    [kernel], num_samples)

        # Optimizer.loopy_opt(s1)
        # Optimizer.SGD(s1, 0.0000001, s1._get_params(), 20000)
        Optimizer.O_BFGS(s1, s1._get_params(), 0.1, 0.1, 1e-7, 100)
        print s1._get_params()
        plot_fit(s1, plot_raw= True)
        # Optimizer.SGD(s1, 1e-3,  s1._get_params(), 100, 1e-6, 1e-6)
        # gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        # gp.plot()
        show(block=True)

    @staticmethod
    def test1():
        dim =3
        A = np.random.uniform(low=3.0, high=10.0, size=dim * dim).reshape(dim,dim)
        print np.diagonal(A)
        A = mdot(np.tril(A), np.tril(A).T)

        def f(L):
            return trace(mdot(inv(A), np.diag(L)))

        def grad_f(L):
            return np.diagonal(inv(A))

        L_len = (dim) * (dim + 1) / 2
        GradChecker.check(f, grad_f, np.random.uniform(low=1.0, high=3.0, size=dim), ["f"] * L_len)

    @staticmethod
    def sparse_GPY():
        """Run a 1D example of a sparse GP regression."""
        # sample inputs and outputs
        num_input_samples = 20
        num_samples = 10000
        gaussian_sigma = 0.02
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        # create simple GP Model
        m = GPy.models.SparseGPRegression(X, Y, kernel=kernel, num_inducing=num_input_samples)

        m.plot()
        show(block=True)

        return m




if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr = line_profiler.LineProfiler()
    # pr.enable()
    try:
        SAVIGP_test.prediction()
        # SAVIGP_test.test_grad()

    finally:
        # print pr.print_stats(sort='cumtime')
        # print pr.print_stats()
        pass
