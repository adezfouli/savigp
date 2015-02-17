import cProfile
from DerApproximator import get_d1
from GPy.inference.optimization.conjugate_gradient_descent import CGD
from GPy.util.linalg import mdot
import math
from numpy.ma import trace
from scipy.linalg import inv, det
from scipy.optimize import check_grad, minimize, lbfgsb, fmin_l_bfgs_b, basinhopping
from GSAVIGP import GSAVIGP
from GSAVIGP_full import GSAVIGP_Full
from Optimizer import Optimizer
from cond_likelihood import multivariate_likelihood
import numpy as np
import GPy

from matplotlib.pyplot import show
from grad_checker import GradChecker
from plot import plot_fit
from util import chol_grad, pddet


class SAVIGP_test:
    def __init__(self):
        pass

    @staticmethod
    def generate_samples(num_samples, input_dim, y_dim):
        np.random.seed()
        noise=0.02
        X = np.random.uniform(low=-1000.0, high=1000.0, size=(num_samples, input_dim))
        X.sort(axis=0)
        rbf = GPy.kern.RBF(input_dim, variance=1., lengthscale=np.array((0.25,)))
        white = GPy.kern.White(input_dim, variance=noise)
        kernel = rbf + white
        Y = np.sin([X.sum(axis=1).T]).T + np.random.randn(num_samples, y_dim) * 0.05
        return X, Y, kernel, noise

    @staticmethod
    def test_grad():
        # number of input data points
        num_input_samples = 1
        input_dim = 1
        num_inducing = num_input_samples
        num_MoG_comp = 1
        num_latent_proc = 1
        # number of samples
        num_samples = 10000
        gaussian_sigma = np.diag(np.ones(num_latent_proc))
        X, Y, kernel, noise = SAVIGP_test.generate_samples(num_input_samples, input_dim, num_latent_proc)
        s1 = GSAVIGP_Full(X, Y, num_inducing, num_MoG_comp, multivariate_likelihood(gaussian_sigma), gaussian_sigma,
                    [kernel] * num_latent_proc, num_samples)

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
        np.random.seed()
        num_in = 1
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
        X.sort(axis=0)
        rbf = GPy.kern.RBF(num_in, variance=1., lengthscale=np.array((0.2,)))
        white = GPy.kern.White(num_in, variance=noise)
        kernel = rbf + white
        K = kernel.K(X)
        y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
        return X, y, rbf

    @staticmethod
    def test_gp():
        num_input_samples = 1
        num_samples = 100000
        gaussian_sigma = 2.0
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp_mean, gp_var = gp.predict(X)
        s1 = GSAVIGP_Full(X, Y, num_input_samples, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    [kernel], num_samples)
        Optimizer.SGD(s1, 0.001, s1._get_params(), 10000,  factor=1)
        s1._predict(X)
        print 'asvig mean:', s1.MoG.m[0,0]
        print 'gp_mean:' , gp_mean.T
        if s1.MoG.s[0,0].ndim > 1:
            print 'asvig var:', np.diagonal(s1.MoG.s[0,0])
        else:
            print 'asvig var:', s1.MoG.s[0,0]
        print 'gp_var:' , gp_var.T
        print 'savigp s', s1.MoG.s


    @staticmethod
    def prediction():
        num_input_samples = 1000
        num_samples = 10000
        gaussian_sigma = 0.02
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP(X, Y, num_input_samples / 10, 1,  multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    [kernel], num_samples)

        Optimizer.loopy_opt(s1)
        plot_fit(s1, plot_raw= True)

        # Optimizer.SGD(s1, 1e-3,  s1._get_params(), 100, 1e-6, 1e-6)
        # gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        # gp.plot()
        show(block=True)


    @staticmethod
    def prediction_full_gp():
        num_input_samples = 200
        num_samples = 10000
        gaussian_sigma = 0.02
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP_Full(X, Y, num_input_samples, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    [kernel], num_samples)

        Optimizer.loopy_opt(s1)
        plot_fit(s1, plot_raw= True)

        # Optimizer.SGD(s1, 1e-3,  s1._get_params(), 100, 1e-6, 1e-6)
        # gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        # gp.plot()
        show(block=True)

    @staticmethod
    def test1():
        dim =5

        def f(L):
            s = np.zeros((dim, dim))
            s[np.tril_indices_from(s)] = L
            X = mdot(s, s.T)
            return trace(mdot(X, X.T))

        def grad_f(L):
            s = np.zeros((dim, dim))
            s[np.tril_indices_from(s)] = L
            return chol_grad(s, 2 * (mdot(s, s.T)))[np.tril_indices_from(s)]

        L_len = (dim) * (dim + 1) / 2
        GradChecker.check(f, grad_f, np.random.uniform(low=1.0, high=3.0, size=L_len), ["f"] * L_len)


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    try:
        SAVIGP_test.prediction()
        # SAVIGP_test.test_gp()
        # SAVIGP_test.test_grad()
    finally:
        # print pr.print_stats(sort='cumtime')
        pass
