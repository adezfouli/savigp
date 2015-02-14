from GPy.inference.optimization.conjugate_gradient_descent import CGD
from scipy.optimize import check_grad, minimize, lbfgsb, fmin_l_bfgs_b, basinhopping
from BFGS import Newton_BFGS
from GSAVIGP import GSAVIGP
from Optimizer import Optimizer
from cond_likelihood import multivariate_likelihood
import numpy as np
import GPy

from matplotlib.pyplot import show
from grad_checker import GradChecker
from plot import plot_fit


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
        num_input_samples = 5
        input_dim = 4
        num_inducing = 5
        num_MoG_comp = 2
        num_latent_proc = 1
        # number of samples
        num_samples = 20000
        gaussian_sigma = np.diag(np.ones(num_latent_proc))
        X, Y, kernel, noise = SAVIGP_test.generate_samples(num_input_samples, input_dim, num_latent_proc)
        s1 = GSAVIGP(X, Y, num_inducing, num_MoG_comp, num_latent_proc, multivariate_likelihood(gaussian_sigma), gaussian_sigma,
                    kernel, num_samples, False)

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
        return X, y, kernel

    @staticmethod
    def test_gp():
        num_input_samples = 1000
        num_samples = 10000
        gaussian_sigma = 0.2
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp_mean, gp_var, _025pm, _975pm = gp.predict(X)
        s1 = GSAVIGP(X, Y, num_input_samples, 1, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    kernel, num_samples, False)
        Optimizer.optimize(s1, 0.001,  s1._get_params(), 20)
        s1._predict(X)
        print 'asvig:', s1.MoG
        print 'gp_mean:' , gp_mean
        print 'gp_var:' , gp_var


    @staticmethod
    def prediction():
        num_input_samples = 1000
        num_samples = 10000
        gaussian_sigma = 0.02
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        s1 = GSAVIGP(X, Y, num_input_samples / 10, 1, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    kernel, num_samples, False)

        Optimizer.loopy_opt(s1)
        plot_fit(s1, plot_raw= True)

        # Optimizer.SGD(s1, 1e-3,  s1._get_params(), 100, 1e-6, 1e-6)
        # gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        # gp.plot()
        show(block=True)

if __name__ == '__main__':
    SAVIGP_test.prediction()
    # SAVIGP_test.test_grad()
    # X = np.random.rand(3, 1)
    # print X
    #
    # grad = GradientChecker(np.sin,np.cos,X,'x')
    # grad.checkgrad(verbose=1)