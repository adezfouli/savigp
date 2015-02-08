from GSAVIGP import GSAVIGP
from SGD import SGD
from cond_likelihood import multivariate_likelihood
import numpy as np
import GPy
from matplotlib.pyplot import show



class SAVIGP_test:
    def __init__(self):
        pass

    @staticmethod
    def generate_samples(num_samples, input_dim, y_dim):
        np.random.seed()
        noise=0.02
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, input_dim))
        X.sort(axis=0)
        rbf = GPy.kern.rbf(input_dim, variance=1., lengthscale=np.array((0.25,)))
        white = GPy.kern.white(input_dim, variance=noise)
        kernel = rbf + white
        Y = np.sin([X.sum(axis=1).T]).T + np.random.randn(num_samples, y_dim) * 0.05
        return X, Y, kernel, noise


    @staticmethod
    def test_grad():
        # number of input data points
        num_input_samples = 500
        input_dim = 1
        num_inducing = 10
        num_MoG_comp = 1
        num_latent_proc = 1
        # number of samples
        num_samples = 1000
        gaussian_sigma = np.diag(np.ones(num_latent_proc))
        X, Y, kernel, noise = SAVIGP_test.generate_samples(num_input_samples, input_dim, num_latent_proc)
        s1 = GSAVIGP(X, Y, num_inducing, num_MoG_comp, num_latent_proc, multivariate_likelihood(gaussian_sigma), gaussian_sigma,
                    kernel, num_samples, False)
        s1.checkgrad(verbose=True)


    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, likelihood=GPy.likelihoods.Gaussian(Y, vairiance, False), kernel=kernel)
        return m

    @staticmethod
    def normal_generate_samples(n_samples, var):
        seed=1000
        num_samples=n_samples
        noise=var
        np.random.seed(seed=seed)
        num_in = 1
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
        X.sort(axis=0)
        rbf = GPy.kern.rbf(num_in, variance=1., lengthscale=np.array((0.25,)))
        white = GPy.kern.white(num_in, variance=noise)
        kernel = rbf + white
        K = kernel.K(X)
        y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
        return X, y, kernel

    @staticmethod
    def test_gp():
        num_input_samples = 10
        num_samples = 10000
        gaussian_sigma = 0.2
        X, Y, kernel = SAVIGP_test.normal_generate_samples(num_input_samples, gaussian_sigma)
        gp = SAVIGP_test.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp_mean, gp_var, _025pm, _975pm = gp.predict(X)
        s1 = GSAVIGP(X, Y, num_input_samples, 1, 1, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
                    kernel, num_samples, False)
        SGD.optimize(s1, 0.001,  s1._get_params(), 10000)
        print 'asvig:', s1.MoG
        print 'gp_mean:' , gp_mean
        print 'gp_var:' , gp_var

if __name__ == '__main__':
    SAVIGP_test.test_gp()
