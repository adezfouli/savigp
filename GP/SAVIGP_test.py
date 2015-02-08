from GSAVIGP import GSAVIGP
from cond_likelihood import multivariate_likelihood
import numpy as np
import GPy

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


def test_grad():
    # number of input data points
    num_input_samples = 10
    input_dim = 1
    num_inducing = 10
    num_MoG_comp = 1
    num_latent_proc = 1
    # number of samples
    num_samples = 1000
    gaussian_sigma = np.diag(np.ones(num_latent_proc))
    X, Y, kernel, noise = generate_samples(num_input_samples, input_dim, num_latent_proc)
    s1 = GSAVIGP(X, Y, num_inducing, num_MoG_comp, num_latent_proc, multivariate_likelihood(gaussian_sigma), gaussian_sigma,
                kernel, num_samples, False)
    s1.checkgrad(verbose=True)



if __name__ == '__main__':
    test_grad()
