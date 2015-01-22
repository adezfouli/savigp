from matplotlib.pyplot import show
import numpy as np
from BCM import BCM_GP
import GPy
import direct_gp


def generate_samples():
    seed=1000
    num_samples=500
    noise=0.02
    np.random.seed(seed=seed)
    num_in = 1
    X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
    X.sort(axis=0)
    rbf = GPy.kern.rbf(num_in, variance=1., lengthscale=np.array((0.25,)))
    white = GPy.kern.white(num_in, variance=noise)
    kernel = rbf + white
    K = kernel.K(X)
    y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
    return X, y, kernel, noise


def test_direct_gp(optimize=True, plot=True):
    X, y, rbf, noise = generate_samples()
    # D = np.zeros((len(X), 2))
    # D[:, 1] = y[:, 0]
    # D[:, 0] = X[:, 0]
    # print D[:, [0]]
    m = BCM_GP(X, GPy.likelihoods.Gaussian(y, False), rbf, False, noise, 1)
    m.plot(plot_raw=True)
    show(block=True)

def test():
    # num_samples = 10
    # X = np.random.uniform(-3., 3., (num_samples, 2))
    # Y = np.sin(X[:, 0:1]) * np.sin(X[:, 1:2]) + np.random.randn(num_samples, 1) * 0.05

    # x = np.array([np.eye(3)] * 3)
    # print x
    # print Y
    np.random.random()


test()