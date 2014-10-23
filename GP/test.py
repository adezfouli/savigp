from matplotlib.pyplot import show
from GPy.examples import tutorials
from GPy.examples.regression import olympic_marathon_men

import numpy as np
import GPy

def test():
    X = np.random.uniform(-3.,3.,(20,1))
    Y = np.sin(X) + np.random.randn(20,1)*0.05

    kernel = GPy.kern.rbf(input_dim=1, variance= 4., lengthscale=3.)

    m = GPy.models.GPRegression(X, Y, kernel)

    print m
    m.plot()
    show(block=True)

test()
