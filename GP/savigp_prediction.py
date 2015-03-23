import time

import GPy
from matplotlib.pyplot import show
import numpy as np

from data_source import DataSource
from gsavigp_diag import GSAVIGP_Diag
from gsavigp_single_comp import GSAVIGP_SignleComponenet
from optimizer import Optimizer
from savigp import Configuration
from likelihood import MultivariateGaussian, UnivariateGaussian
from plot import plot_fit


class SAVIGP_Prediction:
    def __init__(self):
        pass

    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=GPy.likelihoods.Gaussian(None, vairiance))
        return m

    @staticmethod
    def prediction_normal(model_type, verbose, num_input_samples, num_inducing, max_fun=10000):
        np.random.seed(12000)
        gaussian_sigma = 0.2
        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)

        num_samples = 10000
        if model_type == 'diag':
            s1 = GSAVIGP_Diag(X, Y, num_inducing, 2, UnivariateGaussian(gaussian_sigma),

                            [kernel], num_samples,
                                [
                                    Configuration.MoG,
                                    Configuration.ENTROPY,
                                    Configuration.CROSS,
                                    Configuration.ELL,
                                    # Configuration.HYPER
                                ])
        else:
            # for full gaussian with single component
            s1 = GSAVIGP_SignleComponenet(X, Y, num_inducing,
                                          MultivariateGaussian(np.array([[gaussian_sigma]])),

                                          [kernel], num_samples, [
                                              Configuration.MoG,
                                              Configuration.ENTROPY,
                                              Configuration.CROSS,
                                              Configuration.ELL,
                                              # Configuration.HYPER
                                          ])

        # Optimizer.SGD(s1, 1e-16,  s1._get_params(), 2000, verbose=False, adaptive_alpha=False)
        _, t1, t2 = Optimizer.optimize_model(s1, max_fun=max_fun, verbose=verbose, method=['mog'])
        if verbose:
            print 'parameters:', s1.get_params()
            print 'num_input_samples', num_input_samples
            print 'num_samples', num_samples
            print 'gaussian sigma', gaussian_sigma
            print s1.__class__.__name__
            print 'time per iteration:', t1
            print 'total time:', t2
            plot_fit(s1, plot_raw=True)
            gp = SAVIGP_Prediction.gpy_prediction(X, Y, gaussian_sigma, kernel)
            gp.plot()
            show(block=True)
        return model_type, t1, t2

    @staticmethod
    def performance_test():
        models = ['diag', 'full']
        num_input = 3000
        num_inducting = num_input / 10
        for m in models:
            m, t, tt = SAVIGP_Prediction.prediction_normal(m, False, num_input, num_inducting, 10)
            print 'performance for ', m, ':', t, tt

    @staticmethod
    def prediction_test():
        models = ['full']
        num_input = 100
        num_inducting = num_input
        for m in models:
            m, t, tt = SAVIGP_Prediction.prediction_normal(m, True, num_input, num_inducting, 10000)
            print 'performance for ', m, ':', t, tt


if __name__ == '__main__':
    try:
        # SAVIGP_Prediction.performance_test()
        SAVIGP_Prediction.prediction_test()
    finally:
        pass
