import GPy
from sklearn import preprocessing
from cond_likelihood import multivariate_likelihood
from data_source import DataSource
from gsavigp_single_comp import GSAVIGP_SignleComponenet
import numpy as np
from optimizer import Optimizer
from plot import plot_fit
from savigp_prediction import SAVIGP_Prediction
from matplotlib.pyplot import show


__author__ = 'AT'


class Experiments:

    @staticmethod
    def boston_data():
        X, Y = DataSource.boston_data()
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        gaussian_sigma = 0.2
        m = GSAVIGP_SignleComponenet(X, Y, X.shape[0], multivariate_likelihood(np.array([[gaussian_sigma]])),
                                 np.array([[gaussian_sigma]]), kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog'])
        plot_fit(m, plot_raw=True)
        gp = SAVIGP_Prediction.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp.plot()
        show(block=True)


if __name__ == '__main__':
    Experiments.boston_data()