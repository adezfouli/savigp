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
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 300)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        gaussian_sigma = 0.2
        m = GSAVIGP_SignleComponenet(Xtrain, Ytrain, Xtrain.shape[0], multivariate_likelihood(np.array([[gaussian_sigma]])),
                                 np.array([[gaussian_sigma]]), kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp'])
        mu, var = m._predict(Xtest)

    @staticmethod
    def get_train_test(X, Y, n_train):
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        Xn = data[:,:X.shape[1]]
        Yn = data[:,X.shape[1]:]
        return Xn[:n_train], Yn[:n_train], Xn[n_train:], Yn[n_train:]

if __name__ == '__main__':
    Experiments.boston_data()