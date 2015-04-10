from GPy.util import datasets
import pandas
from pandas.util.testing import DataFrame

__author__ = 'AT'

import GPy
import numpy as np


class DataSource:

    def __init__(self):
        pass

    @staticmethod
    def normal_generate_samples(n_samples, var):
        num_samples = n_samples
        noise = var
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
    def normal_1D_data(n_samples, var):
        num_samples = n_samples
        noise = var
        num_in = 1
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
        X.sort(axis=0)
        rbf = GPy.kern.RBF(num_in, variance=0.5, lengthscale=np.array((0.2,)))
        white = GPy.kern.White(num_in, variance=noise)
        kernel = rbf + white
        K = kernel.K(X)
        y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
        return X, y

    @staticmethod
    def wisconsin_breast_cancer_data():
        data_test = pandas.read_csv('../data/breast-cancer-wisconsin.csv', header=None)
        # replacing Y values with 0 and 1
        data_test.loc[data_test[10] == 2, 10] = 0
        data_test.loc[data_test[10] == 4, 10] = 1
        X = data_test.ix[:, 1:9]
        Y = data_test.ix[:, 10]
        return np.array(X), np.array([Y]).T


    @staticmethod
    def boston_data():
        data = datasets.boston_housing()
        X = data['X'].copy()
        Y = data['Y'].copy()
        return X, Y

if __name__ == '__main__':
    X, Y = DataSource.wisconsin_breast_cancer_data()
    pass
