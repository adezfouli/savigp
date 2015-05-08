from GPy.util import datasets
import pandas
from pandas.util.testing import DataFrame
from ExtRBF import ExtRBF

__author__ = 'AT'

import GPy
import numpy as np


class DataSource:

    def __init__(self):
        pass

    @staticmethod
    def normal_generate_samples(n_samples, var, input_dim=3):
        num_samples = n_samples
        noise = var
        num_in = input_dim
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
        # X = preprocessing.scale(X)
        X.sort(axis=0)
        rbf = ExtRBF(num_in, variance=0.5,
                           lengthscale=np.array(np.random.uniform(low=0.1, high=3.0, size=input_dim)), ARD=True)
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
        # uncomment these lines to read directly from original file
        # data_test = pandas.read_csv('../data/breast-cancer-wisconsin.csv', header=None)
        # # replacing Y values with -1 and 1
        # data_test.loc[data_test[10] == 2, 10] = -1
        # data_test.loc[data_test[10] == 4, 10] = 1
        # data_test = data_test.convert_objects(convert_numeric=True).dropna()
        # X = data_test.ix[:, 1:9]
        # Y = data_test.ix[:, 10]
        # return np.array(X), Y[:, np.newaxis]
        data = []
        for i in range(1, 6):
            train = pandas.read_csv('data/wisconsin_cancer/train_' + str(i) + '.csv', header=None)
            test = pandas.read_csv('data/wisconsin_cancer/test_' + str(i) + '.csv', header=None)
            data.append({
                'train_Y': train.ix[:, 0].values[:, np.newaxis],
                'train_X': train.ix[:, 1:].values,
                'test_Y': test.ix[:, 0].values[:, np.newaxis],
                'test_X': test.ix[:, 1:].values,
                'id': i
            })

        return data


    @staticmethod
    def USPS_data():
        def label_to_num(x):
            return  (x[:, 1] + x[:, 2] * 2)[:, np.newaxis]

        data = []
        for i in range(1, 6):
            train = pandas.read_csv('data/USPS/train_' + str(i) + '.csv', header=None)
            test = pandas.read_csv('data/USPS/test_' + str(i) + '.csv', header=None)
            # data.append({
            #     'train_Y': label_to_num(train.ix[:, 0:2].values),
            #     'train_X': train.ix[:, 3:].values,
            #     'test_Y': label_to_num(test.ix[:, 0:2].values),
            #     'test_X': test.ix[:, 3:].values,
            #     'id': i
            # })
            data.append({
                'train_Y': train.ix[:, 0:2].values,
                'train_X': train.ix[:, 3:].values,
                'test_Y': test.ix[:, 0:2].values,
                'test_X': test.ix[:, 3:].values,
                'id': i
            })

        return data


    @staticmethod
    def mining_data():
        data = []
        train = pandas.read_csv('data/mining/data.csv', header=None)
        data.append({
            'train_Y': train.ix[:, 0].values[:, np.newaxis],
            'train_X': train.ix[:, 1].values[:, np.newaxis],
            'test_Y': train.ix[:, 0].values[:, np.newaxis],
            'test_X': train.ix[:, 1].values[:, np.newaxis],
            'id': 1
        })

        return data


    @staticmethod
    def boston_data():
        data = []
        for i in range(1, 6):
            train = pandas.read_csv('data/boston_housing/train_' + str(i) + '.csv', header=None)
            test = pandas.read_csv('data/boston_housing/test_' + str(i) + '.csv', header=None)
            data.append({
                'train_Y': train.ix[:, 0].values[:, np.newaxis],
                'train_X': train.ix[:, 1:].values,
                'test_Y': test.ix[:, 0].values[:, np.newaxis],
                'test_X': test.ix[:, 1:].values,
                'id': i
            })

        return data


    @staticmethod
    def abalone_data():
        data = []
        for i in range(5, 11):
            train = pandas.read_csv('data/abalone/train_' + str(i) + '.csv', header=None)
            test = pandas.read_csv('data/abalone/test_' + str(i) + '.csv', header=None)
            data.append({
                'train_Y': train.ix[:, 0].values[:, np.newaxis],
                'train_X': train.ix[:, 1:].values,
                'test_Y': test.ix[:, 0].values[:, np.newaxis],
                'test_X': test.ix[:, 1:].values,
                'id': i
            })

        return data


    @staticmethod
    def creep_data():
        data = []
        for i in range(1, 6):
            train = pandas.read_csv('data/creep/train_' + str(i) + '.csv', header=None)
            test = pandas.read_csv('data/creep/test_' + str(i) + '.csv', header=None)
            data.append({
                'train_Y': train.ix[:, 0].values[:, np.newaxis],
                'train_X': train.ix[:, 1:].values,
                'test_Y': test.ix[:, 0].values[:, np.newaxis],
                'test_X': test.ix[:, 1:].values,
                'id': i
            })

        return data

if __name__ == '__main__':
    X, Y = DataSource.wisconsin_breast_cancer_data()
    pass
