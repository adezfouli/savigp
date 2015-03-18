import csv
import GPy
from sklearn import preprocessing
from cond_likelihood import multivariate_likelihood
from data_source import DataSource
from gsavigp import GSAVIGP
from gsavigp_single_comp import GSAVIGP_SignleComponenet
import numpy as np
from optimizer import Optimizer
from plot import plot_fit
from savigp import Configuration
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
        mu, var = m._raw_predict(Xtest)
        Experiments.export_results('boston', Xtrain, Ytrain, Xtest, YTest, mu, var)
        Experiments.export_model('boston_model.csv', m)

    @staticmethod
    def gaussian_1D_data():
        gaussian_sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(1000, gaussian_sigma)
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 300)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = GSAVIGP_SignleComponenet(Xtrain, Ytrain, Xtrain.shape[0], multivariate_likelihood(np.array([[gaussian_sigma]])),
                                 np.array([[gaussian_sigma]]), kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp'])
        plot_fit(m)
        show(block=True)

    @staticmethod
    def gaussian_1D_data_diag():
        sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(20, sigma)
        # X = preprocessing.scale(X)
        # Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 20)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = GSAVIGP(Xtrain, Ytrain, Xtrain.shape[0], 1, multivariate_likelihood(np.array([[sigma]])),
                                 np.array([[sigma]]), kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp'])
        plot_fit(m)
        gp = SAVIGP_Prediction.gpy_prediction(X, Y, sigma, kernel[0])
        gp.plot()
        show(block=True)


    @staticmethod
    def export_results(file_name, Xtrain, Ytrain, Xtest, Ytest, mu, var):
        path = '../../results/'

        np.savetxt(path + file_name + '_train' + '.csv', np.hstack((Xtrain, Ytrain))
                   , header=''.join(['X%d,'%(j) for j in range(Xtrain.shape[1])] +
                                    ['Y%d,'%(j) for j in range(Ytrain.shape[1])])
                                        , delimiter=',')

        np.savetxt(path + file_name + '_test' + '.csv', np.hstack((Xtest, Ytest, mu, var))
                   , header=''.join(['X%d,'%(j) for j in range(Xtest.shape[1])] +
                                    ['Y%d,'%(j) for j in range(Ytest.shape[1])] +
                                    ['mu%d,'%(j) for j in range(mu.shape[1])] +
                                    ['var%d,'%(j) for j in range(Ytest.shape[1])])
                                        , delimiter=',')

    @staticmethod
    def export_model(file_name, model):
        model.set_configuration([Configuration.MoG, Configuration.HYPER])
        path = '../../results/'

        with open(path + file_name, 'w') as fp:
            f = csv.writer(fp, delimiter=',')
            f.writerow(['#model', model.__class__])
            params = model.get_params()
            param_names = model.get_param_names()
            for j in range(len(params)):
                f.writerow([param_names[j], params[j]])

    @staticmethod
    def get_train_test(X, Y, n_train):
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        Xn = data[:,:X.shape[1]]
        Yn = data[:,X.shape[1]:]
        return Xn[:n_train], Yn[:n_train], Xn[n_train:], Yn[n_train:]

if __name__ == '__main__':
    # Experiments.gaussian_1D_data()
    # Experiments.boston_data()
    Experiments.gaussian_1D_data_diag()