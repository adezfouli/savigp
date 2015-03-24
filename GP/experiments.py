__author__ = 'AT'

import csv
import GPy
from sklearn import preprocessing
from likelihood import MultivariateGaussian, UnivariateGaussian
from data_source import DataSource
from gsavigp_diag import GSAVIGP_Diag
from gsavigp_single_comp import GSAVIGP_SignleComponenet
import numpy as np
from optimizer import Optimizer
from plot import plot_fit
from savigp_prediction import SAVIGP_Prediction
from matplotlib.pyplot import show
from util import id_generator, check_dir_exists




class Experiments:

    @staticmethod
    def export_result_model(model, Xtest, Xtrain, Ytest, Ytrain, Y_pred, var_pred, name):
        path = '../../results/botson_' + id_generator(size=6) +'/'
        # path = '../../results/botson_' + '/'
        check_dir_exists(path)
        file_name = 'boston_' + name
        np.savetxt(path + file_name + '_train' + '.csv', np.hstack((Ytrain, Xtrain))
                   , header=''.join(
                                    ['Y%d,'%(j) for j in range(Ytrain.shape[1])]+
                                    ['X%d,'%(j) for j in range(Xtrain.shape[1])]
                                    )
                    , delimiter=',', comments='')

        np.savetxt(path + file_name + '_test' + '.csv', np.hstack((Ytest, Y_pred, var_pred, Xtest))
                   , header=''.join(
                                    ['Y%d,'%(j) for j in range(Ytest.shape[1])] +
                                    ['mu%d,'%(j) for j in range(Y_pred.shape[1])] +
                                    ['var%d,'%(j) for j in range(Ytest.shape[1])] +
                                    ['X%d,'%(j) for j in range(Xtest.shape[1])])
                    , delimiter=',', comments='')
        if model is not None:
            with open(path + file_name +'_model.csv', 'w') as fp:
                f = csv.writer(fp, delimiter=',')
                f.writerow(['#model', model.__class__])
                params = model.get_all_params()
                param_names = model.get_all_param_names()
                for j in range(len(params)):
                    f.writerow([param_names[j], params[j]])

    @staticmethod
    def boston_data():
        X, Y = DataSource.boston_data()
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)

        Xtrain, Ytrain, Xtest, Ytest = Experiments.get_train_test(X, Y, 300)
        kernel = [GPy.kern.RBF(1, variance=1, lengthscale=np.array((1.,)))]
        gaussian_sigma = 1.0
        m = GSAVIGP_SignleComponenet(Xtrain, Ytrain, Xtrain.shape[0], UnivariateGaussian(np.array(gaussian_sigma)),
                             kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog'])
        y_pred, var_pred = m._raw_predict(Xtest)
        Experiments.export_result_model(m, Xtest, Xtrain, Ytest, Ytrain, y_pred, var_pred, 'savigp')

        # exporing exact gp predictions
        m = GPy.models.GPRegression(Xtrain, Ytrain)
        m.optimize('bfgs')
        y_pred, var_pred = m.predict(Xtest)
        Experiments.export_result_model(None, Xtest, Xtrain, Ytest, Ytrain, y_pred, var_pred, 'gp')

    @staticmethod
    def gaussian_1D_data():
        gaussian_sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(1000, gaussian_sigma)
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 300)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = GSAVIGP_SignleComponenet(Xtrain, Ytrain, Xtrain.shape[0], MultivariateGaussian(np.array([[gaussian_sigma]])),
                                 kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp'])
        plot_fit(m)
        show(block=True)

    @staticmethod
    def gaussian_1D_data_diag():
        sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(20, sigma)
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 20)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = GSAVIGP_Diag(Xtrain, Ytrain, Xtrain.shape[0], 1, MultivariateGaussian(np.array([[sigma]])),
                                 kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp', 'll'])
        plot_fit(m)
        gp = SAVIGP_Prediction.gpy_prediction(X, Y, sigma, kernel[0])
        gp.plot()
        show(block=True)


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