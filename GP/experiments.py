from plot_results import PlotOutput
from savigp import SAVIGP
from savigp_diag import SAVIGP_Diag
from savigp_single_comp import SAVIGP_SingleComponent

__author__ = 'AT'

import csv
import GPy
from sklearn import preprocessing
from likelihood import MultivariateGaussian, UnivariateGaussian, LogisticLL
from data_source import DataSource
import numpy as np
from optimizer import Optimizer
from plot import plot_fit
from savigp_prediction import SAVIGP_Prediction
from matplotlib.pyplot import show
from util import id_generator, check_dir_exists




class Experiments:

    @staticmethod
    def get_output_path():
        return '../../results/'

    @staticmethod
    def export_train(name, Xtrain, Ytrain):
        path = Experiments.get_output_path() + name +'/'
        check_dir_exists(path)
        file_name = 'train_'
        np.savetxt(path + file_name + '.csv', np.hstack((Ytrain, Xtrain))
                   , header=''.join(
                                    ['Y%d,'%(j) for j in range(Ytrain.shape[1])]+
                                    ['X%d,'%(j) for j in range(Xtrain.shape[1])]
                                    )
                    , delimiter=',', comments='')

    @staticmethod
    def export_model(model, name):
        path = Experiments.get_output_path() + name +'/'
        check_dir_exists(path)
        file_name = 'model_'
        if model is not None:
            with open(path + file_name + '.csv', 'w') as fp:
                f = csv.writer(fp, delimiter=',')
                f.writerow(['#model', model.__class__])
                params = model.get_all_params()
                param_names = model.get_all_param_names()
                for j in range(len(params)):
                    f.writerow([param_names[j], params[j]])


    @staticmethod
    def export_test(name, X, Ytrue, Ypred, Yvar_pred, pred_names):
        path = Experiments.get_output_path() + name +'/'
        check_dir_exists(path)
        file_name = 'test_'
        out = []
        out.append(Ytrue)
        out += Ypred
        out += Yvar_pred
        out.append(X)
        out = np.hstack(out)
        np.savetxt(path + file_name + '.csv', out
                   , header=''.join(
                                    ['Ytrue%d,'%(j) for j in range(Ytrue.shape[1])] +
                                    ['Ypred_%s_%d,'%(m, j) for m in pred_names for j in range(Ypred[0].shape[1])] +
                                    ['Yvar_pred_%s_%d,'%(m, j) for m in pred_names for j in range(Yvar_pred[0].shape[1])] +
                                    ['X%d,'%(j) for j in range(X.shape[1])]
                                    )
                    , delimiter=',', comments='')


    @staticmethod
    def export_configuration(name, config):
        path = Experiments.get_output_path() + name +'/'
        check_dir_exists(path)
        file_name = path + 'config_' + '.csv'
        with open(file_name, 'wb') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())
            writer.writeheader()
            writer.writerow(config)

    @staticmethod
    def get_ID():
        return id_generator(size=6)

    @staticmethod
    def run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, num_inducing, num_samples,
                  sparsify_factor, to_optimize):
        if method == 'full':
            m = SAVIGP_SingleComponent(Xtrain, Ytrain, num_inducing, cond_ll,
                                         kernel, num_samples, None, 0.001, False)
            Optimizer.optimize_model(m, 100000, True, to_optimize)
            y_pred, var_pred = m.predict(Xtest)
        if method == 'mix1':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 1, cond_ll,
                             kernel, num_samples, None, 0.001, False)
            Optimizer.optimize_model(m, 100000, True, to_optimize)
            y_pred, var_pred = m.predict(Xtest)
        if method == 'mix2':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 2, cond_ll,
                             kernel, num_samples, None, 0.001, False)
            Optimizer.optimize_model(m, 100000, True, to_optimize)
            y_pred, var_pred = m.predict(Xtest)
        if method == 'gp':
            m = GPy.models.GPRegression(Xtrain, Ytrain)
            m.optimize('bfgs')
            y_pred, var_pred = m.predict(Xtest)
        Experiments.export_train(name, Xtrain, Ytrain)
        Experiments.export_test(name, Xtest, Ytest, [y_pred], [var_pred], [''])
        if isinstance(m, SAVIGP):
            Experiments.export_model(m, name)
        Experiments.export_configuration(name, {'m': method, 'c': sparsify_factor, 's': num_samples})
        return name

    @staticmethod
    def boston_data(method, sparsify_factor):
        np.random.seed(12000)
        X, Y = DataSource.boston_data()
        X = preprocessing.scale(X)
        # Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, Ytest = Experiments.get_train_test(X, Y, 300)
        name = 'boston_' + Experiments.get_ID()
        kernel = [GPy.kern.RBF(X.shape[1], variance=1, lengthscale=np.array((1.,)))]
        gaussian_sigma = 1.0

        #number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 100000
        cond_ll = UnivariateGaussian(np.array(gaussian_sigma))

        return Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, num_inducing,
                                     num_samples, sparsify_factor, ['mog', 'hyp', 'll'])

    @staticmethod
    def breast_caner_data(method, sparsify_factor):
        np.random.seed(12000)
        X, Y = DataSource.wisconsin_breast_cancer_data()
        X = preprocessing.scale(X)
        # Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, Ytest = Experiments.get_train_test(X, Y, 300)
        name = 'breast_cancer_' + Experiments.get_ID()
        kernel = [GPy.kern.RBF(X.shape[1], variance=1, lengthscale=np.array((1.,)))]

        #number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 20000
        cond_ll = LogisticLL()

        return Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, num_inducing,
                                     num_samples, sparsify_factor, ['mog', 'hyp'])


    @staticmethod
    def gaussian_1D_data():
        gaussian_sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(1000, gaussian_sigma)
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 300)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = SAVIGP_SingleComponent(Xtrain, Ytrain, Xtrain.shape[0], MultivariateGaussian(np.array([[gaussian_sigma]])),
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
        m = SAVIGP_Diag(Xtrain, Ytrain, Xtrain.shape[0], 1, MultivariateGaussian(np.array([[sigma]])),
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
    plots = []

    # plots.append(Experiments.boston_data('gp', 1))
    plots.append(Experiments.boston_data('mix1', 0.1))


    # methods = ['mix1', 'mix2', 'full']
    # for m in methods:
        # plots.append(Experiments.boston_data(m, 1))

        # plots.append(Experiments.boston_data(m, 0.2))
        #
        # plots.append(Experiments.boston_data(m, 0.6))
        #
        # plots.append(Experiments.boston_data(m, 0.4))
        #
        # plots.append(Experiments.boston_data(m, 0.8))

    # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
    #                            lambda x: (x['m'] in ['mix2']), False)

    # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
    #                            lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)

    # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
    #                            lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)
