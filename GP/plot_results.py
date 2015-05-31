import csv
import math
import os
from re import match
from matplotlib.lines import Line2D
from matplotlib.pyplot import show, ion, savefig
import pandas
from pandas.util.testing import DataFrame, Series
from likelihood import SoftmaxLL, LogisticLL, UnivariateGaussian, LogGaussianCox, WarpLL
from util import check_dir_exists
import numpy as np
import matplotlib.pyplot as plt

class PlotOutput:

    @staticmethod
    def plot_output(name, infile_path, model_names, filter, export_pdf):
        graphs = {}
        graphs['SSE'] = {}
        graphs['NLPD'] = {}
        graphs['ER'] = {}
        graphs['intensity'] = {}
        graph_n = {}
        for m in model_names:
            data_config = PlotOutput.read_config(infile_path + m + '/' + 'config_' + '.csv')
            if filter is None or filter(data_config):
                data_test = pandas.read_csv(infile_path + m + '/' + 'test_' +  '.csv')
                cols = data_test.columns
                dim = 0
                for element in cols:
                    if element.startswith('Ytrue'):
                        dim += 1

                data_train = pandas.read_csv(infile_path + m + '/' + 'train_' + '.csv')
                Y_mean = data_train['Y0'].mean()

                Ypred = np.array([data_test['Ypred__%d' % (d)] for d in range(dim)])
                Ytrue = np.array([data_test['Ytrue%d' % (d)] for d in range(dim)])
                Yvar = np.array([data_test['Yvar_pred__%d' % (d)] for d in range(dim)])

                if not (PlotOutput.config_to_str(data_config) in graph_n.keys()):
                    graph_n[PlotOutput.config_to_str(data_config)] = 0
                graph_n[PlotOutput.config_to_str(data_config)] += 1

                if data_config['ll'] in [UnivariateGaussian.__name__, WarpLL.__name__]:
                    NLPD = np.array(data_test['nlpd'])
                    PlotOutput.add_to_list(graphs['SSE'], PlotOutput.config_to_str(data_config),
                                           (Ypred[0] - Ytrue[0])**2 / ((Y_mean - Ytrue[0]) **2).mean())
                    PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config), NLPD)

                if data_config['ll'] in [LogisticLL.__name__]:
                    NLPD = np.array(data_test['nlpd'])
                    PlotOutput.add_to_list(graphs['ER'], PlotOutput.config_to_str(data_config), np.array([(((Ypred[0] > 0.5) & (Ytrue[0] == -1))
                                                                 | ((Ypred[0] < 0.5) & (Ytrue[0] == 1))
                                                                 ).mean()]))
                    PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config), NLPD)

                if data_config['ll'] in [SoftmaxLL.__name__]:
                    NLPD = np.array(data_test['nlpd'])
                    PlotOutput.add_to_list(graphs['ER'], PlotOutput.config_to_str(data_config), np.array(
                        [(np.argmax(Ytrue, axis=0) != np.argmax(Ypred, axis=0)).mean()]))
                    PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config), NLPD)

                if data_config['ll'] in [LogGaussianCox.__name__]:
                    X0 = np.array([data_test['X0']])

                    PlotOutput.add_to_list(graphs['intensity'], PlotOutput.config_to_str(data_config),
                                           np.array([X0[0,:]/365+1851.2026, Ypred[0,:], Yvar[0,:]] ).T)

        for n, g in graphs.iteritems():
            if g:
                ion()
                for k in g.keys():
                    print k, 'n: ', graph_n[k]
                if n in ['SSE', 'NLPD']:
                    g= DataFrame(dict([(k,Series(v)) for k,v in g.iteritems()]))
                    ax = g.plot(kind='box', title=n)
                    check_dir_exists('../graph_data/')
                    g.to_csv('../graph_data/' + name  + '_' + n + '_data.csv')
                if n in ['ER']:
                    g= DataFrame(dict([(k,Series(v)) for k,v in g.iteritems()]))
                    check_dir_exists('../graph_data/')
                    g.to_csv('../graph_data/' + name  + '_' + n + '_data.csv')
                    m = g.mean()
                    errors = g.std()
                    ax =m.plot(kind='bar', yerr=errors, title=n)
                    patches, labels = ax.get_legend_handles_labels()
                    ax.legend(patches, labels, loc='lower center')
                if n in ['intensity']:
                    X = g.values()[0][:, 0]
                    plt.figure()
                    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
                    c = 0
                    check_dir_exists('../graph_data/')
                    graph_data = DataFrame()
                    for k,v in g.iteritems():
                        plt.plot(X, v[:, 1], hold=True, color=color[c], label=k)
                        plt.fill_between(X, v[:, 1] - 2 * np.sqrt(v[:, 2]), v[:, 1] + 2 * np.sqrt(v[:, 2]), alpha=0.2, facecolor=color[c])
                        graph_data = graph_data.append(DataFrame({'x': X, 'm' : v[:, 1], 'v' :v[:, 2], 'model_sp' :[k] * X.shape[0]}
                                                                 ))
                        c += 1
                    plt.legend(loc='upper center')
                    graph_data.to_csv('../graph_data/' + name  + '_' + n + '_data.csv')

                if export_pdf:
                    check_dir_exists(infile_path + name + '/graphs/')
                    savefig(infile_path + name + '/graphs/'+'n' + '.pdf')
                show(block=True)

    @staticmethod
    def add_to_list(l, name, value):
        if not (name in l):
            l[name] = value
        else:
            l[name] = np.hstack((l[name], value))

    @staticmethod
    def config_to_str(config):
        return config['method'] +':' + str(config['sparsify_factor'])

    @staticmethod
    def plot_output_all(name, path, filter, export_pdf):
        dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        PlotOutput.plot_output(name, path, dir, filter, export_pdf)


    @staticmethod
    def find_all(path, filter):
        dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for m in dir:
            data_config = PlotOutput.read_config(path + m + '/' + 'config_' + '.csv')
            if filter is None or filter(data_config):
                print m

    @staticmethod
    def read_config(path):
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            return reader.next()

if __name__ == '__main__':
    PlotOutput.plot_output('boston', '../../results/', ['boston_full', 'boston_mix1', 'boston_mix2', 'boston_gp'], None, True)

