import csv
import math
import os
from matplotlib.pyplot import show, ion, savefig
import pandas
from pandas.util.testing import DataFrame
from util import check_dir_exists
import numpy as np

class PlotOutput:

    @staticmethod
    def plot_output(name, infile_path, model_names, filter, export_pdf):
        graphs = {}
        graphs['SSE'] = {}
        graphs['NLPD'] = {}
        graphs['ER'] = {}
        for m in model_names:
            data_config = PlotOutput.read_config(infile_path + m + '/' + 'config_' + '.csv')
            if filter is None or filter(data_config):
                data_test = pandas.read_csv(infile_path + m + '/' + 'test_' +  '.csv')
                data_train = pandas.read_csv(infile_path + m + '/' + 'train_' + '.csv')
                Y_mean = data_train['Y0'].mean()
                Ypred = data_test['Ypred__0']
                Ytrue = data_test['Ytrue0']
                Yvar = data_test['Yvar_pred__0']

                if data_config['ll'] in ['UnivariateGaussian']:
                    graphs['SSE'][str(data_config)] = (Ypred - Ytrue)**2 / ((Y_mean - Ytrue) **2).mean()
                    graphs['NLPD'][str(data_config)] = 0.5*(Ytrue-Ypred) ** 2./Yvar+np.log(2*math.pi*Yvar)

                if data_config['ll'] in ['LogisticLL']:
                    graphs['ER'][str(data_config)] = np.array([(((Ypred > 0.5) & (Ytrue == -1))
                                                                 | ((Ypred < 0.5) & (Ytrue == 1))
                                                                 ).mean()])
                    graphs['NLPD'][str(data_config)] = -np.log((-Ytrue + 1) / 2 + Ytrue * Ypred)


        for n, g in graphs.iteritems():
            if g:
                g= DataFrame(g)
                ion()
                if n in ['SSE', 'NLPD']:
                    g.plot(kind='box', title=n)
                if n in ['ER']:
                    ax =g.plot(kind='bar', title=n)
                    patches, labels = ax.get_legend_handles_labels()
                    ax.legend(patches, labels, loc='lower center')
                if export_pdf:
                    check_dir_exists(infile_path + name + '/graphs/')
                    savefig(infile_path + name + '/graphs/'+'n' + '.pdf')
                show(block=True)

    @staticmethod
    def plot_output_all(name, path, filter, export_pdf):
        dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        PlotOutput.plot_output(name, path, dir, filter, export_pdf)

    @staticmethod
    def read_config(path):
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            return reader.next()

if __name__ == '__main__':
    PlotOutput.plot_output('boston', '../../results/', ['boston_full', 'boston_mix1', 'boston_mix2', 'boston_gp'], None, True)

