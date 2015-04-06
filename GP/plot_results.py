import csv
import math
from matplotlib.pyplot import show, ion, savefig
import pandas
from pandas.util.testing import DataFrame
from util import check_dir_exists
import numpy as np

class PlotOutput:

    @staticmethod
    def plot_output(name, infile_path, model_names):
        SSE = {}
        NLPD = {}
        for m in model_names:
            data_test = pandas.read_csv(infile_path + m + '/' + 'test_' + m + '.csv')
            data_train = pandas.read_csv(infile_path + m + '/' + 'train_' + m + '.csv')
            Y_mean = data_train['Y0'].mean()
            Ypred = data_test['Ypred__0']
            Ytrue = data_test['Ytrue0']
            Yvar = data_test['Yvar_pred__0']
            SSE[m] = (Ypred - Ytrue)**2 / ((Y_mean - Ytrue) **2).mean()
            NLPD[m] = 0.5*(Ytrue-Ypred) ** 2./Yvar+np.log(2*math.pi*Yvar)
        SSE = DataFrame(SSE)
        ion()
        ax = SSE.plot(kind='box', title="SSE")
        ax.set_ylim(0, 1)
        check_dir_exists(infile_path + name + '/graphs/')
        savefig(infile_path + name + '/graphs/SSE.pdf')
        show(block=True)
        NLPD = DataFrame(NLPD)
        NLPD.plot(kind='box', title="NLPD")
        check_dir_exists(infile_path + name + '/graphs/')
        savefig(infile_path + name + '/graphs/NLPD.pdf')
        show(block=True)


if __name__ == '__main__':
    PlotOutput.plot_output('boston', '../../results/', ['boston_full', 'boston_mix1'])

