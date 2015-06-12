__author__ = 'AT'

from DerApproximator import get_d1
from numpy import concatenate
from texttable import Texttable


class GradChecker:
    """ A class for checking gradients. """
    def __init__(self):
        pass

    @staticmethod
    def check(f, f_grad, x0, name, verbose=False):
        """
        Checks whether gradient of function 'f' at point x0 is same as the gradients provided by 'f_grad'. 'error'
        is the difference between numerical and provided gradients. '%error' = abs(error) / numerical gradient.

        :param f: input function to check gradients against \n
        :param f_grad: input function which provides gradients \n
        :param x0: the point at which gradients should be calculated \n
        :param name: a vector with the size of the number of parameters, which provides name for each parameter. This
        name will be used when generating output table\n
        :param verbose: whether to print output for each parameter separately\n
        :return: average of the percentage error over all the parameters, i.e., mean(%error)
        """

        g = f_grad(x0)
        if len(g) != len(x0):
            raise Exception('dimensions mismatch')
        table = Texttable()
        table.set_cols_align(["l", "r", "c", "c", "c"])
        table.set_cols_valign(["t", "m", "b" , "r", "c"])
        rows = []
        rows += [["Name  ", "analytical  ", "numerical   ", "error   ", "% error   "]]
        if verbose:
            print 'dimensions:', len(x0)
        aver_error = 0
        for i in range(len(x0)):
            def f_i(x):
                return f((concatenate((x0[:i], x, x0[(i+1):]))))
            t = get_d1(f_i, [x0[i]])
            p_errro=None
            if t != 0:
                p_errro = abs(t-g[i]) / abs(t)
            rows += [[name[i], g[i], t, abs(t-g[i]), p_errro]]
            if abs(g[i]) <1e-4 and abs(t) < 1e-4:
                pass
            else:
                aver_error += abs(t-g[i]) / abs(t)
            if verbose:
                print 'element:', i
        table.add_rows(rows)
        if verbose:
            print(table.draw())
        return aver_error / len(x0)