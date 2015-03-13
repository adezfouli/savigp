from DerApproximator import get_d1
from numpy import hstack, concatenate
from texttable import Texttable


__author__ = 'AT'

class GradChecker:
    @staticmethod
    def check(f, f_grad, x0, name, verbose= False):
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
            if g[i] != 0:
                p_errro = abs(t-g[i]) / abs(g[i])
            rows += [[name[i], g[i], t, abs(t-g[i]), p_errro]]
            if g[i] != 0:
                aver_error += abs(t-g[i]) / abs(g[i])
            if verbose:
                print 'element:', i
        table.add_rows(rows)
        if verbose:
            print(table.draw())
        return aver_error / len(x0)