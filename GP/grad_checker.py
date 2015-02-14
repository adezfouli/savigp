from DerApproximator import get_d1
from numpy import hstack, concatenate
from texttable import Texttable


__author__ = 'AT'

class GradChecker:
    @staticmethod
    def check(f, f_grad, x0, name):
        g = f_grad(x0)
        table = Texttable()
        table.set_cols_align(["l", "r", "c", "c"])
        table.set_cols_valign(["t", "m", "b" , "r"])
        rows = []
        rows += [["Name  ", "analytical  ", "numerical   ", "error   "]]
        print 'dimensions:', len(x0)
        for i in range(len(x0)):
            def f_i(x):
                return f((concatenate((x0[:i], x, x0[(i+1):]))))
            t = get_d1(f_i, [x0[i]], diffInt=1.0)
            rows += [[name[i], g[i], t, abs(t-g[i])]]
            print 'element:', i
        table.add_rows(rows)
        print(table.draw())