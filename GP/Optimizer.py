import math
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

__author__ = 'AT'


class Optimizer:

    def __init__(self):
        pass

    @staticmethod
    def SGD(model, alpha, start, max_iter, ftol= 0.0001, xtol = 0.0001, verbose= True, factor = 1.0):
        iter = 0
        x = start
        last_f = float('Inf')
        k = 0
        while iter < max_iter:
            model._set_params(x)
            new_f = model.objective_function()
            if verbose:
                print 'iter:' , iter, 'objective fun:', new_f, 'alpha:', alpha
            grad = model.objective_function_gradients()

            if alpha > factor / (new_f):
                alpha = factor / (new_f)

            x -= grad * alpha
            if math.fabs(new_f - last_f) < ftol:
                return x, new_f

            # if new_f < last_f:
            #     alpha = (last_f - new_f) * 10
            last_f = new_f
            iter += 1

        return x


    @staticmethod
    def loopy_opt(model):
        x_old = [model._get_params()]

        def f(x):
            model._set_params(x)
            if np.isnan(model.objective_function()):
                model._set_params(x_old[0])
                print 'restart'
                raise Exception('restart')
            x_old[0] = x
            print model.objective_function()
            return model.objective_function()

        def f_grad(x):
            model._set_params(x)
            return model.objective_function_gradients()

        restart = True
        while restart:
            # try:
                fmin_l_bfgs_b(f, model._get_params(), f_grad, factr=100, epsilon=1e-3)
                restart = False
            # except Exception as e:
            #     print e
            #     restart = True
