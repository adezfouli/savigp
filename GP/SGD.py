import math

__author__ = 'AT'


class SGD:

    def __init__(self):
        pass

    @staticmethod
    def optimize(model, alpha, start, max_iter, ftol= 0.0001, xtol = 0.0001, verbose= True):
        iter = 0
        x = start
        last_f = float('Inf')
        while iter < max_iter:
            # model._set_params(x)
            new_f = model.objective_function(x)
            grad = model.objective_function_gradients(x)
            x -= grad * alpha
            if verbose:
                print 'iter:' , iter, 'objective fun:', new_f
            if math.fabs(new_f - last_f) < ftol:
                return x, new_f
            last_f = new_f
            iter += 1

        return x