import math
from GPy.util.linalg import mdot
import nlopt
from numpy.ma import concatenate
from scipy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b, minimize, fmin_cg
import numpy as np

__author__ = 'AT'


class Optimizer:

    def __init__(self):
        pass

    @staticmethod
    def SGD(model, alpha, start, max_iter, ftol= 0.0001, xtol = 0.0001, verbose= True,
            factor = 1.0, opt_indices=None, adaptive_alpha=True):
        if opt_indices is None:
            opt_indices = range(0, len(start))
        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, start, opt_indices, verbose=verbose)
        iter = 0
        x = start
        last_f = float('Inf')
        delta_LR = 0.1
        avg_ftol = 100.
        while iter < max_iter:
            update(x)
            new_f = f()
            grad = f_grad()

            if adaptive_alpha and alpha > 1./ max(abs(grad)) / 10:
                alpha = 1./ max(abs(grad))  / 100
            if adaptive_alpha and new_f < last_f and alpha < 1./ max(abs(grad)) / 1000:
                alpha = min(1. / max(abs(grad)) / 500, 0.001)
            print alpha
            if verbose:
                print 'alpha', alpha,
            x -= grad * alpha
            if avg_ftol < ftol:
                return x, new_f
            if iter > 1 and new_f < last_f:
                avg_ftol = (1 - delta_LR) * avg_ftol + delta_LR * math.fabs(last_f - new_f)
            last_f = new_f
            iter += 1
        return x

    @staticmethod
    def get_f_f_grad_from_model(model, x0, opt_indices, verbose = False):
        last_x = np.empty((1, x0.shape[0]))
        def update(x):
            if np.array_equal(x, last_x[0]):
                return
            last_x[0] = x
            p = x0.copy()
            p[opt_indices] = x[opt_indices]
            model.set_params(x)

        def f(X=None):
            if X is not None:
                update(X)
            return model.objective_function()

        def f_grad(X=None):
            if X is not None:
                update(X)

            g = np.zeros(len(x0))
            g[opt_indices] = model.objective_function_gradients().copy()[opt_indices]
            if verbose:
                # print 'grad:', Optimizer.print_short(g)
                print 'objective:', "%.4f" % model.objective_function()
            return g
        update(x0)
        return f, f_grad, update


    @staticmethod
    def BFGS(model, opt_indices=None, max_fun=None, verbose=False):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, verbose=verbose)
        x, f, d = fmin_l_bfgs_b(f, start, f_grad, factr=5, epsilon=1e-3, maxfun=max_fun,
                      callback=lambda x: update(x))
        return d

    @staticmethod
    def CG(model, opt_indices=None):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices)
        fmin_cg(f, start, f_grad, epsilon=1e-6,
                      callback=lambda x: update(x))

    @staticmethod
    def NLOPT(model, algorithm, opt_indices=None):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices)

        def myfunc(x, grad):
            update(x)
            if grad.size > 0:
                grad[:] = f_grad()
            return f()

        opt = nlopt.opt(algorithm, len(model.get_params()))
        opt.set_min_objective(myfunc)
        opt.set_ftol_rel(1e-3)
        return opt.optimize(model.get_params())

    @staticmethod
    def general(model, opt_indices=None):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices)
        minimize(f, start, jac=f_grad, method='Newton-CG',
                      callback=lambda x: update(x))

    @staticmethod
    def print_short(a):
        return ["%.2f" % a[j] for j in range(len(a))]