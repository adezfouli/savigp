import math
from GPy.util.linalg import mdot
import nlopt
from numpy.ma import concatenate
from scipy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b, minimize, fmin_cg
import numpy as np
import time
from savigp import Configuration

__author__ = 'AT'


class Optimizer:
    def __init__(self):
        pass

    @staticmethod
    def SGD(model, alpha, start, max_iter, ftol=0.0001, xtol=0.0001, verbose=True,
            factor=1.0, opt_indices=None, adaptive_alpha=True, show_alpha= False):
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

            if adaptive_alpha and alpha > 1. / max(abs(grad)) / 10:
                alpha = 1. / max(abs(grad)) / 100
            if adaptive_alpha and new_f < last_f and alpha < 1. / max(abs(grad)) / 1000:
                alpha = min(1. / max(abs(grad)) / 500, 0.001)
            if show_alpha:
                print 'alpha', alpha,
            x -= grad * alpha
            if avg_ftol < ftol:
                return x, new_f
            if iter > 1 and new_f < last_f:
                avg_ftol = (1 - delta_LR) * avg_ftol + delta_LR * math.fabs(last_f - new_f)
            last_f = new_f
            iter += 1
        d = {}
        d['funcalls'] = iter
        return d

    @staticmethod
    def get_f_f_grad_from_model(model, x0, opt_indices, verbose=False):
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
    def CG(model, opt_indices=None, verbose=False):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, verbose=verbose)
        fmin_cg(f, start, f_grad, epsilon=1e-6,
                callback=lambda x: update(x))

    @staticmethod
    def NLOPT(model, algorithm, opt_indices=None, verbose=False):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, verbose=verbose)

        def myfunc(x, grad):
            update(x)
            if grad.size > 0:
                grad[:] = f_grad()
            return f()

        opt = nlopt.opt(algorithm, len(model.get_params()))
        opt.set_min_objective(myfunc)
        opt.set_ftol_rel(1e-3)
        opt_x = opt.optimize(model.get_params())
        d = {'opt_params': opt_x, 'funcalls': 1}
        return d

    @staticmethod
    def general(model, opt_indices=None, verbose=False):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, verbose=verbose)
        minimize(f, start, jac=f_grad, method='Newton-CG',
                 callback=lambda x: update(x))

        return {'funcalls': 1}

    @staticmethod
    def print_short(a):
        return ["%.2f" % a[j] for j in range(len(a))]

    @staticmethod
    def optimize_model(model, max_fun, verbose, method=None, epsilon=1e-4, opt_iter=200):
        if not method:
            method=['hyp', 'mog']
        if not (max_fun is None):
            opt_iter = min(max_fun, opt_iter)
        converged=False
        start=time.time()
        total_evals = 0
        last_param = None
        try:
            while not converged:
                if 'mog' in method:
                    print 'mog params'
                    model.set_configuration([
                        Configuration.MoG,
                        Configuration.ENTROPY,
                        Configuration.CROSS,
                        Configuration.ELL,
                    ])
                    d = Optimizer.BFGS(model, max_fun=opt_iter, verbose=verbose)
                    # d = Optimizer.NLOPT(model, algorithm=nlopt.LD_LBFGS, verbose=verbose)
                    # d = Optimizer.SGD(model, alpha=1e-6, start=model.get_params(), max_iter=10, adaptive_alpha=False)
                    # d = Optimizer.general(model, verbose=verbose)
                    total_evals += d['funcalls']

                # check for convergence
                new_params = model.get_params()
                if last_param is not None:
                    if np.mean(np.absolute(new_params - last_param)) < epsilon:
                        break
                    print 'diff:', np.mean(np.absolute(new_params - last_param))
                last_param = new_params

                if 'hyp' in method:
                    print 'hyp params'
                    model.set_configuration([
                        Configuration.ENTROPY,
                        Configuration.CROSS,
                        Configuration.ELL,
                        Configuration.HYPER
                    ])
                    d = Optimizer.BFGS(model, max_fun=opt_iter, verbose=verbose)
                    total_evals += d['funcalls']

                if 'll' in method:
                    print 'll params'
                    model.set_configuration([
                        Configuration.ELL,
                        Configuration.LL
                    ])
                    d = Optimizer.BFGS(model, max_fun=opt_iter, verbose=verbose)
                    total_evals += d['funcalls']

                if not (max_fun is None) and total_evals > max_fun:
                    break

        except KeyboardInterrupt:
            print 'interrupted by the user'
            if total_evals == 0:
                total_evals = float('Nan')
        end=time.time()
        return model, (end - start) / total_evals, (end - start)
