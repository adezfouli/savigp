__author__ = 'AT'

import math
from GPy.util.linalg import mdot
# import nlopt
from numpy.ma import concatenate
from scipy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b, minimize, fmin_cg
import numpy as np
import time
from util import JitChol
from savigp import Configuration



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
    def get_f_f_grad_from_model(model, x0, opt_indices, tracker, logger):
        last_x = np.empty((1, x0.shape[0]))
        best_f = {'f': None}
        best_x = np.empty((1, x0.shape[0]))
        best_x[0] = x0.copy()
        best_f['f'] = model.objective_function()

        def update(x):
            if np.array_equal(x, last_x[0]):
                return
            # p = x0.copy()
            # p[opt_indices] = x[opt_indices]
            try:
                model.set_params(x)
                last_x[0] = x
            except (ValueError, JitChol) as e:
                best_x[0] = last_x[0].copy()
                raise OptTermination(e)
            if best_f['f'] is None:
                best_f['f'] = model.objective_function()
            else:
                if model.objective_function() < best_f['f']:
                    best_f['f'] = model.objective_function()
                    best_x[0] = x.copy()

        def f(X=None):
            if X is not None:
                update(X)
            obj = model.objective_function()
            tracker.append(obj)
            logger.debug('objective:' + "%.4f" % model.objective_function())
            return obj

        def f_grad(X=None):
            if X is not None:
                update(X)

            g = np.zeros(len(x0))
            g[opt_indices] = model.objective_function_gradients().copy()[opt_indices]
            return g

        def min_x():
            return best_x[0]

        update(x0)
        return f, f_grad, update, min_x


    @staticmethod
    def BFGS(model, logger, opt_indices=None, max_fun=None, apply_bound=False):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        tracker = []
        bounds = None
        if apply_bound:
            bounds = []
            for x in range(start.shape[0]):
                bounds.append((None, math.log(1e+10)))
        init_x = model.get_params().copy()
        f, f_grad, update, best_x = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, tracker, logger)
        restart_opt = True
        opt_tries = 0
        while restart_opt:
            try:
                x, f, d = fmin_l_bfgs_b(f, start, f_grad, factr=5, epsilon=1e-3, maxfun=max_fun,
                              callback=lambda x: update(x), bounds=bounds)
                update(best_x())
                restart_opt = False
            except OptTermination as e:
                logger.warning('invalid value encountered. Opt restarted')
                update(init_x)
                opt_tries += 1
                max_fun = 2
                if opt_tries < 3:
                    restart_opt = True
                    max_fun = 3 - opt_tries
                else:
                    logger.warning('cannot restart opt. Opt terminated')
                    restart_opt = False

        d = {}
        d['funcalls'] = 1
        return d, tracker

    @staticmethod
    def CG(model, opt_indices=None, verbose=False):
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, verbose=verbose)
        fmin_cg(f, start, f_grad, epsilon=1e-6,
                callback=lambda x: update(x))

    # @staticmethod
    # def NLOPT(model, algorithm, opt_indices=None, verbose=False):
    #     start = model.get_params()
    #     if opt_indices is None:
    #         opt_indices = range(0, len(start))
    #
    #     f, f_grad, update = Optimizer.get_f_f_grad_from_model(model, model.get_params(), opt_indices, verbose=verbose)
    #
    #     def myfunc(x, grad):
    #         update(x)
    #         if grad.size > 0:
    #             grad[:] = f_grad()
    #         return f()
    #
    #     opt = nlopt.opt(algorithm, len(model.get_params()))
    #     opt.set_min_objective(myfunc)
    #     opt.set_ftol_rel(1e-3)
    #     opt_x = opt.optimize(model.get_params())
    #     d = {'opt_params': opt_x, 'funcalls': 1}
    #     return d

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
    def optimize_model(model, max_fun_evals, logger, method=None, xtol=1e-4, iters_per_opt=15000, max_iters=200, ftol =1e-5):
        if not method:
            method=['hyp', 'mog']
        if not (max_fun_evals is None):
            iters_per_opt = min(max_fun_evals, iters_per_opt)
        converged=False
        start=time.time()
        total_evals = 0
        last_param_m = None
        last_param_s = None
        obj_track = []
        current_iter = 1
        last_obj = None
        try:
            while not converged:
                if 'hyp' in method:
                    logger.info('hyp params')
                    model.set_configuration([
                        Configuration.ENTROPY,
                        Configuration.CROSS,
                        Configuration.ELL,
                        Configuration.HYPER
                    ])
                    d, tracker = Optimizer.BFGS(model, logger, max_fun=iters_per_opt, apply_bound=True)
                    obj_track += tracker
                    total_evals += d['funcalls']

                if 'mog' in method:
                    logger.info('mog params')
                    model.set_configuration([
                        Configuration.MoG,
                        Configuration.ENTROPY,
                        Configuration.CROSS,
                        Configuration.ELL,
                    ])
                    d, tracker = Optimizer.BFGS(model, logger, max_fun=iters_per_opt)
                    # d = Optimizer.NLOPT(model, algorithm=nlopt.LD_LBFGS, verbose=verbose)
                    # d = Optimizer.SGD(model, alpha=1e-6, start=model.get_params(), max_iter=10, adaptive_alpha=False)
                    # d = Optimizer.general(model, verbose=verbose)
                    obj_track += tracker
                    total_evals += d['funcalls']

                # check for convergence
                new_params_m, new_params_s = model.get_posterior_params()
                if last_param_m is not None:
                    delta_m = np.absolute(new_params_m - last_param_m).mean()
                    delta_s = np.absolute(new_params_s - last_param_s).mean()
                    if (delta_m + delta_s) / 2 < xtol or \
                            ((last_obj > model.objective_function()) and (last_obj - model.objective_function() < ftol)):
                        logger.info('best obj found: ' + str(model.objective_function()))
                        break
                    logger.debug('ftol: ' + str(last_obj - model.objective_function()))
                    logger.info('diff:' + 'm:' +  str(delta_m) + ' s:' + str(delta_s))
                last_param_m = new_params_m
                last_param_s = new_params_s
                last_obj = model.objective_function()

                if 'll' in method:
                    logger.info('ll params')
                    model.set_configuration([
                        Configuration.ELL,
                        Configuration.LL
                    ])
                    d, tracker = Optimizer.BFGS(model, logger, max_fun=iters_per_opt)
                    obj_track += tracker
                    total_evals += d['funcalls']



                if not (max_fun_evals is None) and total_evals > max_fun_evals:
                    break

                current_iter += 1

                if not (max_iters is None) and current_iter > max_iters:
                    break


        except KeyboardInterrupt:
            logger.info('interrupted by the user')
            logger.info('last obj: ' + str(model.objective_function()))
            if total_evals == 0:
                total_evals = float('Nan')
        end=time.time()
        return model, (end - start) / total_evals, (end - start), obj_track

class OptTermination(Exception):
    pass