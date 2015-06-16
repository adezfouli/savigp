__author__ = 'AT'

import math
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
        """
        Receives a model and extracts needed functions and attributes to use the model with an optimiser.

        Parameters
        ----------
        model : object
         any object which exposes `objective_function` and `objective_function_gradients`

        x0 : ndarray
         starting point of optimisation

        opt_indices:
         indices of the elements to optimise

        tracker : list
         keeps track of objective function values

        logger : logger
         use for logging objective function and gradients

        Returns
        -------
        f : callable
         a function which returns objective function for the input parameter

        f_grad : callable
         a function which returns gradient of the objective function for the input parameter

        update : callable
         updates the model to reflect new parameters

        min_x : callable
         the parameter with lowest objective function

        total_evals : int
         total evaluation of the objective function

        """
        last_x = np.empty((1, x0.shape[0]))
        last_x[0] = x0
        best_f = {'f': None}
        best_x = np.empty((1, x0.shape[0]))
        best_x[0] = x0
        best_f['f'] = model.objective_function()
        total_f_evals = np.array([0])

        def update(x):
            if np.array_equal(x, last_x[0]):
                return
            # p = x0.copy()
            # p[opt_indices] = x[opt_indices]
            try:
                model.set_params(x.copy())
                total_f_evals[0] += 1
                last_x[0] = x
            except (ValueError, JitChol) as e:
                best_x[0] = last_x[0].copy()
                raise OptTermination(e)
            if best_f['f'] is None:
                best_f['f'] = model.objective_function()
            else:
                if model.objective_function() < best_f['f']:
                    best_f['f'] = model.objective_function()
                    best_x[0] = x

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

        def total_evals():
            return total_f_evals[0]

        update(x0)
        return f, f_grad, update, min_x, total_evals


    @staticmethod
    def BFGS(model, logger, opt_indices=None, max_fun=None, apply_bound=False):
        """
        Optimise the `model` using l_bfgs_b algorithm.

        Parameters
        ----------
        model : model
         the model to optimise

        logger : logger
         logger used for logging

        opt_indices : ndarray
         indices of the parameters that will be optimised. If None, all the parameters will be optimised.

        max_fun : int
         maximum number of function evaluations

        apply_bound : boolean
         whether to apply bounds. If True, parameters will be limited to be less than log (1e10)

        """
        start = model.get_params()
        if opt_indices is None:
            opt_indices = range(0, len(start))

        tracker = []
        bounds = None
        if apply_bound:
            bounds = []
            for x in range(start.shape[0]):
                bounds.append((None, math.log(1e+10)))
        init_x = model.get_params()
        f, f_grad, update, best_x, total_evals = Optimizer.get_f_f_grad_from_model(model, init_x, opt_indices, tracker, logger)
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
        d['funcalls'] = total_evals()
        return d, tracker



    @staticmethod
    def print_short(a):
        return ["%.2f" % a[j] for j in range(len(a))]

    @staticmethod
    def optimize_model(model, max_fun_evals, logger,
                       method=None, xtol=1e-4, iters_per_opt=[25, 25, 25], max_iters=200,
                       ftol =1e-5, callback=None, current_iter=None):
        """
        Optimised model in an EM manner, i.e., each set of parameters are optimised independently, e.g.,

                 |---> MoG ---> hyp ---> ll ----|
                 |---<-------- <--------<-------|

        Parameters
        ----------
        model : model
         the model to optimise

        max_fun_evals : int
         maximum number of function evaluatioins

        logger : logger
         logger

        method : list
         the set of parameters to optimise. For example method = ['mog', 'hyp'] will optimise posterior distribution
         and hyper-parameters.

        iters_per_opt : dictionary
         a dictionary containing maximum number of function evaluations for each subset of parameters in each local
         optimisation. For example, iters_per_opt = {'mog' : 25, 'hyp' : 30} will update posterior parameters (mog)
         for a maximum of 25 function evaluations and hyper-parameter for a maximum of 30 function evaluation.

        max_iters : int
         maximum number of global optimisations.

        xtol : float
         tolerance in the parameters which determines convergence. Tolerance is calculated as the average of posterior
         mean and covariance, i.e., tol = (delta m + delta s) / 2

        ftol : float
         tolerance in the objective function which determines convergence.

        callback : callable
         a function which will be called after optimisation of the posterior parameters.

        current_iter : int
         current iteration of the optimisation. It is useful for example in the case that the optimisation is continued
         from a previous optimisation.
        """

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
        if current_iter is None:
            current_iter = 0
        last_obj = None
        delta_m = None
        delta_s = None
        try:
            while (max_iters is None) or current_iter < max_iters:
                logger.info('iter started ' + str(current_iter))
                if 'mog' in method:
                    logger.info('mog params')
                    model.set_configuration([
                        Configuration.MoG,
                        Configuration.ENTROPY,
                        Configuration.CROSS,
                        Configuration.ELL,
                    ])
                    d, tracker = Optimizer.BFGS(model, logger, max_fun=iters_per_opt['mog'])
                    obj_track += tracker
                    total_evals += d['funcalls']

                if callback is not None:
                    logger.info('callback...')
                    callback(model, current_iter + 1, total_evals, delta_m, delta_s, obj_track)
                    logger.info('callback finished')

                # check for convergence
                new_params_m, new_params_s = model.get_posterior_params()
                if last_param_m is not None:
                    delta_m = np.absolute(new_params_m - last_param_m).mean()
                    delta_s = np.absolute(new_params_s - last_param_s).mean()
                    logger.debug('ftol: ' + str(last_obj - model.objective_function()))
                    logger.info('diff:' + 'm:' +  str(delta_m) + ' s:' + str(delta_s))
                    if (delta_m + delta_s) / 2 < xtol or \
                            ((last_obj > model.objective_function()) and (last_obj - model.objective_function() < ftol)):
                        logger.info('best obj found: ' + str(model.objective_function()))
                        break
                last_param_m = new_params_m
                last_param_s = new_params_s
                last_obj = model.objective_function()

                if 'll' in method:
                    logger.info('ll params')
                    model.set_configuration([
                        Configuration.ELL,
                        Configuration.LL
                    ])
                    d, tracker = Optimizer.BFGS(model, logger, max_fun=iters_per_opt['ll'])
                    obj_track += tracker
                    total_evals += d['funcalls']

                if 'hyp' in method:
                    logger.info('hyp params')
                    model.set_configuration([
                        Configuration.ENTROPY,
                        Configuration.CROSS,
                        Configuration.ELL,
                        Configuration.HYPER
                    ])
                    d, tracker = Optimizer.BFGS(model, logger, max_fun=iters_per_opt['hyp'], apply_bound=True)
                    obj_track += tracker
                    total_evals += d['funcalls']


                if not (max_fun_evals is None) and total_evals > max_fun_evals:
                    break

                current_iter += 1

        except KeyboardInterrupt:
            logger.info('interrupted by the user')
            logger.info('last obj: ' + str(model.objective_function()))
            if total_evals == 0:
                total_evals = float('Nan')
        end=time.time()
        if total_evals == 0:
            avg_time = None
        else:
            avg_time = (end - start) / total_evals
        return model, avg_time, (end - start), obj_track, total_evals


class OptTermination(Exception):
    """
    a specific class when to indicate problems during optimisation, e.g., problems in function evaluation.
    """
    pass