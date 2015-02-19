import math
from GPy.util.linalg import mdot
from scipy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

__author__ = 'AT'


class Optimizer:

    def __init__(self):
        pass

    @staticmethod
    def O_BFGS(model, x0, c, lambda_, epsilon, max_iter):
        f, f_grad = Optimizer.get_f_f_grad_from_model(model)
        dim = x0.shape[0]
        theta = x0
        B = epsilon * np.eye(dim)
        iter = 0
        eta = 0.1
        g = f_grad(theta)
        while(iter < max_iter):
            P = -mdot(B, g)
            s = eta / c * P
            theta = theta + s
            # theta = theta - 0.0001 * g
            g_n = f_grad(theta)
            y = g_n - g + lambda_ * s
            g = g_n
            if iter == 0:
                B = np.dot(s, y) / np.dot(y,y) * np.eye(dim)
            rho = 1 / np.dot(s, y)
            B = mdot((np.eye(dim) - rho * mdot(s[np.newaxis], y[np.newaxis].T)),
                     B,
                     (np.eye(dim) - rho * mdot(y[np.newaxis], s[np.newaxis].T))
            ) + c * rho * mdot(s[np.newaxis], s[np.newaxis].T)

            print f(theta)
            iter += 1


    @staticmethod
    def SGD(model, alpha, start, max_iter, ftol= 0.0001, xtol = 0.0001, verbose= True, factor = 1.0):
        f, f_grad = Optimizer.get_f_f_grad_from_model(model)
        iter = 0
        x = start
        last_f = float('Inf')
        k = 0
        while iter < max_iter:
            model._set_params(x)
            new_f = f(x)
            if verbose:
                print 'iter:' , iter, 'objective fun:', new_f, 'alpha:', alpha
            grad = f_grad(x)

            if alpha > factor / (new_f):
                alpha = factor / (new_f)

            if alpha < factor / (10 * new_f):
                alpha = factor / (10 * new_f)
            x -= grad * alpha
            if math.fabs(new_f - last_f) < ftol:
                return x, new_f

            # if new_f < last_f:
            #     alpha = (last_f - new_f) * 10
            last_f = new_f
            iter += 1

        return x

    @staticmethod
    def get_f_f_grad_from_model(model):
        def f(x):
            model._set_params(x)
            if np.isnan(model.objective_function()):
                print 'restart'
                raise Exception('restart')
            return model.objective_function()

        def f_grad(x):
            model._set_params(x)
            return model.objective_function_gradients()
        return f, f_grad


    @staticmethod
    def BFGS(model):
        f, f_grad = Optimizer.get_f_f_grad_from_model(model)
        fmin_l_bfgs_b(f, model._get_params(), f_grad, factr=100, epsilon=1e-3)
