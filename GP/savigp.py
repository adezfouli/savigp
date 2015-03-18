from scipy.misc import logsumexp
from mog_diag import MoG_Diag
from util import mdiag_dot, jitchol, pddet, inv_chol, nearPD, cross_ent_normal, log_diag_gaussian
from aetypes import Enum
import math
from GPy.util.linalg import mdot
import numpy as np
from numpy.ma import trace, vstack
from scipy.linalg import logm, det, cho_solve
import scipy.stats
from GPy import likelihoods
from GPy.core import Model
from line_profiler import *


class Configuration(Enum):
    ENTROPY = 'ENT'
    CROSS = 'CRO'
    ELL = 'ELL'
    HYPER = 'HYP'
    MoG = 'MOG'


class SAVIGP(Model):
    """
    Scalable Variational Inference Gaussian Process

    :param X: input observations
    :param Y: outputs
    :param num_inducing: number of inducing variables
    :param num_MoG_comp: number of components of the MoG
    :param likelihood: conditional likelihood function
    :param kernels: list of kernels of the HP
    :param n_samples: number of samples drawn for approximating ell and its gradient
    :rtype: model object
    """

    def __init__(self, X, Y, num_inducing, num_mog_comp, likelihood, kernels, n_samples, config_list=None):

        super(SAVIGP, self).__init__("SAVIGP")
        if config_list is None:
            self.config_list = [Configuration.CROSS, Configuration.ELL, Configuration.ENTROPY]
        else:
            self.config_list = config_list
        self.num_latent_proc = len(kernels)
        self.num_mog_comp = num_mog_comp
        self.num_inducing = num_inducing
        self.MoG = self._get_mog()
        self.input_dim = X[0].shape[0]
        self.output_dim = Y[0].shape[0]
        self.kernels = kernels
        self.cond_likelihood = likelihood
        self.X = X
        self.Y = Y
        self.n_samples = n_samples
        self.param_names = []
        self.last_param = None
        self.hyper_params = None
        self.sparse = X.shape[0] != self.num_inducing

        Z = np.array([np.zeros((self.num_inducing, self.input_dim))] * self.num_latent_proc)

        for j in range(self.num_latent_proc):
            if self.num_inducing == X.shape[0]:
                i = range(self.X.shape[0])
            else:
                i = np.random.permutation(X.shape[0])[:self.num_inducing]
            Z[j, :, :] = X[i].copy()

        # Z is Q * M * D
        self.Z = Z
        self.invZ = np.array([np.empty((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.chol = np.array([np.zeros((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.invZ = np.array([np.zeros((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.log_detZ = np.zeros(self.num_latent_proc)

        self.normal_samples = np.random.normal(0, 1, self.n_samples * self.num_latent_proc) \
            .reshape((self.num_latent_proc, self.n_samples))

        self._update_inverses()

        self.init_mog()

        self._update()

    def init_mog(self):
        pass

    def rand_init_mog(self):
        self.MoG.random_init()

    def _get_mog(self):
        return MoG_Diag(self.num_mog_comp, self.num_latent_proc, self.num_inducing)

    def get_param_names(self):
        if Configuration.MoG in self.config_list:
            self.param_names += ['m'] * self.MoG.get_m_size() + ['s'] * \
                                                                self.MoG.get_s_size() + ['pi'] * self.num_mog_comp

        if Configuration.HYPER in self.config_list:
            self.param_names += ['k'] * self.num_latent_proc * self.num_hyper_params

        return self.param_names

    def _update_inverses(self):
        for j in range(self.num_latent_proc):
            self.chol[j, :, :] = jitchol(self.kernels[j].K(self.Z[j, :, :], self.Z[j, :, :]))
            self.invZ[j, :, :] = inv_chol(self.chol[j, :, :])
            self.log_detZ[j] = pddet(self.chol[j, :, :])

    def _update(self):

        if Configuration.HYPER in self.config_list:
            self._update_inverses()

        self.ll = 0

        if Configuration.HYPER in self.config_list:
            self.num_hyper_params = self.kernels[0].gradient.shape[0]
            self.hyper_params = np.empty((self.num_latent_proc, self.num_hyper_params))

        if Configuration.MoG in self.config_list:
            grad_m = np.zeros((self.MoG.m_dim()))
            grad_s = np.zeros((self.MoG.get_s_size()))
            grad_pi = np.zeros((self.MoG.pi_dim()))

        if Configuration.HYPER in self.config_list:
            grad_hyper = np.zeros(self.hyper_params.shape)
            for j in range(self.num_latent_proc):
                self.hyper_params[j] = self.kernels[j].param_array[:].copy()

        if Configuration.ENTROPY in self.config_list:
            self.ll += self._l_ent()
            if Configuration.MoG in self.config_list:
                grad_m += self._d_ent_d_m()
                grad_s += self._transformed_d_ent_d_S()
                grad_pi += self._d_ent_d_pi()

        if Configuration.CROSS in self.config_list:
            xcross, xdcorss_dpi = self._cross_dcorss_dpi(0)
            self.ll += xcross
            if Configuration.MoG in self.config_list:
                grad_m += self._dcorss_dm()
                grad_s += self.transform_dcorss_dS()
                grad_pi += xdcorss_dpi
            if Configuration.HYPER in self.config_list:
                grad_hyper += self._dcross_dhyper()

        if Configuration.ELL in self.config_list:
            pX, pY = self._get_data_partition()
            xell, xdell_dm, xdell_ds, xdell_dpi, xdell_hyper = self._ell(self.n_samples, pX, pY, self.cond_likelihood)
            self.ll += xell
            if Configuration.MoG in self.config_list:
                grad_m += xdell_dm
                grad_s += self.MoG.transform_S_grad(xdell_ds)
                grad_pi += xdell_dpi
            if Configuration.HYPER in self.config_list:
                grad_hyper += xdell_hyper

        self.grad_ll = np.array([])
        if Configuration.MoG in self.config_list:
            self.grad_ll = np.hstack([grad_m.flatten(),
                                      grad_s,
                                      self.MoG.transform_pi_grad(grad_pi),
            ])

        if Configuration.HYPER in self.config_list:
            self.grad_ll = np.hstack([self.grad_ll,
                                      (grad_hyper.flatten()) * self.hyper_params.flatten()
                                      ])

    def set_configuration(self, config_list):
        self.config_list = config_list
        self._update()

    def set_params(self, p):
        """
        receives parameter from optimizer and transforms them
        :param p: input parameters
        """
        # print 'set', p
        self.last_param = p
        index = 0
        if Configuration.MoG in self.config_list:
            self.MoG.update_parameters(p[:self.MoG.num_parameters()])
            index = self.MoG.num_parameters()
        if Configuration.HYPER in self.config_list:
            self.hyper_params = np.exp(p[index:].reshape((self.num_latent_proc, self.num_hyper_params)))
            for j in range(self.num_latent_proc):
                self.kernels[j].param_array[:] = self.hyper_params[j]
        self._update()

    def get_params(self):
        """
        exposes parameters to the optimizer
        """
        params = np.array([])
        if Configuration.MoG in self.config_list:
            params = self.MoG.parameters
        if Configuration.HYPER in self.config_list:
            params = np.hstack([params, np.log(self.hyper_params.flatten())])
        return params

    def log_likelihood(self):
        return self.ll

    def _log_likelihood_gradients(self):
        return self.grad_ll

    def _get_data_partition(self):
        return self.X, self.Y

    def _A(self, j, K):
        """
        calculating A for latent process j (eq 4)
        """
        return cho_solve((self.chol[j, :, :], True), K).T

    def _Kdiag(self, p_X, K, A, j):
        """
        calculating diagonal terms of K_tilda for latent process j (eq 4)
        """
        return self.kernels[j].Kdiag(p_X) - mdiag_dot(A, K)

    def _b(self, n, j, Aj):
        """
        calculating [b_k(n)]j for latent process j (eq 19) for all k
        returns: a
        """
        return mdot(Aj[n, :], self.MoG.m[:, j, :].T)

    def _sigma(self, n, j, Kj, Aj):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        if Kj[n] < 0:
            Kj[n] = 0
        return Kj[n] + self.MoG.aSa(Aj[n, :], j)

    def dK_dtheta(self, j):
        return self.kernels[j].gradient

    # @profile
    def _get_A_K(self, p_X):
        A = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        K = np.empty((self.num_latent_proc, len(p_X)))
        Kzx = np.empty((self.num_latent_proc, self.num_inducing, p_X.shape[0]))
        for j in range(self.num_latent_proc):
            Kzx[j, :, :] = self.kernels[j].K(self.Z[j, :, :], p_X)
            A[j] = self._A(j, Kzx[j, :, :])
            K[j] = self._Kdiag(p_X, Kzx[j, :, :], A[j], j)
        return A, Kzx, K

    def _ell(self, n_sample, X, Y, cond_log_likelihood):

        """
        calculating expected log-likelihood, and it's derivatives
        :returns ell, normal ell, dell / dm, dell / ds, dell/dpi
        """

        # print 'ell started'
        total_ell = 0
        d_ell_dm = np.zeros((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        d_ell_ds = np.zeros((self.num_mog_comp, self.num_latent_proc) + self.MoG.S_dim())
        d_ell_dPi = np.zeros(self.num_mog_comp)
        if Configuration.HYPER in self.config_list:
            d_ell_d_hyper = np.zeros((self.num_latent_proc, self.num_hyper_params))
        else:
            d_ell_d_hyper = 0

        if Configuration.MoG in self.config_list or \
            (Configuration.HYPER in self.config_list and self.sparse):
            A, Kzx, K = self._get_A_K(X)
            for n in range(len(X)):
                # print 'ell for point #', n
                mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc))
                sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc))

                s_dell_dm = np.zeros((self.num_mog_comp, self.num_latent_proc))
                s_dell_ds = np.zeros((self.num_mog_comp, self.num_latent_proc))

                f = np.empty((n_sample, self.num_mog_comp, self.num_latent_proc))
                for j in range(self.num_latent_proc):
                    mean_kj[:, j] = self._b(n, j, A[j])
                    sigma_kj[:, j] = self._sigma(n, j, K[j], A[j])
                    for k in range(self.num_mog_comp):
                        f[:, k, j] = self.normal_samples[j, :] * math.sqrt(sigma_kj[k, j]) + mean_kj[k, j]

                for k in range(self.num_mog_comp):
                    cond_ll = cond_log_likelihood(f[:, k, :], Y[n, :])
                    sum_cond_ll = cond_ll.sum()
                    d_ell_dPi[k] += sum_cond_ll
                    total_ell += sum_cond_ll * self.MoG.pi[k]
                    for j in range(self.num_latent_proc):
                        if Configuration.MoG in self.config_list:
                            s_dell_dm[k, j] += np.dot(f[:, k, j] - mean_kj[k, j], cond_ll)
                            s_dell_ds[k, j] += np.dot(
                                sigma_kj[k, j] ** -2 * (f[:, k, j] - mean_kj[k, j]) ** 2 - sigma_kj[k, j] ** -1, cond_ll)

                        # for calculating hyper parameters
                        if self.sparse and Configuration.HYPER in self.config_list:
                            xn = X[np.newaxis, n, :]
                            Kxnz = Kzx[j, :, n]
                            d_sigma_d_hyper = self.dKx_dhyper(j, xn) \
                                              - self.dKzxn_dhyper_mult_x(j, xn, A[j, n]) + \
                                              2 * self.dA_dhyper_mult_x(xn, j, A[j, n],
                                                                        self.MoG.Sa(A[j, n], k, j) - Kxnz.T / 2)

                            # repeats f to aling it with the number of hyper params
                            fr = np.repeat(f[:, k, j, np.newaxis], self.num_hyper_params, axis=1)

                            tmp = 1. / sigma_kj[k, j] * d_sigma_d_hyper \
                                  - 2. * (fr - mean_kj[k, j]) / sigma_kj[k, j] * self.dA_dhyper_mult_x(xn, j, A[j, n],
                                                                                                       self.MoG.m[k, j]) \
                                  - ((fr - mean_kj[k, j]) ** 2) * sigma_kj[k, j] ** (-2) * d_sigma_d_hyper

                            d_ell_d_hyper[j] += -0.5 * self.MoG.pi[k] * np.array(
                                [np.dot(tmp[:, hp], cond_ll) for hp in range(self.num_hyper_params)]).T

                if Configuration.MoG in self.config_list:
                    for k in range(self.num_mog_comp):
                        for j in range(self.num_latent_proc):
                            Kxnz = Kzx[j, :, n]
                            d_ell_dm[k, j] += 1. / sigma_kj[k, j] * s_dell_dm[k, j] * Kxnz
                            d_ell_ds[k, j] += self.mdot_Aj(A[j, n, np.newaxis]) * s_dell_ds[k, j]

            if Configuration.MoG in self.config_list:
                for k in range(self.num_mog_comp):
                    for j in range(self.num_latent_proc):
                        d_ell_dm[k, j, :] = self.MoG.pi[k] / n_sample * cho_solve((self.chol[j, :, :], True), d_ell_dm[k, j, :])
                        d_ell_ds[k, j, :] = self.MoG.pi[k] / n_sample / 2. * d_ell_ds[k, j, :]

        d_ell_dPi = d_ell_dPi / n_sample

        return total_ell / n_sample, d_ell_dm, d_ell_ds, d_ell_dPi, d_ell_d_hyper / n_sample

    def dKzxn_dhyper_mult_x(self, j, x_n, x):
        self.kernels[j].update_gradients_full(x[:, np.newaxis], self.Z[j], x_n)
        return self.kernels[j].gradient.copy()

    def dKx_dhyper(self, j, x_n):
        self.kernels[j].update_gradients_full(np.array([[1]]), x_n)
        return self.kernels[j].gradient.copy()

    def dA_dhyper_mult_x(self, x_n, j, Ajn, x):
        w = mdot(self.invZ[j], x)[:, np.newaxis]
        self.kernels[j].update_gradients_full(w.T, x_n, self.Z[j])
        g1 = self.kernels[j].gradient.copy()
        self.kernels[j].update_gradients_full(mdot(Ajn[:, np.newaxis], w.T), self.Z[j])
        g2 = self.kernels[j].gradient.copy()
        return g1 - g2

    def _dcorss_dm(self):
        """
        calculating d corss / dm
        """
        dcdm = np.empty((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dcdm[:, j, :] = -cho_solve((self.chol[j, :, :], True), self.MoG.m[:, j, :].T).T * self.MoG.pi[:, np.newaxis]
        return dcdm

    def _dcross_ds(self):
        """
        calculating L_corss by s_k for all k's
        """
        dc_ds = np.empty((self.num_mog_comp, self.num_latent_proc, self.MoG.get_sjk_size()))
        for j in range(self.num_latent_proc):
            dc_ds[:, j] = -1. / 2 * np.array(
                [self.MoG.dAS_dS(self.chol[j, :, :], k, j) * self.MoG.pi[k] for k in range(self.num_mog_comp)])
        return dc_ds

    def transform_dcorss_dS(self):
        return self._dcross_ds().flatten()

    def _cross_dcorss_dpi(self, N):
        """
        calculating L_corss by pi_k, and also calculates the cross term
        :returns d cross / d pi, cross
        """
        cross = 0
        d_pi = np.zeros(self.num_mog_comp)
        for j in range(self.num_latent_proc):
            for k in range(self.num_mog_comp):
                d_pi[k] += \
                    N * math.log(2 * math.pi) + \
                    self.log_detZ[j] + \
                    mdot(self.MoG.m[k, j, :].T, cho_solve((self.chol[j, :, :], True), self.MoG.m[k, j, :])) + \
                    self.MoG.tr_A_mult_S(self.chol[j, :, :], k, j)
        for k in range(self.num_mog_comp):
            cross += self.MoG.pi[k] * d_pi[k]

        d_pi *= -1. / 2
        cross *= -1. / 2
        return cross, d_pi

    def _dcross_K(self, j):
        dc_dK = np.zeros((self.num_inducing, self.num_inducing))
        for k in range(self.num_mog_comp):
            dc_dK += -0.5 * self.MoG.pi[k] * (self.invZ[j]
                                              - mdot(self.invZ[j], self.MoG.mmTS(k, j), self.invZ[j])
                                              )
        return dc_dK

    def _dcross_dhyper(self):
        dc_dh = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            self.kernels[j].update_gradients_full(self._dcross_K(j), self.Z[j])
            dc_dh[j] = self.kernels[j].gradient.copy()

        return dc_dh

    def _d_ent_d_m_kj(self, k, j):
        m_k = np.zeros(self.num_inducing)
        for l in range(self.num_mog_comp):
            m_k += self.MoG.pi[k] * self.MoG.pi[l] * (np.exp(self.log_N_kl[k, l] - self.log_z[k]) +
                                                      np.exp(self.log_N_kl[k, l] - self.log_z[l])) * \
                   (self.MoG.C_m(j, k, l))
        return m_k

    def _gaussian_predict(self, t_X, normal_sigma):
        """
        predicting at test points t_X
        :param t_X: test point
        """

        # print 'ell started'
        A, Kzx, K = self._get_A_K(t_X)

        predicted_mu = np.empty((t_X.shape[0], self.num_mog_comp, self.num_latent_proc))
        predicted_var = np.empty((t_X.shape[0], self.num_mog_comp, self.num_latent_proc))
        for n in range(len(t_X)):
            mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc))

            for j in range(self.num_latent_proc):
                mean_kj[:, j] = self._b(n, j, A[j])
                sigma_kj[:, j] = self._sigma(n, j, K[j], A[j])

            predicted_mu[n, :, :] = mean_kj[:, :]
            predicted_var[n, :, :] = normal_sigma + sigma_kj[:, :]

        return predicted_mu, predicted_var

    def _gaussian_ell(self, p_X, p_Y, normal_sigma):
        normal_ell = 0
        A, Kzx, K = self._get_A_K(p_X)

        for n in range(len(p_X)):
            mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:, j] = self._b(n, j, A[j])
                sigma_kj[:, j] = self._sigma(n, j, K[j], A[j])

            for k in range(self.num_mog_comp):
                if normal_sigma is not None:
                    normal_ell += cross_ent_normal(mean_kj[k, :], np.diag(sigma_kj[k, :]), p_Y[n, :], normal_sigma) * \
                                  self.MoG.pi[k]

        return normal_ell

    def _raw_predict(self, Xnew, which_parts='all', full_cov=False, stop=False):
        if self.num_latent_proc > 1:
            raise Exception('unable to plot')

        mu, var = self._predict(Xnew)
        predicted_mu = np.average(mu, axis=1, weights=self.MoG.pi)
        predicted_var = np.average(mu ** 2, axis=1, weights=self.MoG.pi) \
                        + np.average(var, axis=1, weights=self.MoG.pi) - predicted_mu ** 2

        return predicted_mu, predicted_var
