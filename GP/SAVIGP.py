from scipy.misc import logsumexp
from Mog_diag import MoG_Diag
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
    ETNROPY = 1
    CROSS = 2
    ELL = 3
    HYPER = 4
    MoG = 5


class SAVIGP(Model):

    """
    Scalable Variational Inference Gaussian Process

    :param X: input observations
    :param Y: outputs
    :param num_inducing: number of inducing variables
    :param num_MoG_comp: number of components of the MoG
    :param num_latent_proc: number of latent processes
    :param likelihood: conditional likelihood function
    :param kernel: list of kernels of the HP
    :param n_samples: number of samples drawn for approximating ell and its gradient
    :rtype: model object
    """

    def __init__(self, X, Y, num_inducing, num_MoG_comp, likelihood, kernels, n_samples, config_list = None):

        super(SAVIGP, self).__init__("SAVIGP")
        if config_list is None:
            self.config_list = [Configuration.CROSS, Configuration.ELL, Configuration.ETNROPY, Configuration.HYPER]
        else:
            self.config_list = config_list
        self.num_latent_proc = len(kernels)
        self.num_MoG_comp = num_MoG_comp
        self.num_inducing = num_inducing
        self.MoG = self._get_MoG()
        self.input_dim = X[0].shape[0]
        self.output_dim = Y[0].shape[0]
        self.kernels = kernels
        self.cond_likelihood = likelihood
        self.X = X
        self.Y = Y
        self.n_samples = n_samples

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

        self._update_inverses()

        for j in range(self.num_latent_proc):
            K = self.kernels[j].K(self.Z[j], self.Z[j])
            self.MoG.update_covariance(j, K)

        self._update()

    def rand_init_MoG(self):
        self.MoG.random_init()

    def _get_MoG(self):
        return MoG_Diag(self.num_MoG_comp, self.num_latent_proc, self.num_inducing)

    def _get_param_names(self):
        self.param_names = []
        if Configuration.MoG in self.config_list:
            self.param_names += ['m'] * self.MoG.get_m_size() + ['s'] * self.MoG.get_s_size() + ['pi'] * self.num_MoG_comp

        if Configuration.HYPER in self.config_list:
            self.param_names += ['k'] * self.num_latent_proc * self.num_hyper_params

        return self.param_names

    def _update_inverses(self):
        for j in range(self.num_latent_proc):
            self.chol[j,:,:] = jitchol(self.kernels[j].K(self.Z[j,:,:], self.Z[j,:,:]))
            self.invZ[j, :, :] = inv_chol(self.chol[j,:,:])
            self.log_detZ[j] = pddet(self.chol[j,:,:])

    def update_N_z(self):
        self.log_z = np.zeros((self.num_MoG_comp))
        self.log_N_kl = np.zeros((self.num_MoG_comp, self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
            for l in range(self.num_MoG_comp):
                    for j in range(self.num_latent_proc):
                        self.log_N_kl[k, l] += log_diag_gaussian(self.MoG.m[k,j], self.MoG.m[l,j], self.MoG.s[k,j] + self.MoG.s[l,j])
            self.log_z[k]=logsumexp(self.log_N_kl[k, :] + np.log(self.MoG.pi))

    def _update(self):

        if Configuration.HYPER in self.config_list:
            self._update_inverses()

        self.update_N_z()

        self.ll = 0

        if Configuration.HYPER in self.config_list:
            self.num_hyper_params = self.kernels[0].gradient.shape[0]
            self.hyper_params = np.empty((self.num_latent_proc, self.num_hyper_params))

        if Configuration.MoG in self.config_list:
            grad_m = np.zeros((self.MoG.m_dim()))
            grad_s = np.zeros((self.MoG.get_s_size()))
            grad_pi= np.zeros((self.MoG.pi_dim()))

        if Configuration.HYPER in self.config_list:
            grad_hyper = np.zeros(self.hyper_params.shape)
            for j in range(self.num_latent_proc):
                self.hyper_params[j] = self.kernels[j].param_array[:].copy()

        if Configuration.ETNROPY in self.config_list:
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
                grad_hyper += self._dcross_d_hyper().flatten()

        if Configuration.ELL in self.config_list:
            pX, pY = self._get_data_partition()
            xell, xdell_dm, xdell_dS, xdell_dpi, xdell_hyper = self._ell(self.n_samples, pX, pY, self.cond_likelihood)
            self.ll += xell
            if Configuration.MoG in self.config_list:
                grad_m += xdell_dm
                grad_s += self.MoG.transform_S_grad(xdell_dS)
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

    def _set_params(self, p):
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

    def _get_params(self):
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

    def _A(self, p_X, j):
        """
        calculating A for latent process j (eq 4)
        """
        # return mdot(self.kernels[j].K(p_X, self.Z[j,:,:]), self.invZ[j,:,:])
        return cho_solve((self.chol[j,:,:], True), self.kernels[j].K(p_X, self.Z[j,:,:]).T).T

    def _Kdiag(self, p_X, A, j):
        """
        calculating diagonal terms of K_tilda for latent process j (eq 4)
        """
        return self.kernels[j].Kdiag(p_X) - mdiag_dot(A, self.kernels[j].K(self.Z[j,:,:], p_X))

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
        # print Kj[n], self.MoG.aSa(Aj[n, :], j)
        return Kj[n] + self.MoG.aSa(Aj[n, :], j)

    def dK_dtheta(self, j):
        return self.kernels[j].gradient

    # @profile
    def _ell(self, n_sample, p_X, p_Y, cond_log_likelihood):

        """
        calculating expected log-likelihood, and it's derivatives
        :returns ell, normal ell, dell / dm, dell / ds, dell/dpi
        """

        # print 'ell started'
        total_ell = 0
        d_ell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        d_ell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc) + self.MoG.S_dim())
        d_ell_dPi = np.zeros(self.num_MoG_comp)
        if Configuration.HYPER in self.config_list:
            d_ell_d_hyper = np.zeros((self.num_latent_proc, self.num_hyper_params))
        else:
            d_ell_d_hyper = 0
        Aj = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(p_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(p_X, j)
            Kj[j] = self._Kdiag(p_X, Aj[j], j)

        for n in  range(len(p_X)):
            # print 'ell for point #', n
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))

            s_dell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc))
            s_dell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc))

            f = np.empty((n_sample, self.num_MoG_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

                # sigma_kj[:,j] = self.MoG.s[:,j,n]
                for k in range(self.num_MoG_comp):
                    self.normal_samples = np.random.normal(0, 1, self.n_samples)
                    f[:,k, j] = self.normal_samples * math.sqrt(sigma_kj[k,j]) + mean_kj[k,j]
                    # f[:,k, j] = np.random.normal(mean_kj[k,j], math.sqrt(sigma_kj[k,j]), self.n_samples)

            for k in range(self.num_MoG_comp):
                cond_ll = cond_log_likelihood(f[:,k,:], p_Y[n, :])
                sum_cond_ll = cond_ll.sum()
                d_ell_dPi[k] += sum_cond_ll
                total_ell += sum_cond_ll * self.MoG.pi[k]
                for j in range(self.num_latent_proc):
                    s_dell_dm[k,j] += np.dot(f[:,k,j] - mean_kj[k,j], cond_ll)
                    s_dell_dS[k,j] += np.dot(sigma_kj[k,j] ** -2 * (f[:,k,j] - mean_kj[k,j]) ** 2 - sigma_kj[k,j] ** -1, cond_ll)

                    # for calculating hyper parameters
                    if Configuration.HYPER in self.config_list:
                        xn = p_X[np.newaxis, n, :]
                        K_xn_Zj = self.kernels[j].K(xn, self.Z[j,:,:])[0,:]
                        d_sigma_d_hyper = self.d_K_xn_d_hyper(j, xn) - self.d_Ajn_d_hyper_mult_x(xn, j, Aj[j,n], K_xn_Zj.T) \
                                        - self.d_K_zjxn_d_hyper_mult_x(j, xn, Aj[j,n]) + \
                                          2 * self.d_Ajn_d_hyper_mult_x(xn, j, Aj[j,n], self.MoG.Sa(Aj[j,n], k, j))

                        # repeats f to aling it with the number of hyper params
                        fr = np.repeat(f[:,k,j, np.newaxis], self.num_hyper_params, axis=1)

                        tmp = 1. / sigma_kj[k,j] * d_sigma_d_hyper  \
                             - 2. * (fr - mean_kj[k,j])  / sigma_kj[k,j] * self.d_Ajn_d_hyper_mult_x(xn, j, Aj[j,n], self.MoG.m[k,j]) \
                            - ((fr - mean_kj[k,j]) ** 2) * sigma_kj[k,j] ** (-2) * d_sigma_d_hyper

                        d_ell_d_hyper[j] += -0.5 * self.MoG.pi[k] * np.array([np.dot(tmp[:,hp], cond_ll) for hp in range(self.num_hyper_params)]).T

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    xn = p_X[np.newaxis, n, :]
                    K_xn_Zj = self.kernels[j].K(xn, self.Z[j,:,:])[0,:]
                    d_ell_dm[k,j] += 1./sigma_kj[k,j] * s_dell_dm[k,j] * K_xn_Zj
                    d_ell_dS[k,j] += self.mdot_Aj(Aj[j,n, np.newaxis])* s_dell_dS[k,j]

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dm[k,j,:] = self.MoG.pi[k]/n_sample * cho_solve((self.chol[j,:,:], True), d_ell_dm[k,j,:])
                d_ell_dS[k,j,:] = self.MoG.pi[k]/n_sample / 2. * d_ell_dS[k,j,:]

        d_ell_dPi = d_ell_dPi/n_sample

        return total_ell / n_sample, d_ell_dm, d_ell_dS, d_ell_dPi, d_ell_d_hyper/n_sample

    def d_K_zjxn_d_hyper_mult_x(self, j, x_n, x):
        self.kernels[j].update_gradients_full(x[:,np.newaxis], self.Z[j], x_n)
        return self.kernels[j].gradient.copy()

    def d_K_xn_d_hyper(self, j, x_n):
        self.kernels[j].update_gradients_full(np.array([[1]]), x_n)
        return self.kernels[j].gradient.copy()

    def d_Ajn_d_hyper_mult_x(self, x_n, j, Ajn, x):
        w = mdot(self.invZ[j], x)[:,np.newaxis]
        self.kernels[j].update_gradients_full(w.T, x_n, self.Z[j])
        g1 = self.kernels[j].gradient.copy()
        self.kernels[j].update_gradients_full(mdot(Ajn[:,np.newaxis], w.T), self.Z[j])
        g2 = self.kernels[j].gradient.copy()
        return g1 - g2


    def mdot_Aj(self,Ajn):
        return (Ajn[0] * Ajn[0])

    def _dcorss_dm(self):
        """
        calculating d corss / dm
        """
        dcdm = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dcdm[:,j,:] = -cho_solve((self.chol[j,:,:], True), self.MoG.m[:,j,:].T).T * self.MoG.pi[:, np.newaxis]
        return dcdm

    def _dcross_dS(self):
        """
        calculating L_corss by s_k for all k's
        """
        dc_ds = np.empty((self.num_MoG_comp, self.num_latent_proc, self.MoG.get_sjk_size()))
        for j in range(self.num_latent_proc):
            dc_ds[:,j] = -1. /2 * np.array([self.MoG.dAS_dS(self.chol[j,:,:], k, j) * self.MoG.pi[k] for k in range(self.num_MoG_comp)])
        return dc_ds

    def transform_dcorss_dS(self):
        return self._dcross_dS().flatten()

    def _cross_dcorss_dpi(self, N):
        """
        calculating L_corss by pi_k, and also calculates the cross term
        :returns d cross / d pi, cross
        """
        cross = 0
        d_pi = np.zeros(self.num_MoG_comp)
        for j in range(self.num_latent_proc):
            for k in range(self.num_MoG_comp):
                d_pi[k] +=  \
                        N * math.log(2 * math.pi) + \
                        self.log_detZ[j] + \
                        mdot(self.MoG.m[k, j, :].T, cho_solve((self.chol[j,:,:], True), self.MoG.m[k, j, :])) + \
                        self.MoG.tr_A_mult_S(self.chol[j,:,:], k, j)
        for k in range(self.num_MoG_comp):
            cross += self.MoG.pi[k] * d_pi[k]

        d_pi *= -1. / 2
        cross *= -1. / 2
        return cross, d_pi

    def _dcross_K_j(self, j):
        dc_dK = np.zeros((self.num_inducing, self.num_inducing))
        for k in range(self.num_MoG_comp):
            dc_dK += -0.5 * self.MoG.pi[k] * (self.invZ[j]
                     -mdot(self.invZ[j], self.MoG.mmTS(k,j), self.invZ[j])
            )
        return dc_dK

    def _dcross_d_hyper(self):
        dc_dh = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            self.kernels[j].update_gradients_full(self._dcross_K_j(j), self.Z[j])
            dc_dh[j] = self.kernels[j].gradient.copy()

        return dc_dh

    def _d_ent_d_m_kj(self, k, j):
        m_k = np.zeros(self.num_inducing)
        for l in range(self.num_MoG_comp):
            m_k += self.MoG.pi[k] * self.MoG.pi[l] * (np.exp(self.log_N_kl[l, k] - self.log_z[k]) +
                                                      np.exp(self.log_N_kl[l, k] - self.log_z[l])) * \
                    (self.MoG.C_m(j, k, l))
        return m_k

    def _d_ent_d_m(self):
        dent_dm = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                dent_dm[k,j,:] = self._d_ent_d_m_kj(k,j)
        return dent_dm

    def _d_ent_d_pi(self):
        pi = np.empty(self.num_MoG_comp)
        for k in range(self.num_MoG_comp):
            pi[k] = -self.log_z[k]
            for l in range(self.num_MoG_comp):
                pi[k] -= self.MoG.pi[l] * (np.exp(self.log_N_kl[l, k] - self.log_z[k]))
        return pi

    def _d_ent_d_S_kj(self, k, j):
        s_k = np.zeros(self.MoG.S_dim())
        for l in range(self.num_MoG_comp):
            s_k += self.MoG.pi[k] * self.MoG.pi[l] * (np.exp(self.log_N_kl[l, k] - self.log_z[k]) +
                                                      np.exp(self.log_N_kl[l, k] - self.log_z[l])) * \
                   self.MoG.C_m_C(j, k, l)
        return 1./2 * s_k

    def _d_ent_d_S(self):
        dent_ds = np.empty((self.num_MoG_comp, self.num_latent_proc) + self.MoG.S_dim())
        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                dent_ds[k,j] = self._d_ent_d_S_kj(k,j)
        return dent_ds

    def _transformed_d_ent_d_S(self):
        return (self._d_ent_d_S()).flatten()

    def _l_ent(self):
        return -np.dot(self.MoG.pi,  self.log_z)

    def _gaussian_ell(self, p_X, p_Y, normal_sigma):
        normal_ell = 0
        Aj = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(p_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(p_X, j)
            Kj[j] = self._Kdiag(p_X, Aj[j], j)

        for n in  range(len(p_X)):
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

            for k in range(self.num_MoG_comp):
                if normal_sigma is not None:
                    normal_ell += cross_ent_normal(mean_kj[k,:], np.diag(sigma_kj[k,:]), p_Y[n, :], normal_sigma) * self.MoG.pi[k]

        return normal_ell


    def _gaussian_predict(self, t_X, normal_sigma):
        """
        predicting at test points t_X
        :param t_X: test point
        """

        # print 'ell started'
        Aj = np.empty((self.num_latent_proc, len(t_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(t_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(t_X, j)
            Kj[j] = self._Kdiag(t_X, Aj[j], j)

        predicted_mu = np.empty((t_X.shape[0], self.num_MoG_comp, self.num_latent_proc))
        predicted_var = np.empty((t_X.shape[0], self.num_MoG_comp, self.num_latent_proc))
        for n in range(len(t_X)):
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))

            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

            predicted_mu[n,:, :] = mean_kj[:,:]
            predicted_var[n, :, :] = normal_sigma + sigma_kj[:,:]

        return predicted_mu, predicted_var

    def _raw_predict(self, Xnew, which_parts='all', full_cov=False, stop=False):
        if self.num_latent_proc > 1:
            raise Exception('unable to plot')

        mu, var = self._predict(Xnew)
        predicted_mu = np.average(mu, axis=1, weights=self.MoG.pi)
        predicted_var = np.average(mu ** 2, axis=1, weights=self.MoG.pi) \
            + np.average(var, axis=1, weights=self.MoG.pi) - predicted_mu ** 2

        return predicted_mu, predicted_var
