import GPy
from atom.enum import Enum
from scipy.misc import logsumexp
from sklearn.cluster import MiniBatchKMeans, KMeans
from mog_diag import MoG_Diag
from util import mdiag_dot, jitchol, pddet, inv_chol, nearPD, cross_ent_normal, log_diag_gaussian
import math
from GPy.util.linalg import mdot
import numpy as np
from numpy.ma import trace, vstack
from scipy.linalg import logm, det, cho_solve, solve_triangular
import scipy.stats
from GPy import likelihoods
from GPy.core import Model


class Configuration(Enum):
    ENTROPY = 'ENT'
    CROSS = 'CRO'
    ELL = 'ELL'
    HYPER = 'HYP'
    MoG = 'MOG'
    LL = 'LL'


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
    def __init__(self, X, Y, num_inducing, num_mog_comp, likelihood, kernels, n_samples,
                 config_list=None, latent_noise=0, exact_ell=False, random_Z=False):

        super(SAVIGP, self).__init__("SAVIGP")
        if config_list is None:
            self.config_list = [Configuration.CROSS, Configuration.ELL, Configuration.ENTROPY]
        else:
            self.config_list = config_list
        self.num_latent_proc = len(kernels)
        self.num_mog_comp = num_mog_comp
        self.num_inducing = num_inducing
        self.MoG = self._get_mog()
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.kernels = kernels
        self.cond_likelihood = likelihood
        self.X = X
        self.Y = Y
        self.n_samples = n_samples
        self.param_names = []
        self.latent_noise = latent_noise
        self.last_param = None
        self.hyper_params = None
        self.sparse = X.shape[0] != self.num_inducing
        self.num_hyper_params = self.kernels[0].gradient.shape[0]
        self.num_like_params = self.cond_likelihood.get_num_params()
        self.is_exact_ell = exact_ell
        self.num_data_points = X.shape[0]

        self.cached_ell = None
        self.cached_ent = None
        self.cached_cross = None

        if random_Z:
            Z, init_m = self._random_inducing_points(X, Y)
        else:
            Z, init_m = self._clust_inducing_points(X, Y)

        # Z is Q * M * D
        self.Z = Z
        self.Kzz = np.array([np.empty((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.invZ = np.array([np.empty((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.chol = np.array([np.zeros((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.invZ = np.array([np.zeros((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self.log_detZ = np.zeros(self.num_latent_proc)

        self.normal_samples = np.random.normal(0, 1, self.n_samples * self.num_latent_proc * self.X.shape[0]) \
            .reshape((self.num_latent_proc, self.n_samples, self.X.shape[0]))

        self._update_latent_kernel()

        self._update_inverses()

        self.init_mog(init_m)

        self.set_configuration(self.config_list)

    def _clust_inducing_points(self, X, Y):
        Z = np.array([np.zeros((self.num_inducing, self.input_dim))] * self.num_latent_proc)
        init_m = np.empty((self.num_inducing, self.num_latent_proc))
        np.random.seed(12000)
        if self.num_inducing == X.shape[0]:
            for j in range(self.num_latent_proc):
                Z[j, :, :] = X.copy()
                init_m[:, j] = Y[:, j].copy()
        else:
            if (self.num_inducing < self.num_data_points / 10) and self.num_data_points > 10000:
                clst = MiniBatchKMeans(self.num_inducing)
            else:
                clst = KMeans(self.num_inducing)
            c = clst.fit_predict(X)
            centers = clst.cluster_centers_
            for zi in range(self.num_inducing):
                yindx = np.where(c == zi)
                if yindx[0].shape[0] == 0:
                    init_m[zi] = Y[:, :].mean()
                else:
                    init_m[zi] = np.mean(Y[yindx[0], :], axis=0)
            for j in range(self.num_latent_proc):
                Z[j, :, :] = centers.copy()

        return Z, init_m

    def _random_inducing_points(self, X, Y):
        np.random.seed(12000)
        Z = np.array([np.zeros((self.num_inducing, self.input_dim))] * self.num_latent_proc)
        init_m = np.empty((self.num_inducing, self.num_latent_proc))
        for j in range(self.num_latent_proc):
            if self.num_inducing == X.shape[0]:
                i = range(self.X.shape[0])
            else:
                i = np.random.permutation(X.shape[0])[:self.num_inducing]
            Z[j, :, :] = X[i].copy()
            init_m[:, j] = Y[i, j]

        return Z, init_m

    def _update_latent_kernel(self):
        self.kernels_latent = []
        for j in range(len(self.kernels)):
            self.kernels_latent.append(self.kernels[j] + GPy.kern.White(self.X.shape[1], variance=self.latent_noise))

    def init_mog(self, init_m):
        for j in range(self.num_latent_proc):
            self.MoG.updata_mean(j, init_m[:, j])

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

        if Configuration.LL in self.config_list:
            self.param_names += ['ll'] * self.num_like_params

        return self.param_names


    def get_all_param_names(self):
        param_names = []
        param_names += ['m'] * self.MoG.get_m_size() + ['s'] * \
                                                            self.MoG.get_s_size() + ['pi'] * self.num_mog_comp
        param_names += ['k'] * self.num_latent_proc * self.num_hyper_params
        param_names += ['ll'] * self.num_like_params

        return param_names


    def _update_inverses(self):
        for j in range(self.num_latent_proc):
            self.Kzz[j, :, :] = self.kernels_latent[j].K(self.Z[j, :, :])
            self.chol[j, :, :] = jitchol(self.Kzz[j, :, :])
            self.invZ[j, :, :] = inv_chol(self.chol[j, :, :])
            self.log_detZ[j] = pddet(self.chol[j, :, :])

    def kernel_hyp_params(self):
        hyper_params = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            hyper_params[j] = self.kernels[j].param_array[:].copy()
        return hyper_params

    def _update(self):

        if Configuration.HYPER in self.config_list:
            self._update_inverses()

        self.ll = 0

        if Configuration.MoG in self.config_list:
            grad_m = np.zeros((self.MoG.m_dim()))
            grad_s = np.zeros((self.MoG.get_s_size()))
            grad_pi = np.zeros((self.MoG.pi_dim()))

        if Configuration.HYPER in self.config_list:
            self.hyper_params = self.kernel_hyp_params()
            grad_hyper = np.zeros(self.hyper_params.shape)

        if Configuration.ENTROPY in self.config_list or (self.cached_ent is None):
            self.cached_ent = self._l_ent()
            if Configuration.MoG in self.config_list:
                grad_m += self._d_ent_d_m()
                grad_s += self._transformed_d_ent_d_S()
                grad_pi += self._d_ent_d_pi()
            if Configuration.HYPER in self.config_list:
                grad_hyper += self._dent_dhyper()
        self.ll += self.cached_ent

        if Configuration.CROSS in self.config_list or (self.cached_cross is None):
            xcross, xdcorss_dpi = self._cross_dcorss_dpi(0)
            self.cached_cross = xcross
            if Configuration.MoG in self.config_list:
                grad_m += self._dcorss_dm()
                grad_s += self.transform_dcorss_dS()
                grad_pi += xdcorss_dpi
            if Configuration.HYPER in self.config_list:
                grad_hyper += self._dcross_dhyper()

        self.ll += self.cached_cross


        if Configuration.ELL in self.config_list:
            pX, pY = self._get_data_partition()
            xell, xdell_dm, xdell_ds, xdell_dpi, xdell_hyper, xdell_dll = self._ell(self.n_samples, pX, pY, self.cond_likelihood)
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

        if Configuration.LL in self.config_list:
            self.grad_ll = np.hstack([self.grad_ll,
                                      xdell_dll
                                      ])

    def set_configuration(self, config_list):
        self.config_list = config_list
        self._clear_cache()
        self._update()

    def _clear_cache(self):
        self.cached_ell = None
        self.cached_cross = None
        self.cached_ent = None

    def set_params(self, p):
        """
        receives parameter from optimizer and transforms them
        :param p: input parameters
        """
        self.last_param = p
        index = 0
        if Configuration.MoG in self.config_list:
            self.MoG.update_parameters(p[:self.MoG.num_parameters()])
            index = self.MoG.num_parameters()
        if Configuration.HYPER in self.config_list:
            self.hyper_params = np.exp(p[index:(index + self.num_latent_proc * self.num_hyper_params)].
                                       reshape((self.num_latent_proc, self.num_hyper_params)))
            for j in range(self.num_latent_proc):
                self.kernels[j].param_array[:] = self.hyper_params[j]
            index += self.num_latent_proc * self.num_hyper_params
            self._update_latent_kernel()

        if Configuration.LL in self.config_list:
            self.cond_likelihood.set_params(p[index:index + self.num_like_params])

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
        if Configuration.LL in self.config_list:
            params = np.hstack([params, self.cond_likelihood.get_params()])
        return params

    def get_posterior_params(self):
        return self.MoG.get_m_S_params()

    def get_all_params(self):
        """
        returns all the parameters in the model
        """
        params = self.MoG.parameters
        params = np.hstack([params, np.log(self.kernel_hyp_params().flatten())])
        params = np.hstack([params, self.cond_likelihood.get_params()])
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
        return self.kernels_latent[j].Kdiag(p_X) - mdiag_dot(A, K)

    def _b_n(self, n, j, Aj, Kzx):
        """
        calculating [b_k(n)]j for latent process j (eq 19) for all k
        returns: a
        """
        return mdot(Aj[n, :], self.MoG.m[:, j, :].T)

    def _b(self, k, j, Aj, Kzx):
        """
        calculating [b_k(n)]j for latent process j (eq 19) for all k
        returns: a
        """
        return mdot(Aj, self.MoG.m[k, j, :].T)


    def _sigma_n(self, n, j, Kj, Aj, Kzx):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        if Kj[n] < 0:
            Kj[n] = 0
        return Kj[n] + self.MoG.aSa(Aj[n, :], j)


    def _sigma(self, k, j, Kj, Aj, Kzx):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        return Kj + self.MoG.aSkja(Aj, k, j)

    def dK_dtheta(self, j):
        return self.kernels_latent[j].gradient

    # @profile
    def _get_A_K(self, p_X):
        A = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        K = np.empty((self.num_latent_proc, len(p_X)))
        Kzx = np.empty((self.num_latent_proc, self.num_inducing, p_X.shape[0]))
        for j in range(self.num_latent_proc):
            Kzx[j, :, :] = self.kernels_latent[j].K(self.Z[j, :, :], p_X)
            A[j] = self._A(j, Kzx[j, :, :])
            K[j] = self._Kdiag(p_X, Kzx[j, :, :], A[j], j)
        return A, Kzx, K

    def _dell_ds(self, k, j, cond_ll, A, n_sample, sigma_kj):
        raise Exception("method not implemented")

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

        if Configuration.LL in self.config_list:
            d_ell_d_ll = np.zeros(self.num_like_params)
        else:
            d_ell_d_ll = 0

        if Configuration.MoG in self.config_list or \
            Configuration.LL in self.config_list or \
            self.cached_ell is None or \
            self.calculate_dhyper():
            A, Kzx, K = self._get_A_K(X)
            mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0]))
            sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0]))
            F = np.empty((self.n_samples, self.X.shape[0], self.num_latent_proc))
            for k in range(self.num_mog_comp):
                for j in range(self.num_latent_proc):
                    mean_kj[k,j] = self._b(k, j, A[j], Kzx[j].T)
                    sigma_kj[k,j] = self._sigma(k, j, K[j], A[j], Kzx[j].T)
                    F[:, :, j] = (self.normal_samples[j, :, :] * np.sqrt(sigma_kj[k,j]))
                    F[:, :, j] = F[:, :, j] + mean_kj[k,j]
                cond_ll, grad_ll = cond_log_likelihood.ll_F_Y(F, Y)
                for j in range(self.num_latent_proc):
                    m = self._average(cond_ll, self.normal_samples[j, :, :] / np.sqrt(sigma_kj[k,j]), True)
                    d_ell_dm[k,j] = self._proj_m_grad(j, mdot(m, Kzx[j].T)) * self.MoG.pi[k]
                    d_ell_ds[k,j] = self._dell_ds(k, j, cond_ll, A, n_sample, sigma_kj)
                    if self.calculate_dhyper():
                        ds_dhyp = self._dsigma_dhyp(j, k, A[j], Kzx, X)
                        db_dhyp = self._db_dhyp(j, k, A[j], X)
                        for h in range(self.num_hyper_params):
                            d_ell_d_hyper[j, h] += -1./2 * self.MoG.pi[k] * (
                                                self._average(cond_ll,
                                                np.ones(cond_ll.shape) / sigma_kj[k, j] * ds_dhyp[:, h] +
                                                -2. * self.normal_samples[j] / np.sqrt(sigma_kj[k,j]) * db_dhyp[:, h]
                                                - np.square(self.normal_samples[j])/sigma_kj[k, j] * ds_dhyp[:, h], True)).sum()

                sum_cond_ll = cond_ll.sum() / n_sample
                total_ell += sum_cond_ll * self.MoG.pi[k]
                d_ell_dPi[k] = sum_cond_ll

                if Configuration.LL in self.config_list:
                    d_ell_d_ll += self.MoG.pi[k] * grad_ll.sum() / n_sample

            self.cached_ell = total_ell

            total_ell = 0
            if self.is_exact_ell:
                for n in range(len(X)):
                    for k in range(self.num_mog_comp):
                            total_ell += self.cond_likelihood.ell([mean_kj[k, j, n]], [sigma_kj[k, j, n]], Y[n, :]) * self.MoG.pi[k]

        return self.cached_ell, d_ell_dm, d_ell_ds, d_ell_dPi, d_ell_d_hyper, d_ell_d_ll

    def _average(self, condll, X, variance_reduction):
        if variance_reduction:
            X = X.T
            condll = condll.T
            cvsamples = self.n_samples / 10
            pz = X[:, 0:cvsamples]
            py = np.multiply(condll[:, 0:cvsamples], pz)
            above = np.multiply((py.T-py.mean(1)), pz.T).sum(axis=0)/(cvsamples-1)
            below = np.square(pz).sum(axis=1)/(cvsamples-1)
            cvopt = np.divide(above, below)
            cvopt = np.nan_to_num(cvopt)

            grads = np.multiply(condll, X) - np.multiply(cvopt, X.T).T
        else:
            grads = np.multiply(condll, X)
        return grads.mean(axis=1)


    def calculate_dhyper(self):
        return self.sparse and Configuration.HYPER in self.config_list

    def _proj_m_grad(self, j, dl_dm):
        return cho_solve((self.chol[j, :, :], True), dl_dm)

    def _dsigma_n_dhyp(self, j, k, A, Kxnz, n, xn):
        return self.dKx_dhyper(j, xn) \
               - self.dKzxn_dhyper_mult_x(j, xn, A[j, n]) + \
               2 * self.dA_dhyper_n_mult_x(xn, j, A[j, n],
                                         self.MoG.Sa(A[j, n], k, j) - Kxnz.T / 2)

    def _dsigma_dhyp(self, j, k, Aj, Kzx, X):
        return self.kernels[j].get_gradients_Kdiagn(X) \
               - self.kernels[j].get_gradients_Kn(Aj, X, self.Z[j]) + \
               2 * self.dA_dhyper_mult_x(j, X, Aj,
                                         self.MoG.Sa(Aj.T, k, j) - Kzx[j] / 2)

    def _db_n_dhyp(self, j, k, A, n, xn):
        return self.dA_dhyper_n_mult_x(xn, j, A[j, n], self.MoG.m[k, j])

    def _db_dhyp(self, j, k, Aj, X):
        return self.dA_dhyper_mult_x(j, X, Aj, np.repeat(self.MoG.m[k,j][:, np.newaxis], X.shape[0], axis=1))


    def dKzxn_dhyper_mult_x(self, j, x_n, x):
        self.kernels[j].update_gradients_full(x[:, np.newaxis], self.Z[j], x_n)
        return self.kernels[j].gradient.copy()

    def dKx_dhyper(self, j, x_n):
        self.kernels[j].update_gradients_full(np.array([[1]]), x_n)
        return self.kernels[j].gradient.copy()

    def dA_dhyper_n_mult_x(self, x_n, j, Ajn, x):
        w = mdot(self.invZ[j], x)[:, np.newaxis]
        self.kernels[j].update_gradients_full(w.T, x_n, self.Z[j])
        g1 = self.kernels[j].gradient.copy()
        self.kernels[j].update_gradients_full(mdot(Ajn[:, np.newaxis], w.T), self.Z[j])
        g2 = self.kernels[j].gradient.copy()
        return g1 - g2

    def dA_dhyper_mult_x(self, j, X, Aj, m):
        w = mdot(self.invZ[j], m)
        return self.kernels[j].get_gradients_Kn(w.T, X, self.Z[j]) - \
            self.kernels[j].get_gradients_Kzz(Aj, w, self.Z[j])


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
                [self.MoG.dAinvS_dS(self.chol[j, :, :], k, j) * self.MoG.pi[k] for k in range(self.num_mog_comp)])
        return dc_ds

    def transform_dcorss_dS(self):
        return self._dcross_ds().flatten()

    # def _cross_dcorss_dpi(self, N):
    #     """
    #     calculating L_corss by pi_k, and also calculates the cross term
    #     :returns d cross / d pi, cross
    #     """
    #     cross = 0
    #     d_pi = np.zeros(self.num_mog_comp)
    #     for j in range(self.num_latent_proc):
    #         for k in range(self.num_mog_comp):
    #             d_pi[k] += \
    #                 N * math.log(2 * math.pi) + \
    #                 self.log_detZ[j] + \
    #                 mdot(self.MoG.m[k, j, :].T, cho_solve((self.chol[j, :, :], True), self.MoG.m[k, j, :])) + \
    #                 self.MoG.tr_A_mult_S(self.chol[j, :, :], k, j)
    #     for k in range(self.num_mog_comp):
    #         cross += self.MoG.pi[k] * d_pi[k]
    #
    #     d_pi *= -1. / 2
    #     cross *= -1. / 2
    #     return cross, d_pi

    def _cross_dcorss_dpi(self, N):
        """
        calculating L_corss by pi_k, and also calculates the cross term
        :returns d cross / d pi, cross
        """
        cross = 0
        d_pi = np.zeros(self.num_mog_comp)
        for j in range(self.num_latent_proc):
            for k in range(self.num_mog_comp):
                a = solve_triangular(self.chol[j, :, :], self.MoG.m[k, j, :], lower=True)
                d_pi[k] += \
                    N * math.log(2 * math.pi) + \
                    self.log_detZ[j] + \
                     + np.dot(a, a.T) + \
                    self.MoG.tr_Ainv_mult_S(self.chol[j, :, :], k, j)
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
            self.kernels_latent[j].update_gradients_full(self._dcross_K(j), self.Z[j])
            dc_dh[j] = self.kernels[j].gradient.copy()

        return dc_dh

    def _dent_dhyper(self):
        return np.zeros((self.num_latent_proc, self.num_hyper_params))

    def _d_ent_d_m_kj(self, k, j):
        m_k = np.zeros(self.num_inducing)
        for l in range(self.num_mog_comp):
            m_k += self.MoG.pi[k] * self.MoG.pi[l] * (np.exp(self.log_N_kl[k, l] - self.log_z[k]) +
                                                      np.exp(self.log_N_kl[k, l] - self.log_z[l])) * \
                   (self.MoG.C_m(j, k, l))
        return m_k

    def _predict_comp(self, Xs, Ys):
        """
        predicting at test points t_X
        :param Xs: test point
        """

        # print 'ell started'
        A, Kzx, K = self._get_A_K(Xs)

        predicted_mu = np.empty((Xs.shape[0], self.num_mog_comp, self.output_dim))
        predicted_var = np.empty((Xs.shape[0], self.num_mog_comp, self.output_dim))
        nlpd = None
        if not (Ys is None):
            nlpd = np.empty((Xs.shape[0], self.num_mog_comp))

        mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0]))
        sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0]))
        for k in range(self.num_mog_comp):
            for j in range(self.num_latent_proc):
                mean_kj[k,j] = self._b(k, j, A[j], Kzx[j].T)
                sigma_kj[k,j] = self._sigma(k, j, K[j], A[j], Kzx[j].T)

            if not (Ys is None):
                predicted_mu[:, k, :], predicted_var[:, k, :], nlpd[:, k] = \
                    self.cond_likelihood.predict(mean_kj[k, :].T, sigma_kj[k, :].T, Ys, self)
            else:
                predicted_mu[:, k, :], predicted_var[:, k, :], _ = \
                    self.cond_likelihood.predict(mean_kj[k, :].T, sigma_kj[k, :].T, Ys, self)

        return predicted_mu, predicted_var, -logsumexp(nlpd, 1, self.MoG.pi)

    def predict(self, Xs, Ys=None):
        mu, var, nlpd = self._predict_comp(Xs, Ys)
        predicted_mu = np.average(mu, axis=1, weights=self.MoG.pi)
        predicted_var = np.average(mu ** 2, axis=1, weights=self.MoG.pi) \
                        + np.average(var, axis=1, weights=self.MoG.pi) - predicted_mu ** 2

        return predicted_mu, predicted_var, nlpd[:, np.newaxis]
