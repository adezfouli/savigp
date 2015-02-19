import math
from GPy.util.linalg import mdot
import numpy as np
from numpy.ma import trace, vstack
import scipy.stats
from GPy import likelihoods
from GPy.core import Model
from GPy.core.gp import GP
from MoG_Diag import MoG_Diag
from util import mdiag_dot, jitchol, pddet, inv_chol


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

    def __init__(self, X, Y, num_inducing, num_MoG_comp, likelihood, kernels, n_samples):

        super(SAVIGP, self).__init__("SAVIGP")
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
        self.num_hyper_params = kernels[0].gradient.shape[0]

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
        self.log_detZ = np.array([0] * self.num_latent_proc)

        for j in range(self.num_latent_proc):
            L = jitchol(self.kernels[j].K(self.Z[j,:,:], self.Z[j,:,:]))
            self.invZ[j, :, :], self.log_detZ[j] = (inv_chol(L), pddet(L))

        self._update()

    def _get_MoG(self):
        return MoG_Diag(self.num_MoG_comp, self.num_latent_proc, self.num_inducing)

    def _get_param_names(self):
        return ['m'] * self.MoG.get_m_size() + ['s'] * self.MoG.get_s_size() + ['pi'] * self.num_MoG_comp \
                + ['k'] * self.num_latent_proc * self.num_hyper_params

    def _update(self):
        """
        updating internal variables for later use
        """
        #updating N_lk, and z_k used for updating d_ent
        self.N_kll = np.ones((self.num_MoG_comp, self.num_MoG_comp, self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
            for l1 in range(self.num_MoG_comp):
                for l2 in range(self.num_MoG_comp):
                    for j in range(self.num_latent_proc):
                        self.N_kll[k,l1, l2] *= self.MoG.ratio(j, k, l1, l2)

        self.N_kl_z_k = np.zeros((self.num_MoG_comp, self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
            for l in range(self.num_MoG_comp):
                for x in range(self.num_MoG_comp):
                    self.N_kl_z_k[k,l] += self.MoG.pi[x] * self.N_kll[k,x,l]
                self.N_kl_z_k[k,l] = 1.0 / self.N_kl_z_k[k,l]


        self.N_k0 = np.zeros((self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                l = 0
                self.N_k0[k] += self.MoG.log_pdf(j, k, l)

        self.log_z = np.zeros((self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
                self.log_z[k] = -math.log(self.N_kl_z_k[k,0]) + self.N_k0[k]

        #calculating ll and gradients for future uses
        pX, pY = self._get_data_partition()

        xell, xdell_dm, xdell_dS, xdell_dpi, xdell_hyper = self._ell(self.n_samples, pX, pY, self.cond_likelihood)
        xcross, xdcorss_dpi = self._cross_dcorss_dpi(0)
        self.ll = xell + xcross + self._l_ent()

        self.grad_ll = np.hstack([xdell_dm.flatten() + self._dcorss_dm().flatten() + self._d_ent_d_m().flatten(),
                                 self.MoG.transform_S_grad(xdell_dS + self._dcross_dS() + self._d_ent_d_S()),
                                 self.MoG.transform_pi_grad(xdell_dpi + xdcorss_dpi + self._d_ent_d_pi())
                                   ,xdell_hyper.flatten()
        ])

    def _set_params(self, p):
        """
        receives parameter from optimizer and transforms them
        :param p: input parameters
        """
        # print 'set', p
        self.last_param = p
        self.MoG.update_parameters(p[:self.MoG.num_parameters()])
        self.hyper_params = p[self.MoG.num_parameters():].reshape((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            self.kernels[j].param_array[:] = self.hyper_params[j]
        self._update()

    def _get_params(self):
        """
        exposes parameters to the optimizer
        """
        hyper = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            hyper[j] = self.kernels[j].param_array
        return np.hstack([self.MoG.parameters, hyper.flatten()])

    def log_likelihood(self):
        # print 'll', self.ll
        return self.ll

    def _log_likelihood_gradients(self):
        # print 'll grad', self.grad_ll
        return self.grad_ll

    def _get_data_partition(self):
        return self.X, self.Y

    def _A(self, p_X, j):
        """
        calculating A for latent process j (eq 4)
        """
        return mdot(self.kernels[j].K(p_X, self.Z[j,:,:]), self.invZ[j,:,:])

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
        return Kj[n] + self.MoG.aSa(Aj[n, :], j)

    def dK_dtheta(self, j):
        return self.kernels[j].gradient

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
        d_ell_d_hyper = np.zeros((self.num_latent_proc, self.num_hyper_params))
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
                for k in range(self.num_MoG_comp):
                    f[:,k, j] = np.random.normal(mean_kj[k,j], math.sqrt(sigma_kj[k,j]), n_sample)

            for k in range(self.num_MoG_comp):
                cond_ll = cond_log_likelihood(f[:,k,:], p_Y[n, :])
                d_ell_dPi[k] += sum(cond_ll)
                total_ell += sum(cond_ll) * self.MoG.pi[k]
                for j in range(self.num_latent_proc):
                    s_dell_dm[k,j] += np.dot(f[:,k,j] - mean_kj[k,j], cond_ll)
                    s_dell_dS[k,j] += np.dot(sigma_kj[k,j] ** -2 * (f[:,k,j] - mean_kj[k,j]) ** 2 - sigma_kj[k,j] ** -1, cond_ll)

                    # for calculating hyper parameters
                    xn = p_X[np.newaxis, n, :]
                    K_xn_Zj = self.kernels[j].K(xn, self.Z[j,:,:])[0,:]
                    d_sigma_d_hyper = self.d_K_xn_d_hyper(j, xn) - self.d_Ajn_d_hyper_mult_x(xn, j, Aj[j,n], K_xn_Zj.T) \
                                    - self.d_K_zjxn_d_hyper_mult_x(j, xn, Aj[j,n]) + \
                                      2 * self.d_Ajn_d_hyper_mult_x(xn, j, Aj[j,n], self.MoG.Sa(Aj[j,n], k, j))

                    # repeats f to aling it with the number of hyper params
                    fr = np.repeat(f[:,k,j, np.newaxis], self.num_hyper_params, axis=1)

                    tmp = 1. / sigma_kj[k,j] * d_sigma_d_hyper  \
                         - 2 * (fr - mean_kj[k,j]) \
                           / sigma_kj[k,j] * self.d_Ajn_d_hyper_mult_x(xn, j, Aj[j,n], self.MoG.m[k,j]) \
                        - (fr - mean_kj[k,j]) ** 2 * sigma_kj[k,j] ** (-2) * d_sigma_d_hyper

                    d_ell_d_hyper[j] += -0.5 * self.MoG.pi[k] * np.array([np.dot(tmp[:,hp], cond_ll) for hp in range(self.num_hyper_params)]).T

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    xn = p_X[np.newaxis, n, :]
                    K_xn_Zj = self.kernels[j].K(xn, self.Z[j,:,:])[0,:]
                    d_ell_dm[k,j] += 1./sigma_kj[k,j] * s_dell_dm[k,j] * K_xn_Zj
                    d_ell_dS[k,j] += self.mdot_Aj(Aj[j,n, np.newaxis])* s_dell_dS[k,j]

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dm[k,j,:] = self.MoG.pi[k]/n_sample * mdot(self.invZ[j,:,:], d_ell_dm[k,j,:])
                d_ell_dS[k,j,:] = self.MoG.pi[k]/n_sample / 2. * d_ell_dS[k,j,:]

        d_ell_dPi = d_ell_dPi/n_sample

        return total_ell / n_sample, d_ell_dm, d_ell_dS, d_ell_dPi, d_ell_d_hyper/ n_sample

    def d_K_zjxn_d_hyper_mult_x(self, j, x_n, x):
        self.kernels[j].update_gradients_full(x[:,np.newaxis], self.Z[j], x_n)
        return self.kernels[j].gradient.copy()

    def d_K_xn_d_hyper(self, j, x_n):
        self.kernels[j].update_gradients_full(np.array([[1]]), x_n)
        return self.kernels[j].gradient.copy()

    def d_Ajn_d_hyper_mult_x(self, x_n, j, Ajn, x):
        w = mdot(self.invZ[j], x)[:,np.newaxis]
        self.kernels[j].update_gradients_full(w, x_n, self.Z[j])
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
            dcdm[:,j,:] = -mdot(self.MoG.m[:,j,:], self.invZ[j,:,:]) * self.MoG.pi[:, np.newaxis]
        return dcdm

    def _dcross_dS(self):
        """
        calculating L_corss by s_k for all k's
        """
        dc_ds = np.empty((self.num_MoG_comp, self.num_latent_proc) + self.MoG.S_dim())
        for j in range(self.num_latent_proc):
            dc_ds[:,j] = -1. /2 * np.array([self.MoG.dAS_dS(self.invZ[j,:,:]) * self.MoG.pi[k] for k in range(self.num_MoG_comp)])
        return dc_ds

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
                        mdot(self.MoG.m[k, j, :].T, self.invZ[j,:,:], self.MoG.m[k, j, :]) + \
                        self.MoG.tr_A_mult_S(self.invZ[j,:,:], k, j)
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

    def _dcross_dtheta(self):
        for j in range(self.num_latent_proc):
            self.kernels[j].update_gradients_full(self._dcross_K_j(j), self.Z[j])
            print self.kernels[j].gradient

    def _d_ent_d_m_kj(self, k, j):
        m_k = np.zeros(self.num_inducing)
        for l in range(self.num_MoG_comp):
            m_k += self.MoG.pi[k] * self.MoG.pi[l] * (self.N_kl_z_k[k,l] + self.N_kl_z_k[l, k]) * \
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
                pi[k] -= self.MoG.pi[l] * self.N_kl_z_k[l,k]
        return pi

    def _d_ent_d_S_kj(self, k, j):
        s_k = np.zeros(self.MoG.S_dim())
        for l in range(self.num_MoG_comp):
            s_k += self.MoG.pi[k] * self.MoG.pi[l] * (self.N_kl_z_k[k,l] + self.N_kl_z_k[l, k]) * \
                   self.MoG.C_m_C(j, k, l)
        return 1./2 * s_k

    def _d_ent_d_S(self):
        dent_ds = np.empty((self.num_MoG_comp, self.num_latent_proc) + self.MoG.S_dim())
        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                dent_ds[k,j] = self._d_ent_d_S_kj(k,j)
        return dent_ds

    def _l_ent(self):
        return -np.dot(self.MoG.pi,  self.log_z)






