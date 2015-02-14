import math
from GPy.util.linalg import mdot
import numpy as np
from numpy.ma import trace
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
    :param kernel: of the GP
    :param n_samples: number of samples drawn for approximating ell and its gradient
    :rtype: model object
    """

    def __init__(self, X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, kernel, n_samples, normalize_X):
        super(SAVIGP, self).__init__("SAVIGP")

        self.MoG = MoG_Diag(num_MoG_comp, num_latent_proc, num_inducing)
        self.input_dim = X[0].shape[0]
        self.output_dim = Y[0].shape[0]
        self.num_inducing = num_inducing
        self.num_latent_proc = num_latent_proc
        self.num_MoG_comp = num_MoG_comp
        self.kernel = kernel
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
        self.log_detZ = np.array([0] * self.num_latent_proc)

        for j in range(self.num_latent_proc):
            L = jitchol(self.kernel.K(self.Z[j,:,:], self.Z[j,:,:]))
            self.invZ[j, :, :], self.log_detZ[j] = (inv_chol(L), pddet(L))

        self._update()

    def _get_param_names(self):
        return ['m'] * self.MoG.n + ['s'] * self.MoG.n + ['pi'] * self.num_MoG_comp



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

        self.MoG.update()

        #calculating ll and gradients for future uses
        pX, pY = self._get_data_partition()

        #calculating gradient of pi wrt to the parameters
        pi = np.repeat(np.array([self.MoG.pi.T]), self.num_MoG_comp, 0)
        dpi_dx = pi * (-pi.T + np.eye(self.num_MoG_comp))

        xell, xdell_dm, xdell_dS, xdell_dpi = self._ell(self.n_samples, pX, pY, self.cond_likelihood)
        xcross, xdcorss_dpi = self._cross_dcorss_dpi(0)
        self.ll = xell + xcross + self._l_ent()

        ds_dx = self.MoG.s.flatten()
        self.grad_ll = np.hstack([xdell_dm.flatten() + self._dcorss_dm().flatten() + self._d_ent_d_m().flatten(),
                                 (xdell_dS.flatten() + self._dcross_dS().flatten() + self._d_ent_d_S().flatten())* ds_dx,
                                 mdot(xdell_dpi + xdcorss_dpi + self._d_ent_d_pi(), dpi_dx)
                                   ])

    def _set_params(self, p):
        """
        receives parameter from optimizer and transforms them
        :param p: input parameters
        """
        # print 'set', p
        self.last_param = p
        self.MoG.m_from_array(p[:self.MoG.n])
        self.MoG.s_from_array(np.exp(p[self.MoG.n:(2 * self.MoG.n)]))
        pis = np.exp(p[(2 * self.MoG.n):])
        self.MoG.pi = pis / sum(pis)
        self._update()

    def _get_params(self):
        """
        exposes parameters to the optimizer
        """
        # print 'get'
        return np.hstack([self.MoG.m.flatten(), self.MoG.s.flatten(), self.MoG.pi])

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
        return mdot(self.kernel.K(p_X, self.Z[j,:,:]), self.invZ[j,:,:])

    def _Kdiag(self, p_X, A, j):
        """
        calculating diagonal terms of K_tilda for latent process j (eq 4)
        """
        return self.kernel.Kdiag(p_X) - mdiag_dot(A, self.kernel.K(self.Z[j,:,:], p_X))

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


    def _ell(self, n_sample, p_X, p_Y, cond_log_likelihood):

        """
        calculating expected log-likelihood, and it's derivatives
        :returns ell, normal ell, dell / dm, dell / ds, dell/dpi
        """

        # print 'ell started'
        total_ell = 0
        d_ell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        d_ell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        d_ell_dPi = np.zeros(self.num_MoG_comp)
        Aj = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(p_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(p_X, j)
            Kj[j] = self._Kdiag(p_X, Aj[j], j)

        for n in  range(len(p_X)):
            # print 'ell for point #', n
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))

            k_ell = np.zeros(self.num_MoG_comp)
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
                k_ell[k] += sum(cond_ll)
                d_ell_dPi[k] += sum(cond_ll)
                for j in range(self.num_latent_proc):
                    s_dell_dm[k,j] += np.dot(f[:,k,j] - mean_kj[k,j], cond_ll)
                    s_dell_dS[k,j] += np.dot(sigma_kj[k,j] ** -2 * (f[:,k,j] - mean_kj[k,j]) ** 2 - sigma_kj[k,j] ** -1, cond_ll)

                total_ell += k_ell[k] * self.MoG.pi[k]

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    d_ell_dm[k,j] += 1./sigma_kj[k,j] * s_dell_dm[k,j] * self.kernel.K(p_X[np.newaxis, n, :], self.Z[j,:,:])[0,:]

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    d_ell_dS[k,j] += (Aj[j,n] * Aj[j,n].T)* s_dell_dS[k,j]

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dm[k,j,:] = self.MoG.pi[k]/n_sample * mdot(self.invZ[j,:,:], d_ell_dm[k,j,:])

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dS[k,j,:] = self.MoG.pi[k]/n_sample / 2. * d_ell_dS[k,j,:]

        d_ell_dPi = d_ell_dPi/n_sample

        return total_ell / n_sample, d_ell_dm, d_ell_dS, d_ell_dPi

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
        dc_ds = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dc_ds[:,j] = -1. /2 * np.array([np.diag(self.invZ[j,:,:]) * self.MoG.pi[k] for k in range(self.num_MoG_comp)])
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
        s_k = np.zeros(self.num_inducing)
        for l in range(self.num_MoG_comp):
            s_k += self.MoG.pi[k] * self.MoG.pi[l] * (self.N_kl_z_k[k,l] + self.N_kl_z_k[l, k]) * \
                   self.MoG.C_m_C(j, k, l)
        return 1./2 * s_k

    def _d_ent_d_S(self):
        dent_ds = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                dent_ds[k,j] = self._d_ent_d_S_kj(k,j)
        return dent_ds

    def _l_ent(self):
        return -np.dot(self.MoG.pi,  self.log_z)






