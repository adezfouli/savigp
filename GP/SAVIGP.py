# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import copy
import math
from DerApproximator import get_d1, check_d1

import numpy as np
from numpy.linalg import inv, det
from scipy.linalg import block_diag
import scipy.stats
import GPy
from GPy.core import Model
from GPy.util.linalg import mdot
from MoG import MoG
from MoG_Diag import MoG_Diag
from util import mdiag_dot

class SAVIGP(Model):

    def __init__(self, X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, normalize_X):
        super(SAVIGP, self).__init__()

        self.MoG = MoG_Diag(num_MoG_comp, num_latent_proc, num_inducing)
        self.input_dim = X[0].shape[0]
        self.num_inducing = num_inducing
        self.num_latent_proc = num_latent_proc
        self.num_MoG_comp = num_MoG_comp
        self.kernel = GPy.kern.rbf(X.shape[1])
        self.likelihood = likelihood


        Z = np.array([np.zeros((self.num_inducing, self.input_dim))] * self.num_latent_proc)
        for j in range(self.num_latent_proc):
            i = np.random.permutation(X.shape[0])[:self.num_inducing]
            Z[j, :, :] = X[i].copy()

        # Z is Q * M * D
        self.Z = Z
        self.invZ = np.array([np.zeros((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        self._update()


    def _update(self):
        """
        updating internal variables for later use
        """

        # updating inverse of Zjj
        for j in range(self.num_latent_proc):
            self.invZ[j, :, :] = inv(self.kernel.K(self.Z[j,:,:], self.Z[j,:,:]))


        #updating N_lk, and z_k used for updating d_ent
        self.N_kl = np.ones((self.num_MoG_comp, self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
            for l in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    self.N_kl[k,l] *= \
                        scipy.stats.multivariate_normal.pdf(self.MoG.m[k,j,:],
                                  mean=self.MoG.m[l,j,:], cov=np.diag(self.MoG.s[l,j,:] + self.MoG.s[k,j,:]))


        self.z = np.zeros((self.num_MoG_comp))
        for k in range(self.num_MoG_comp):
            for l in range(self.num_MoG_comp):
                self.z[k] += self.MoG.pi[l] * self.N_kl[k,l]

        self.invC_klj = np.empty((self.num_MoG_comp, self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for k in range(self.num_MoG_comp):
            for l in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    self.invC_klj[k,l,j] = 1. /(self.MoG.s[l,j,:] + self.MoG.s[k,j,:])


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
        return Kj[n] + mdot(self.MoG.s[:,j,:], (Aj[n, :] * Aj[n, :].T))

    def _ell(self, n_sample, p_X, p_Y):

        """
        calculating expected log-likelihood, and it's derivatives
        """

        print 'ell started'
        total_ell = 0
        d_ell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        d_ell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing, self.num_inducing))
        d_ell_dPi = np.zeros(self.num_MoG_comp)
        Aj = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(p_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(p_X, j)
            Kj[j] = self._Kdiag(p_X, Aj[j], j)

        for n in  range(len(p_X)):
            print 'ell for point #', n
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

            k_ell = np.zeros(self.num_MoG_comp)
            s_dell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc))
            s_dell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc))

            f = np.empty((n_sample, self.num_MoG_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])
                for k in range(self.num_MoG_comp):
                    f[:,k, j] = np.random.normal(mean_kj[k,j], math.sqrt(sigma_kj[k,j]))

            for k in range(self.num_MoG_comp):
                for i in range(n_sample):
                    ### for calculating ell
                    self_likelihood = self.likelihood(f[i,k,:], p_Y[n, :])
                    k_ell[k] += self_likelihood
                    d_ell_dPi[k] += self_likelihood
                    ### for calculating d_ell_dm
                    for j in range(self.num_latent_proc):
                        s_dell_dm[k,j] += (f[i,k,j] - mean_kj[k,j]) * self_likelihood
                        s_dell_dS[k,j] += (sigma_kj[k,j] ** -2 * (f[i,k,j] - mean_kj[k,j]) ** 2 - sigma_kj[k,j] ** -1) * self_likelihood
                    pass
                total_ell += k_ell[k] * self.MoG.pi[k]

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    d_ell_dm[k,j] += 1./sigma_kj[k,j] * s_dell_dm[k,j] * self.kernel.K(p_X[np.newaxis, n, :], self.Z[j,:,:])[0,:]

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    d_ell_dS[k,j] += np.multiply(Aj[j,n], Aj[j,n].T)* s_dell_dm[k,j]

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dm[k,j,:] = self.MoG.pi[k]/n_sample * mdot(self.invZ[j,:,:], d_ell_dm[k,j,:])

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dS[k,j,:] = self.MoG.pi[k]/n_sample / 2 * d_ell_dm[k,j,:]

        for k in range(self.num_MoG_comp):
            d_ell_dPi /= n_sample

        return total_ell / n_sample, d_ell_dm, d_ell_dS, d_ell_dPi

    def _dcorss_dm(self):
        """
        calculating d corss / dm
        """
        dcdm = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dcdm[:,j,:] = -mdot(self.MoG.m[:,j,:], self.invZ[j,:,:]) * self.MoG.pi[:, np.newaxis]
        return dcdm

    def _dcorss_dS(self):
        """
        calculating L_corss by s_k for all k's
        """
        dcds = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dcds[:,j,:] = -1. /2 * np.array([np.diag(self.invZ[j,:,:]) * self.MoG.pi[k] for k in range(self.num_MoG_comp)])
        return dcds

    def _cross_dcorss_dpi(self, N):
        """
        calculating L_corss by pi_k, and also calculates the cross term
        returns: d cross / d pi, cross
        """
        cross = 0
        d_pi = np.zeros(self.num_MoG_comp)
        for j in range(self.num_latent_proc):
            detK = math.log(det(self.kernel.K(self.Z[j,:,:], self.Z[j, :, :])))
            for k in range(self.num_MoG_comp):
                d_pi[k] +=  \
                        N * math.log(2 * math.pi) + \
                        detK + \
                        mdot(self.MoG.m[k, j, :].T, self.invZ[j,:,:], self.MoG.m[k, j, :]) + \
                        np.dot(np.diagonal(self.invZ[j,:,:]), self.MoG.s[k,j,:])
        for k in range(self.num_MoG_comp):
            cross += self.MoG.pi[k] * d_pi[k]
        d_pi *= -1. / 2
        cross *= -1. / 2
        return cross, d_pi

    def _d_ent_d_m_kj(self, k, j):
        m_k = np.zeros(self.num_inducing)
        for l in range(self.num_MoG_comp):
            m_k += self.MoG.pi[k] * self.MoG.pi[l] * (self.N_kl[k,l] / self.z[k] + self.N_kl[k,l] / self.z[l]) * \
                    (self.invC_klj[k,l,j] * (self.MoG.m[k,j,:] - self.MoG.m[l,j,:]))
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
            pi[k] = -math.log(self.z[k])
            for l in range(self.num_MoG_comp):
                pi[k] -= self.MoG.pi[l] * self.N_kl[k,l] / self.z[l]
        return pi

    def _d_ent_d_S_kj(self, k, j):
        s_k = np.zeros(self.num_inducing)
        for l in range(self.num_MoG_comp):
            s_k += self.MoG.pi[k] * self.MoG.pi[l] * (self.N_kl[k,l] / self.z[k] + self.N_kl[k,l] / self.z[l]) * \
                    (self.invC_klj[k,l,j] -
                     self.invC_klj[k,l,j] * (self.MoG.m[k,j,:] -  self.MoG.m[l,j,:]) *
                    (self.MoG.m[k,j,:] - self.MoG.m[l,j,:]) * self.invC_klj[k,l,j])
        return 1./2 * s_k

    def _d_ent_d_S(self):
        dent_ds = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                dent_ds[k,j,:] = self._d_ent_d_S_kj(k,j)
        return dent_ds

    def l_ent(self):
        return -np.dot(self.MoG.pi,  np.log(self.z))



    @staticmethod
    def normal_likelihood(epsilon):
        def ll(f, y):
            return sum(f) - y ** 2
            return scipy.stats.norm.logpdf(y, loc=f[0], scale=epsilon)
        return ll


def generate_samples(num_samples, input_dim):
    # seed=1000
    noise=0.02
    # np.random.seed(seed=seed)
    X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, input_dim))
    X.sort(axis=0)
    rbf = GPy.kern.rbf(input_dim, variance=1., lengthscale=np.array((0.25,)))
    white = GPy.kern.white(input_dim, variance=noise)
    kernel = rbf + white
    Y = np.sin([X.sum(axis=1).T]).T + np.random.randn(num_samples, 1) * 0.05
    return X, Y, kernel, noise


def test():
    num_input_samples = 50
    input_dim = 20
    num_inducing = 3
    num_MoG = 3
    num_latent_proc = 2
    num_samples = 1000

    X, Y, kernel, noise = generate_samples(num_input_samples, input_dim)
    print 'samples generated'
    np.random.seed()

    s1 = SAVIGP(X, Y, num_inducing, num_MoG, num_latent_proc, SAVIGP.normal_likelihood(1), False)
    # print s1._ell(num_samples, X, Y)


