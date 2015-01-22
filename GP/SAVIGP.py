# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import math

import numpy as np
from numpy.linalg import inv, det
import scipy.stats
import GPy
from GPy.core import Model
from GPy.util.linalg import mdot
from MoG import MoG


class SAVIGP(Model):

    def __init__(self, X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, normalize_X):
        super(SAVIGP, self).__init__()

        self.MoG = MoG(num_MoG_comp, num_latent_proc, num_inducing)
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
        updating interval variables for later use
        """

        # updating inverse of Zjj
        for j in range(self.num_latent_proc):
            self.invZ[j, :, :] = inv(self.kernel.K(self.Z[j,:,:], self.Z[j,:,:]))


    def _A(self, p_X, j):
        """
        calculating A for latent process j (eq 4)
        """
        return mdot(self.kernel.K(p_X, self.Z[j,:,:]), self.invZ[j,:,:])

    def _K(self, p_X, A, j):
        """
        calculating K_tilda for latent process j (eq 4)
        """
        return self.kernel.K(p_X, p_X) -  mdot(A, self.kernel.K(self.Z[j,:,:], p_X))

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
        return Kj[n,n] + mdot(Aj[n, :], self.MoG.sigma[:, j, :, :], Aj[n, :].T)

    def _ell(self, n_sample, p_X, p_Y):
        """
        calculating expected log-likelihood, and it's derivatives
        """
        total_ell = 0
        d_ell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing))
        d_ell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc, self.num_inducing, self.num_inducing))
        d_ell_dPi = np.zeros((self.num_MoG_comp))
        Aj = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(p_X), len(p_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(p_X, j)
            Kj[j] = self._K(p_X, Aj[j], j)

        for n in  range(len(p_X)):
            print 'ell for point #', n
            mean_kj = np.zeros((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.zeros((self.num_MoG_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

            k_ell = np.zeros(self.num_MoG_comp)
            s_dell_dm = np.zeros((self.num_MoG_comp, self.num_latent_proc))
            s_dell_dS = np.zeros((self.num_MoG_comp, self.num_latent_proc))
            for k in range(self.num_MoG_comp):
                for i in range(n_sample):
                    f = [np.random.normal(mean_kj[k,j], math.sqrt(sigma_kj[k,j])) for j in range(self.num_latent_proc)]
                    #### for calculating ell
                    self_likelihood = self.likelihood(f, p_Y[n, :])
                    k_ell[k] += self_likelihood
                    d_ell_dPi[k] += self_likelihood
                    #### for calculating d_ell_dm
                    for j in range(self.num_latent_proc):
                        s_dell_dm[k,j] += (f[j] - mean_kj[k,j]) * self_likelihood
                        s_dell_dS[k,j] += (sigma_kj[k,j] ** -2 * (f[j] - mean_kj[k,j]) ** 2 - sigma_kj[k,j] ** -1) * self_likelihood
                total_ell += k_ell[k] * self.MoG.pi[k]

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    d_ell_dm[k,j] += 1./sigma_kj[k,j] * s_dell_dm[k,j] * self.kernel.K(p_X[np.newaxis, n, :], self.Z[j,:,:])[0,:]

            for k in range(self.num_MoG_comp):
                for j in range(self.num_latent_proc):
                    d_ell_dS[k,j] += mdot(Aj[j,n], Aj[j,n].T)* s_dell_dm[k,j]

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dm[k,j,:] = self.MoG.pi[k]/n_sample * mdot(self.invZ[j,:,:], d_ell_dm[k,j,:])

        for k in range(self.num_MoG_comp):
            for j in range(self.num_latent_proc):
                d_ell_dS[k,j,:] = self.MoG.pi[k]/n_sample / 2 * d_ell_dm[k,j,:]

        for k in range(self.num_MoG_comp):
            d_ell_dPi /= n_sample

        return total_ell / n_sample, d_ell_dm, d_ell_dS, d_ell_dPi

    def _dcorss_dm(self, j):
        """
        calculating L_corss by m_k for all k's: equation 26
        """
        return mdot(self.MoG.m[:,j,:], self.invZ[j,:,:]) * self.MoG.pi[:, np.newaxis]

    def _dcorss_dS(self, j):
        """
        calculating L_corss by s_k for all k's: equation 27
        """
        return np.array([self.invZ[j,:,:] * self.MoG.pi[k] for k in range(self.num_MoG_comp)])

    def _dcorss_dpi(self, N):
        d_pi = np.zeros(self.num_MoG_comp)
        for j in range(self.num_latent_proc):
            detK = math.log(det(self.kernel.K(self.Z[j,:,:], self.Z[j, :, :])))
            for k in range(self.num_MoG_comp):
                d_pi[k] +=  \
                        N * math.log(2 * math.pi) + \
                        detK + \
                        mdot(self.MoG.m[k, j, :].T, self.invZ[j,:,:], self.MoG.m[k, j, :]) + \
                        np.matrix.trace(mdot(self.invZ[j,:,:], self.MoG.sigma[k,j,:]))
        d_pi *= -1. / 2
        return d_pi


    @staticmethod
    def normal_likelihood(epsilon):
        def ll(f, y):
            return scipy.stats.multivariate_normal(mean=sum(f), cov=np.eye(len(y)) * math.sqrt(epsilon)).logpdf(y)
        return ll

def generate_samples():
    # seed=1000
    num_samples=2
    noise=0.02
    # np.random.seed(seed=seed)
    num_in = 1
    X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, num_in))
    X.sort(axis=0)
    rbf = GPy.kern.rbf(num_in, variance=1., lengthscale=np.array((0.25,)))
    white = GPy.kern.white(num_in, variance=noise)
    kernel = rbf + white
    K = kernel.K(X)
    y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
    return X, y, kernel, noise



def test():
    X, Y, kernel, noise = generate_samples()
    np.random.seed()
    s1 = SAVIGP(X, Y, 2, 1, 1, SAVIGP.normal_likelihood(1), False)
    s2 = SAVIGP(X, Y, 2, 1, 1, SAVIGP.normal_likelihood(1), False)
    print s1._ell(4000, X, Y)

test()

