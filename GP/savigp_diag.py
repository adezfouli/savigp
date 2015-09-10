__author__ = 'AT'

from GPy.util.linalg import mdot
from mog_diag import MoG_Diag
from scipy.misc import logsumexp
from util import log_diag_gaussian
import numpy as np

from savigp import SAVIGP


class SAVIGP_Diag(SAVIGP):
    """
    Implementation of the SAVIGP model in the case that posterior is the mixture of diagonal Gaussians.
    """

    def __init__(self, X, Y, num_inducing, num_mog_comp, likelihood, kernels, n_samples, config_list,
                 latent_noise, is_exact_ell, inducing_on_Xs, n_threads=1, image=None, partition_size=3000):
        super(SAVIGP_Diag, self).__init__(X, Y, num_inducing, num_mog_comp, likelihood,
                                          kernels, n_samples, config_list, latent_noise, is_exact_ell,
                                          inducing_on_Xs, n_threads, image, partition_size)

    def _get_mog(self):
        return MoG_Diag(self.num_mog_comp, self.num_latent_proc, self.num_inducing)

    def _dell_ds(self, k, j, cond_ll, A, sigma_kj, norm_samples):
        s = self._average(cond_ll, (np.square(norm_samples) - 1) / sigma_kj[k, j], True)
        return (mdot(s, np.square(A[j])) * self.MoG.pi[k] / 2.)

    def update_N_z(self):
        self.log_z = np.zeros((self.num_mog_comp))
        self.log_N_kl = np.zeros((self.num_mog_comp, self.num_mog_comp))
        for k in range(self.num_mog_comp):
            for l in range(self.num_mog_comp):
                for j in range(self.num_latent_proc):
                    self.log_N_kl[k, l] += log_diag_gaussian(self.MoG.m[k, j], self.MoG.m[l, j],
                                                             logsumexp(
                                                                 [self.MoG.log_s[k, j, :], self.MoG.log_s[l, j, :]],
                                                                 axis=0))
            self.log_z[k] = logsumexp(self.log_N_kl[k, :] + np.log(self.MoG.pi))

    def _update(self):
        self.update_N_z()
        SAVIGP._update(self)

    def mdot_Aj(self, Ajn, Kxnz):
        return Ajn[0] * Ajn[0]

    def _d_ent_d_m(self):
        dent_dm = np.empty((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        for k in range(self.num_mog_comp):
            for j in range(self.num_latent_proc):
                dent_dm[k, j, :] = self._d_ent_d_m_kj(k, j)
        return dent_dm

    def _d_ent_d_pi(self):
        pi = np.empty(self.num_mog_comp)
        for k in range(self.num_mog_comp):
            pi[k] = -self.log_z[k]
            for l in range(self.num_mog_comp):
                pi[k] -= self.MoG.pi[l] * (np.exp(self.log_N_kl[k, l] - self.log_z[l]))
        return pi

    def _d_ent_d_S_kj(self, k, j):
        """
        Calculates gradient of the entropy term of ELBO wrt to the posterior covariance for component ``k`` and latent
        process ``j``. The returned gradient will be in the raw space.
        """
        s_k = np.zeros(self.MoG.S_dim())
        for l in range(self.num_mog_comp):
            s_k += self.MoG.pi[k] * self.MoG.pi[l] * (np.exp(self.log_N_kl[l, k] - self.log_z[k]) +
                                                      np.exp(self.log_N_kl[l, k] - self.log_z[l])) * \
                   self.MoG.C_m_C(j, k, l)
        return 1. / 2 * s_k

    def _d_ent_d_S(self):
        r"""
        Calculated gradient of the entropy term of ELBO wrt to the posterior covariance.

        Returns
        -------
        ds : ndarray
         dent \\ ds. Gradients will be in the raw space. Dimensions : K * Q * ``self.MoG.S_dim()``

        """
        dent_ds = np.empty((self.num_mog_comp, self.num_latent_proc) + self.MoG.S_dim())
        for k in range(self.num_mog_comp):
            for j in range(self.num_latent_proc):
                dent_ds[k, j] = self._d_ent_d_S_kj(k, j)
        return dent_ds

    def _transformed_d_ent_d_S(self):
        return (self._d_ent_d_S()).flatten()

    def _l_ent(self):
        return -np.dot(self.MoG.pi, self.log_z)

