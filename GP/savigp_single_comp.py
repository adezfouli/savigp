from savigp import SAVIGP

__author__ = 'AT'

from GPy.util.linalg import mdot
from mog_single_comp import MoG_SingleComponent
import numpy as np


class SAVIGP_SingleComponent(SAVIGP):
    def __init__(self, X, Y, num_inducing, likelihood, kernels, n_samples, config_list, latent_noise, is_exact_ell, random_Z, n_threads =1):
        super(SAVIGP_SingleComponent, self).__init__(X, Y, num_inducing, 1, likelihood,
                                                     kernels, n_samples, config_list, latent_noise, is_exact_ell, random_Z, n_threads)

    def _dell_ds(self, k, j, cond_ll, A, sigma_kj, norm_samples):
        return  mdot(A[j].T * self._average(cond_ll, (norm_samples**2 - 1)/sigma_kj[k,j], True), A[j]) \
                                                * self.MoG.pi[k] / 2.

        # a bit less memory intensive
        # return np.einsum('i,ij,ki->jk', mdot(self.normal_samples[j,:]**2 - 1, cond_ll / sigma_kj[k,j])
        #                                           , A[j], A[j].T) * self.MoG.pi[k] / n_sample / 2.

        # a bit faster but high memory
        # return mdot(self.normal_samples[j,:]**2 - 1, cond_ll / sigma_kj[k,j]
        #                                           , np.einsum('ij,ki->ijk', A[j], A[j].T)) * self.MoG.pi[k] / n_sample / 2.

    def init_mog(self, init_m):
        super(SAVIGP_SingleComponent, self).init_mog(init_m)
        for j in range(self.num_latent_proc):
            self.MoG.update_covariance(j, self.Kzz[j])

    def _update(self):
        self.update_N_z()
        SAVIGP._update(self)

    def mdot_Aj(self, Ajn, Kxnz):
        return mdot(Ajn.T, Ajn)

    def _get_mog(self):
        return MoG_SingleComponent(self.num_latent_proc, self.num_inducing)

    def _d_ent_d_m(self):
        return np.zeros((self.num_mog_comp, self.num_latent_proc, self.num_inducing))

    def _d_ent_d_pi(self):
        return -self.log_z[0] - 1

    def _d_ent_d_S_kj(self, k, j):
        return self.MoG.invC_klj[0,0,j]

    def _d_ent_d_S(self):
        dent_ds = np.empty((self.num_mog_comp, self.num_latent_proc) + self.MoG.S_dim())
        for k in range(self.num_mog_comp):
            for j in range(self.num_latent_proc):
                dent_ds[k,j] = self._d_ent_d_S_kj(k,j)
        return dent_ds

    def _l_ent(self):
        return -np.dot(self.MoG.pi,  self.log_z)

    def update_N_z(self):
        self.log_z = np.zeros((self.num_mog_comp))
        for j in range(self.num_latent_proc):
            self.log_z[0] += self.MoG.log_pdf(j, 0, 0)

    def _transformed_d_ent_d_S(self):
        return self.MoG.transform_eye_grad()
