__author__ = 'AT'


from GPy.util.linalg import mdot
import math
from numpy.linalg import inv
from savigp_single_comp import SAVIGP_SingleComponent
from savigp import SAVIGP, Configuration
import numpy as np

class SAVIGP_Reparam(SAVIGP_SingleComponent):

    def init_mog(self, init_m):
        super(SAVIGP_SingleComponent, self).init_mog(init_m)
        for j in range(self.num_latent_proc):
            self.MoG.update_covariance(j, inv(self.Kzz[j] + 0.001))

    def _proj_m_grad(self, j, dl_dm):
        return dl_dm

    def mdot_Aj(self, Ajn, Kxnz):
        return mdot(Kxnz.T, Kxnz)

    def _b_n(self, n, j, Aj, Kzx):
        """
        calculating [b_k(n)]j for latent process j (eq 19) for all k
        returns: a
        """
        return mdot(Kzx[n, :], self.MoG.m[:, j, :].T)

    def _sigma_n(self, n, j, Kj, Aj, Kzx):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        if Kj[n] < 0:
            Kj[n] = 0
        return Kj[n] + self.MoG.aSa(Kzx[n, :], j)


    def _dcorss_dm(self):
        """
        calculating d corss / dm
        """
        dcdm = np.empty((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dcdm[:, j, :] = -mdot(self.Kzz[j, :, :], self.MoG.m[:, j, :].T).T * self.MoG.pi[:, np.newaxis]
        return dcdm

    def _dcross_ds(self):
        """
        calculating L_corss by s_k for all k's
        """
        dc_ds = np.empty((self.num_mog_comp, self.num_latent_proc, self.MoG.get_sjk_size()))
        for j in range(self.num_latent_proc):
            dc_ds[:, j] = -1. / 2 * np.array(
                [self.MoG.dAS_dS(self.Kzz[j, :, :], k, j) * self.MoG.pi[k] for k in range(self.num_mog_comp)])
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
                    mdot(self.MoG.m[k, j, :].T, self.Kzz[j, :, :], self.MoG.m[k, j, :].T) + \
                    self.MoG.tr_A_mult_S(self.Kzz[j, :, :], k, j)
        for k in range(self.num_mog_comp):
            cross += self.MoG.pi[k] * d_pi[k]

        d_pi *= -1. / 2
        cross *= -1. / 2
        return cross, d_pi

    def _dcross_K(self, j):
        dc_dK = np.zeros((self.num_inducing, self.num_inducing))
        for k in range(self.num_mog_comp):
            dc_dK += -0.5 * self.MoG.pi[k] * (self.invZ[j]
                                              + mdot(self.MoG.m[k, j, :, np.newaxis], self.MoG.m[k, j, :, np.newaxis].T) +
                                              self.MoG.s[k, j, :, :]

                                              )
        return dc_dK

    def _dent_dhyper(self):
        dc_dh = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            self.kernels_latent[j].update_gradients_full(self.invZ[j], self.Z[j])
            dc_dh[j] = self.kernels[j].gradient.copy()
        return dc_dh

    def _l_ent(self):
        ent = -np.dot(self.MoG.pi,  self.log_z)
        for j in range(self.num_latent_proc):
            ent += self.log_detZ[j]
        return ent

    def _dsigma_n_dhyp(self, j, k, A, Kxnz, n, xn):
        return self.dKx_dhyper(j, xn) \
               - self.dA_dhyper_n_mult_x(xn, j, A[j, n], -Kxnz.T) \
               - self.dKzxn_dhyper_mult_x(j, xn, A[j, n]) + \
               2 * self.dKzxn_dhyper_mult_x(j, xn, self.MoG.Sa(Kxnz, k, j))


    def _db_n_dhyp(self, j, k, A, n, xn):
        return self.dKzxn_dhyper_mult_x(j, xn, self.MoG.m[k, j])

    def calculate_dhyper(self):
        return Configuration.HYPER in self.config_list

