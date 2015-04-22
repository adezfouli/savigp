from GPy.util.linalg import mdot
import math

__author__ = 'AT'

from savigp import SAVIGP
import numpy as np

class SAVIGP_Reparam(SAVIGP):


    def _proj_m_grad(self, dl_dm, j):
        return dl_dm

    def mdot_Aj(self, Ajn, Kxnz):
        return mdot(Kxnz.T, Kxnz)

    def _b(self, n, j, Aj, Kzx):
        """
        calculating [b_k(n)]j for latent process j (eq 19) for all k
        returns: a
        """
        return mdot(Kzx, self.MoG.m[:, j, :].T)

    def _sigma(self, n, j, Kj, Aj, Kzx):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        if Kj[n] < 0:
            Kj[n] = 0
        return Kj[n] + self.MoG.aSa(Kzx, j)


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
