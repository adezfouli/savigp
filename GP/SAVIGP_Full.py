from SAVIGP import SAVIGP
import numpy as np
__author__ = 'AT'


class SAVIGP_Full(SAVIGP):
    def __init__(self, X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, kernel, n_samples, normalize_X):
        super(SAVIGP_Full, self).__init__(X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, kernel, n_samples, normalize_X)

    def _dcross_dS(self):
        """
        calculating L_corss by s_k for all k's
        """
        dc_ds = np.empty((self.num_MoG_comp, self.num_latent_proc, self.num_inducing, self.num_inducing))
        for j in range(self.num_latent_proc):
            dc_ds[:,j,:, :] = -1. /2 * np.array([self.invZ[j,:,:] * self.MoG.pi[k] for k in range(self.num_MoG_comp)])
        return dc_ds
