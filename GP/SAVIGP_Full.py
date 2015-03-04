from GPy.util.linalg import mdot
from MoG_Full import MoG_Full

__author__ = 'AT'

from SAVIGP import SAVIGP


class SAVIGP_Full(SAVIGP):
    def __init__(self, X, Y, num_inducing, num_MoG_comp, likelihood, kernels, n_samples, config_list):
        super(SAVIGP_Full, self).__init__(X, Y, num_inducing, num_MoG_comp, likelihood, kernels, n_samples, config_list)

    def _get_MoG(self):
        return MoG_Full(self.num_MoG_comp, self.num_latent_proc, self.num_inducing)

    def mdot_Aj(self,Ajn):
        return mdot(Ajn.T, Ajn)
