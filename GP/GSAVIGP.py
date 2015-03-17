from savigp import SAVIGP
from savigp_diag import SAVIGP_Diag
from util import cross_ent_normal
import numpy as np


class GSAVIGP(SAVIGP_Diag):
    """
    Scalable Variational Inference Gaussian Process where the conditional likelihood is gaussian

    :param X: input observations
    :param Y: outputs
    :param num_inducing: number of inducing variables
    :param num_MoG_comp: number of components of the MoG
    :param likelihood: conditional likelihood function
    :param normal_sigma: covariance matrix of the conditional likelihood function
    :param kernels: of the GP
    :param n_samples: number of samples drawn for approximating ell and its gradient
    :rtype: model object
    """
    def __init__(self, X, Y, num_inducing, num_mog_comp, likelihood, normal_sigma, kernels, n_samples, config_list):
        self.normal_sigma = normal_sigma
        super(GSAVIGP, self).__init__(X, Y, num_inducing, num_mog_comp, likelihood, kernels, n_samples, config_list)


    def _ell(self, n_sample, X, Y, cond_log_likelihood):
        xell, xdell_dm, xdell_dS, xdell_dpi, xell_hyper = super(GSAVIGP, self)._ell(n_sample, X, Y, cond_log_likelihood)
        gell = self._gaussian_ell(X, Y, self.normal_sigma)
        return gell, xdell_dm, xdell_dS, xdell_dpi, xell_hyper

    def _predict(self, Xnew, which_parts='all', full_cov=False, stop=False):
        return self._gaussian_predict(Xnew, self.normal_sigma)
