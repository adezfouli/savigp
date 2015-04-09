from savigp import SAVIGP
from savigp_diag import SAVIGP_Diag
from util import cross_ent_normal
import numpy as np


class GSAVIGP_Diag(SAVIGP_Diag):
    """
    Scalable Variational Inference Gaussian Process where the conditional likelihood is gaussian

    :param X: input observations
    :param Y: outputs
    :param num_inducing: number of inducing variables
    :param num_mog_comp: number of components of the MoG
    :param likelihood: conditional likelihood function
    :param kernels: of the GP
    :param n_samples: number of samples drawn for approximating ell and its gradient
    :rtype: model object
    """
    def __init__(self, X, Y, num_inducing, num_mog_comp, likelihood, kernels, n_samples,
                 config_list, latent_noise, exact_ell):
        self.exact_ell = exact_ell
        super(GSAVIGP_Diag, self).__init__(X, Y, num_inducing, num_mog_comp, likelihood, kernels, n_samples,
                                           config_list, latent_noise)

    def _ell(self, n_sample, X, Y, cond_log_likelihood):
        xell, xdell_dm, xdell_dS, xdell_dpi, xell_hyper, xdell_dll = super(GSAVIGP_Diag, self)._ell(n_sample, X, Y, cond_log_likelihood)
        if self.exact_ell:
            xell = self._gaussian_ell(X, Y, cond_log_likelihood.get_sigma())
        return xell, xdell_dm, xdell_dS, xdell_dpi, xell_hyper, xdell_dll

    def _predict_kj(self, Xnew):
        return self._gaussian_predict(Xnew, self.cond_likelihood.get_sigma())
