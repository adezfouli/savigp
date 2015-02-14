from SAVIGP import SAVIGP
from util import cross_ent_normal
import numpy as np


class GSAVIGP(SAVIGP):
    """
    Scalable Variational Inference Gaussian Process where the conditional likelihood is gaussian

    :param X: input observations
    :param Y: outputs
    :param num_inducing: number of inducing variables
    :param num_MoG_comp: number of components of the MoG
    :param num_latent_proc: number of latent processes
    :param likelihood: conditional likelihood function
    :param normal_sigma: covariance matrix of the conditional likelihood function
    :param kernel: of the GP
    :param n_samples: number of samples drawn for approximating ell and its gradient
    :rtype: model object
    """
    def __init__(self, X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, normal_sigma, kernel, n_samples, normalize_X):
        self.normal_sigma = normal_sigma
        super(GSAVIGP, self).__init__(X, Y, num_inducing, num_MoG_comp, num_latent_proc, likelihood, kernel, n_samples, normalize_X)

    def _gaussian_ell(self, p_X, p_Y):
        normal_ell = 0
        Aj = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(p_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(p_X, j)
            Kj[j] = self._Kdiag(p_X, Aj[j], j)

        for n in  range(len(p_X)):
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

            for k in range(self.num_MoG_comp):
                if self.normal_sigma is not None:
                    normal_ell += cross_ent_normal(mean_kj[k,:], np.diag(sigma_kj[k,:]), p_Y[n, :], self.normal_sigma) * self.MoG.pi[k]

        return normal_ell

    def _ell(self, n_sample, p_X, p_Y, cond_log_likelihood):
        xell, xdell_dm, xdell_dS, xdell_dpi = super(GSAVIGP, self)._ell(n_sample, p_X, p_Y, cond_log_likelihood)
        gell = self._gaussian_ell(p_X, p_Y)
        return xell, xdell_dm, xdell_dS, xdell_dpi

    def _predict(self, t_X):
        """
        predicting at test points t_X
        :param t_X: test point
        """

        # print 'ell started'
        Aj = np.empty((self.num_latent_proc, len(t_X), self.num_inducing))
        Kj = np.empty((self.num_latent_proc, len(t_X)))
        for j in range(self.num_latent_proc):
            Aj[j] = self._A(t_X, j)
            Kj[j] = self._Kdiag(t_X, Aj[j], j)

        predicted_mu = np.empty((t_X.shape[0], self.num_MoG_comp, self.num_latent_proc))
        predicted_var = np.empty((t_X.shape[0], self.num_MoG_comp, self.num_latent_proc, self.num_latent_proc))
        for n in range(len(t_X)):
            mean_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))
            sigma_kj = np.empty((self.num_MoG_comp, self.num_latent_proc))

            for j in range(self.num_latent_proc):
                mean_kj[:,j] = self._b(n, j, Aj[j])
                sigma_kj[:,j] = self._sigma(n, j, Kj[j], Aj[j])

            predicted_mu[n,:, :] = mean_kj[:,:]
            predicted_var[n, :, :, :] = self.normal_sigma + sigma_kj[:,:]

        return predicted_mu, predicted_var

    def _raw_predict(self, Xnew, which_parts='all', full_cov=False, stop=False):
        mu, var = self._predict(Xnew)
        if self.num_MoG_comp > 1 or self.num_latent_proc > 1:
            raise Exception('unable to plot')
        return np.sum(mu, (1,2))[:, np.newaxis], np.sum(var, (1,2,3))[:, np.newaxis]