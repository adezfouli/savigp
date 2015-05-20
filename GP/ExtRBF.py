from GPy.kern import RBF
from GPy.util.linalg import mdot
import numpy as np
from numpy.core.umath_tests import inner1d


class ExtRBF(RBF):
    def get_gradients_Kn(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        variance_gradient = inner1d(self.K(X, X2), dL_dK) *  1./ self.variance

        dL_dr = (self.dK_dr_via_X(X, X2) * dL_dK)
        if self.ARD:
            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None: X2 = X
            lengthscale_gradient = np.array([np.einsum('ij,ij,...->i', tmp, np.square(X[:,q:q+1] - X2[:,q:q+1].T), -1./self.lengthscale[q]**3) for q in xrange(self.input_dim)])
        else:
            r = self._scaled_dist(X, X2)
            lengthscale_gradient = -np.sum(dL_dr*r, axis=1)/self.lengthscale
            lengthscale_gradient = lengthscale_gradient[np.newaxis, :]

        return np.hstack((variance_gradient[:, np.newaxis], lengthscale_gradient.T))


    def get_gradients_Kdiagn(self, X):

        variance_gradient = self.Kdiag(X) * 1./self.variance
        return np.hstack((variance_gradient[:, np.newaxis], np.zeros((X.shape[0], self.lengthscale.shape[0]))))

    def get_gradients_Kzz(self, S, D, X, X2=None):
        variance_gradient = mdot(S, self.K(X, X2), D) * 1./self.variance

        if X2 is None: X2 = X
        if self.ARD:
            rinv = self._inv_dist(X, X2)
            d =  X[:, None, :] - X2[None, :, :]
            x_xl3 = np.square(d) * (rinv * self.dK_dr_via_X(X, X2))[:,:,None]
            lengthscale_gradient = -np.tensordot(D, np.tensordot(S, x_xl3, (1,0)), (0,1)) / self.lengthscale**3
            lengthscale_gradient = np.diagonal(lengthscale_gradient).T
        else:
            lengthscale_gradient = np.diagonal(-mdot(S, (self._scaled_dist(X, X2) * self.dK_dr_via_X(X, X2)).T, D) / self.lengthscale)[:, np.newaxis]

        return np.hstack((np.diagonal(variance_gradient)[:, np.newaxis], lengthscale_gradient))
