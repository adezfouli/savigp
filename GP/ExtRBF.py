from GPy.kern import RBF
from GPy.util import config
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

        #now the lengthscale gradient(s)
        dL_dr = (self.dK_dr_via_X(X, X2) * dL_dK)
        if self.ARD:
            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None: X2 = X
            lengthscale_gradient = np.array([np.einsum('ij,ij,...->i', tmp, np.square(X[:,q:q+1] - X2[:,q:q+1].T), -1./self.lengthscale[q]**3) for q in xrange(self.input_dim)])
        else:
            r = self._scaled_dist(X, X2)
            lengthscale_gradient = -np.sum(dL_dr*r, axis=1)/self.lengthscale

        return variance_gradient, lengthscale_gradient

    def get_gradients_Kzz(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        variance_gradient = inner1d(self.K(X, X2), dL_dK) *  1./ self.variance

        #now the lengthscale gradient(s)
        dL_dr = (self.dK_dr_via_X(X, X2) * dL_dK)
        if self.ARD:
            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None: X2 = X
            lengthscale_gradient = np.array([np.einsum('ij,ij,...->i', tmp, np.square(X[:,q:q+1] - X2[:,q:q+1].T), -1./self.lengthscale[q]**3) for q in xrange(self.input_dim)])
        else:
            r = self._scaled_dist(X, X2)
            lengthscale_gradient = -np.sum(dL_dr*r, axis=1)/self.lengthscale

        return variance_gradient, lengthscale_gradient
