from GPy.kern import RBF
from GPy.util.linalg import mdot
import numpy as np
from numpy.core.umath_tests import inner1d


class ExtRBF(RBF):
    """
    This class extends RBF class in order to provide fast methods for calculating gradients wrt to the
    hyper-parameters of the kernel. The base class provides a similar functionality, but that implementation can be
    slow.
    """

    def get_gradients_Kn(self, dL_dK, X, X2=None):
        """
        Dimensions: \n
        dim(dL_DK) = N * M \n
        dim(X) = N * D \n
        dim(X2) = M * D \n
        where D is the dimensionality of input.

        Assume we have a function Ln of the kernel, which its gradient wrt to the hyper-parameters is as follows:

        dLn\dH = (dLn\dK) * dK(X2, Xn)\dH

        where dLn_dK = dLn_dK[n, :], Xn = X[n, :]. The function then returns a matrix containing dLn_dH for all 'n's.

        :returns dL\dH, which is a matrix of dimension N * dim(H), where dim(H) is the number of hyper-parameters.
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
        """
        Dimensions: \n
        dim(X) = N * D, where D is the dimension of input. \n
        Assume we have a function Ln of the kernel we follows: \n
        dL_n\dH = dK(Xn, Xn)\dH \n
        where Xn=X[n, :]. Then the function returns a matrix which contains dL_n\dH for all 'n's

        :return: dL\dH which is a matrix of dimension N * dim(H), where dim(H) is the number of hyper-parameters.
        """

        variance_gradient = self.Kdiag(X) * 1./self.variance
        return np.hstack((variance_gradient[:, np.newaxis], np.zeros((X.shape[0], self.lengthscale.shape[0]))))

    def get_gradients_Kzz(self, S, D, X, X2=None):
        """
        Dimensions:\n
        dim(S) = N * M \n
        dim(D) = M * N \n
        dim(X) = M * d, where d is the input dimensionality \n
        dim(X2) = M * d \n
        Assume we have a function Ln, which its gradient wrt to the hyper-parameters (H), is as follows: \n
        dLn\dH = S[:, n] *  dK(X,X2)\dH * D[n, :] \n
        then this function calculates dLn\dH for all 'n's.

        :returns dL\dH which is a matrix by dimensions N * dim(H), where dim(H) is the number of hyper-parameters.
        """
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
