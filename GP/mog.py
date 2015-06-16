__author__ = 'AT'

from GPy.util.linalg import mdot
import numpy as np


class MoG:
    """
    A generic class for representing posterior distribution. MoG stands for mixture of Gaussians, and this class is
    suitable when posterior is a mixture of Gaussians. The posterior distribution is as follows:

    q(u) = \sum_K \mult_Q N(u, m[k,j], s[k,j])

    """

    def __init__(self, num_comp, num_process, num_dim):
        """
        :param num_comp: number of components
        :param num_process: number of latent processes
        :param num_dim: dimensionality of each Gaussian.
        :return: None
        """
        self.num_comp = num_comp
        self.num_process = num_process
        self.num_dim = num_dim
        self.m = []
        self.pi = []
        self.parameters = []

    def __str__(self):
        return 'm:' + str(self.m) + '\n' + 's:' + str(self.s) + '\n' + 'pi:' + str(self.pi)

    def update_parameters(self, params):
        """
        Sets parameters of the posterior distribution.

        Parameters
        ----------
        params: ndarray
         an array of size  = self.get_m_size() + self.get_s_size() + self.num_ comp, which will be used to update
         parameters
        :return: None
        """
        self.parameters = params
        self.m_from_array(params[:self.get_m_size()])
        self.s_from_array(params[self.get_m_size():(self.get_m_size() + self.get_s_size())])
        self.pi_from_array(params[(self.get_m_size() + self.get_s_size()):])
        self._update()

    def pi_dim(self):
        """
        :return: number of components (dimensionality of the `pi` array)
        """
        return self.num_comp

    def m_dim(self):
        """
        :return: dimensionality of the mean of distributions
        """
        return self.num_comp, self.num_process, self.num_dim

    def _fixed_init(self):
        """
        Initializes posterior distributions using fixed numbers
        :return: None
        """
        self.m = np.random.uniform(low=0.0, high=0.0, size=(self.num_comp, self.num_process, self.num_dim))
        self.pi_from_array(np.random.uniform(low=1.0, high=5.0, size=self.num_comp))

    def transform_S_grad(self, g):
        r"""
        transforms gradients of of `s` to be in the original space, i.e., space of the values the was used
        in `updata_parameters`. Assume:

        g = df \\ dS, where S is the posterior covariance,

        then this function returns:

        :returns: df \\ dL, where L the representation of the parameter in the raw space.
        """
        raise NotImplementedError

    def _random_init(self):
        """
        Initialised posterior parameters randomly
        :return:
        """
        self.m = np.random.uniform(low=-15.1, high=15.1, size=(self.num_comp, self.num_process, self.num_dim))
        self.pi_from_array(np.random.uniform(low=1.0, high=10.0, size=self.num_comp))

    def random_init(self):
        self._random_init()
        self._update()

    def pi_from_array(self, p):
        """
        Builds p (weight of each component) from an unconstrained array.
        :param p:
         a ndarray of size num_comp
        :return:
         None
        """
        pis = np.exp(p)
        self.pi = pis / sum(pis)
        self.pi_untrans = p.copy()

    def dpi_dx(self):
        pit = np.repeat(np.array([self.pi.T]), self.num_comp, 0)
        return pit * (-pit.T + np.eye(self.num_comp))

    def transform_pi_grad(self, p):
        """
        Returns gradient of the `p` array wrt to the untransformed parameters, i.e., the parameters that will be exposed
        to the optimiser.
        :param p: input array to calculate its gradient
        :return:
        """
        return mdot(p, self.dpi_dx())

    def get_m_size(self):
        """
        :return: total size of the array containing `m` of all components and processes
        """
        return self.num_comp * self.num_process * self.num_dim

    def updata_mean(self, j, mj):

        """
        Update mean of the latenet process `j` using `mj` for all components.
        :param j:
         the latent process to update
        :param mj:
         the mean used to update
        :return: None
        """
        for k in range(self.num_comp):
            self.m[k, j, :] = mj.copy()
        self._update()

    def update_covariance(self, j, Sj):
        """ updates covariance matrix j using Sj """
        raise NotImplementedError

    def num_parameters(self):
        """ return number of free parameters of a model """
        raise NotImplementedError

    def get_s_size(self):
        """ return size of s when flattened """
        raise NotImplementedError

    def S_dim(self):
        """ dimensionality of nonzero elements in the covariance matrix """
        raise NotImplementedError

    def m_from_array(self, ma):
        """ initializes the mean from ma"""
        raise NotImplementedError

    def get_sjk_size(self):
        """ returns number of elements needed to represent covariance matrix of each component and each latent process"""
        raise NotImplementedError

    def s_from_array(self, sa):
        """ initializes the covariance matrix from `sa`. Note that `sa` is in the raw space, ie., it is coming directly
        form the optimiser"""
        raise NotImplementedError

    def log_pdf(self, j, k, l):
        """ :return N_j(m_k|m_l, S_l + S_k)"""
        raise NotImplementedError

    def tr_AinvS(self, L, k, j):
        """
        Assuming that `L` is the cholesky decomposition of A

        :return  trace(A^-1 s[k,j]) """
        raise NotImplementedError

    def tr_AS(self, A, k, j):
        """
        :return: trace (A s[k,j])
        """
        raise NotImplementedError

    def aSa(self, a, k, j):
        """ :return  a s[k,j] a"""
        raise NotImplementedError

    def mmTS(self, k, j):
        """ :return  m_kj m_kj^T  + s_kj  """
        raise NotImplementedError

    def dAinvS_dS(self, L, k, j):
        r"""
        Assuming L = chol (A), then this function calculates dA^{-1}s[k,j] \\ ds[k,j] and transforms the results to the
        raw space i.e., ready for exposing to the optimiser"""
        raise NotImplementedError

    def dAS_dS(self, L, k, j):
        """ :return  dA^{-1}S dS  """
        raise NotImplementedError


    def Sa(self, a, k, j):
        """ :return  S_kj a  """
        raise NotImplementedError

    def _update(self):
        """ updates internal variables of the class """
        pass

    def get_m_S_params(self):
        """
        Returns a tuple (m, s), which contains mean and covariance matrix of the posterior, which can be used for
        example by the optimize to evaluate the amount of change in posterior parameters.
        """
        raise NotImplementedError
