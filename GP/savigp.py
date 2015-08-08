import threading
import math

import GPy
from atom.enum import Enum
from scipy.misc import logsumexp
from sklearn.cluster import MiniBatchKMeans, KMeans
from GPy.util.linalg import mdot
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from GPy.core import Model
from util import mdiag_dot, jitchol, pddet, inv_chol


class Configuration(Enum):
    ENTROPY = 'ENT'
    CROSS = 'CRO'
    ELL = 'ELL'
    HYPER = 'HYP'
    MoG = 'MOG'
    LL = 'LL'
    INDUCING = 'INDUC'


class SAVIGP(Model):
    """
    Provides a general class for Scalable Variational Inference Gaussian Process models.

    Parameters
    ----------
    X : ndarray
     a N * D matrix containing N observation each in a D dimensional space

    Y : ndarray
     a N * O matrix, containing N outputs, where each output is in a O dimensional space

    num_inducing : int
     number of inducing points

    num_mog_comp : int
     number of mixture of Gaussians components used for representing posterior

    likelihood : subclass of likelihood/Likelihood
     likelihood object

    kernels : list
     a list containing kernels (kernels should expose kernel class as in ``GPy``, and methods in ``ExtRBF``)

    n_samples : int
     number of samples used to approximate gradients and objective function

    config_list : list
     configuration of the model. For example:

     config_list = [Configuration.CROSS, Configuration.ELL, Configuration.ENTROPY]

     means that cross entropy, expected log likelihood, and entropy term all contribute to the calculation of gradients.
     The config list also can contain for example:

     config_list = [Configuration.CROSS, Configuration.ELL, Configuration.ENTROPY, Configuration.MOG], which means that
     posterior parameters will be in the objective function gradient. Similarly, including Configuration.HYP, and
     Configuration.LL mean that hyper-parameters, and likelihood parameters will be in the objective function gradient.

    latent_noise : float
     the amount of latent noise that will be added to the kernel.

    exact_ell : boolean
     whether to use exact log likelihood provided by the ``likelihood`` method. If ``exact_ell`` is False, log likelihood
     will be calculated using sampling. The exact likelihood if useful for checking gradients.

    inducing_on_Xs: boolean
     whether to put inducing points randomly on training data. If False, inducing points will be determined using
     clustering.

    n_threads : int
     number of threads used for calculating expected likelihood andi its gradients.

    image : dictionary
     a dictionary containing ``params`` and ``Z``, using which posterior parameters and inducing points will be
     initialized.

    max_X_partition_size : int
     for memory efficiency, the algorithm partitions training data (X), to partitions of size ``max_X_partition_size``,
     and calculated the quantities for each using a separate thread.
    """

    def __init__(self, X, Y,
                 num_inducing,
                 num_mog_comp,
                 likelihood,
                 kernels,
                 n_samples,
                 config_list=None,
                 latent_noise=0,
                 exact_ell=False,
                 inducing_on_Xs=False,
                 n_threads=1,
                 image=None,
                 max_X_partizion_size=3000):

        super(SAVIGP, self).__init__("SAVIGP")
        if config_list is None:
            self.config_list = [Configuration.CROSS, Configuration.ELL, Configuration.ENTROPY]
        else:
            self.config_list = config_list
        self.num_latent_proc = len(kernels)
        """ number of latent processes """

        self.num_mog_comp = num_mog_comp
        """ number of mixture components """

        self.num_inducing = num_inducing
        """ number of inducing points """

        self.MoG = self._get_mog()
        """ posterior distribution """

        self.input_dim = X.shape[1]
        """ dimensionality of input """

        self.kernels = kernels
        """ list containing all the kernels """

        self.cond_likelihood = likelihood
        """ the conditional likelihood function """

        self.X = X
        """ input data. Dimensions: N * D """
        self.Y = Y
        """ output data """

        self.n_samples = n_samples
        """ number of samples used for approximations """

        self.param_names = []
        """ name of the parameters """

        self.latent_noise = latent_noise
        """ amount of latent process """

        self.last_param = None
        """ last parameter that was used in `set_params` """

        self.hyper_params = None
        """ hyper-parameters """

        self.sparse = X.shape[0] != self.num_inducing
        """ bool : whether the model is sparse """

        self.num_hyper_params = self.kernels[0].gradient.shape[0]
        """ number of hyper-parameters in each kernel """

        self.num_like_params = self.cond_likelihood.get_num_params()
        """ number of likelihood parameters """

        self.is_exact_ell = exact_ell
        """ whether to use exact likelihood """

        self.num_data_points = X.shape[0]
        """ number of data points (N) """

        self.n_threads = n_threads
        """ number of threads """

        self.max_x_partition_size = max_X_partizion_size
        """ maximum number of data points to consider for calculations """

        self.cached_ell = None
        """ current expected log likelihood """

        self.cached_ent = None
        """ current entropy """

        self.cached_cross = None
        """ current cross entropy term """

        self.Z = None
        """ position of inducing points. Dimensions: Q * M * D """

        if not image:
            if inducing_on_Xs:
                self.Z, init_m = self._random_inducing_points(X, Y)
            else:
                self.Z, init_m = self._clust_inducing_points(X, Y)
        else:
            self.Z = image['Z']

        # Z is Q * M * D
        self.Kzz = np.array([np.empty((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        """ kernel values for each latent process. Dimension: Q * M * M """

        self.invZ = np.array([np.empty((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        """ inverse of the kernels. Dimension: Q * M * M """

        self.chol = np.array([np.zeros((self.num_inducing, self.num_inducing))] * self.num_latent_proc)
        """ Cholesky decomposition of the kernels. Dimension: Q * M * M """

        self.log_detZ = np.zeros(self.num_latent_proc)
        """ logarithm of determinant of each kernel : log det K(Z[j], Z[j]) """

        # self._sub_parition()
        self.X_partitions, self.Y_partitions, self.n_partitions, self.partition_size = self._partition_data(X, Y)

        np.random.seed(12000)
        self.normal_samples = np.random.normal(0, 1, self.n_samples * self.num_latent_proc * self.partition_size) \
            .reshape((self.num_latent_proc, self.n_samples, self.partition_size))
        """ samples from a normal distribution with mean 0 and variance 1. Dimensions: Q * S * partition_size """

        # uncomment to use sample samples for all data points
        # self.normal_samples = np.random.normal(0, 1, self.n_samples * self.num_latent_proc) \
        # .reshape((self.num_latent_proc, self.n_samples))
        #
        # self.normal_samples = np.repeat(self.normal_samples[:, :, np.newaxis], self.partition_size, 2)

        self.ll = None
        """ evidence lower bound (ELBO) """

        self.grad_ll = None
        """ gradient of evidence lower bound (ELBO) wrt to the parameters """

        if image:
            self.set_all_params(image['params'])
        else:
            self._update_latent_kernel()

            self._update_inverses()

            self.init_mog(init_m)

        self.set_configuration(self.config_list)

    def _partition_data(self, X, Y):
        """
        Partitions ``X`` and ``Y`` into batches of size ``self._max_partition_size()``

        Returns
        -------
        X_partition : list
         a list containing partitions of X. Each partition has dimension: P * D, where P < ``self._max_partition_size()``

        Y_partition : list
         a list containing partitions of Y.

        n_partitions : int
         number of partitions

        partition_size : int
         size of each partition
        """

        X_partitions = []
        Y_partitions = []
        if 0 == (X.shape[0] % self._max_partition_size()):
            n_partitions = X.shape[0] / self._max_partition_size()
        else:
            n_partitions = X.shape[0] / self._max_partition_size() + 1
        if X.shape[0] > self._max_partition_size():
            paritions = np.array_split(np.hstack((X, Y)), n_partitions)
            partition_size = self._max_partition_size()

            for p in paritions:
                X_partitions.append(p[:, :X.shape[1]])
                Y_partitions.append(p[:, X.shape[1]:X.shape[1] + Y.shape[1]])
        else:
            X_partitions = ([X])
            Y_partitions = ([Y])
            partition_size = X.shape[0]
        return X_partitions, Y_partitions, n_partitions, partition_size

    def _sub_parition(self):
        self.partition_size = 50
        inducing_index = np.random.permutation(self.X.shape[0])[:self.partition_size]
        self.X_partitions = []
        self.Y_partitions = []
        self.X_partitions.append(self.X[inducing_index])
        self.Y_partitions.append(self.Y[inducing_index])
        self.cached_ell = None
        self.n_partitions = 1

    def _max_partition_size(self):
        """
        :return: maximum number of elements in each partition
        """
        return self.max_x_partition_size

    def _clust_inducing_points(self, X, Y):
        """
        Determines the position of inducing points using k-means or mini-batch k-means clustering.

        Parameters
        ----------
        X : ndarray
         inputs

        Y : ndarray
         outputs

        Returns
        -------
        Z : ndarray
         position of inducting points. Dimensions: Q * M * M

        init_m : ndarray
          initial value for the mean of posterior distribution which is the mean of Y of data points in
          the corresponding cluster. Dimensions: M * Q
        """

        Z = np.array([np.zeros((self.num_inducing, self.input_dim))] * self.num_latent_proc)
        init_m = np.empty((self.num_inducing, self.num_latent_proc))
        np.random.seed(12000)
        if self.num_inducing == X.shape[0]:
            for j in range(self.num_latent_proc):
                Z[j, :, :] = X.copy()
                init_m[:, j] = Y[:, j].copy()
            for i in range(self.num_inducing):
                init_m[i] = self.cond_likelihood.map_Y_to_f(np.array([Y[i]])).copy()

        else:
            if (self.num_inducing < self.num_data_points / 10) and self.num_data_points > 10000:
                clst = MiniBatchKMeans(self.num_inducing)
            else:
                clst = KMeans(self.num_inducing)
            c = clst.fit_predict(X)
            centers = clst.cluster_centers_
            for zi in range(self.num_inducing):
                yindx = np.where(c == zi)
                if yindx[0].shape[0] == 0:
                    init_m[zi] = self.cond_likelihood.map_Y_to_f(Y).copy()
                else:
                    init_m[zi] = self.cond_likelihood.map_Y_to_f(Y[yindx[0], :]).copy()
            for j in range(self.num_latent_proc):
                Z[j, :, :] = centers.copy()

        return Z, init_m

    def _random_inducing_points(self, X, Y):
        """
        Determines position of the inducing point by random positioning them on the training data.

        Returns
        -------
        Z : ndarray
         position of inducting points. Dimensions: Q * M * M

        init_m : ndarray
          initial value for the mean of posterior distribution which is the Y of the training data over which the
          inducing point is positioned. Dimensions: M * Q

        """

        np.random.seed(12000)
        Z = np.array([np.zeros((self.num_inducing, self.input_dim))] * self.num_latent_proc)
        init_m = np.empty((self.num_inducing, self.num_latent_proc))
        for j in range(self.num_latent_proc):
            if self.num_inducing == X.shape[0]:
                inducing_index = range(self.X.shape[0])
            else:
                inducing_index = np.random.permutation(X.shape[0])[:self.num_inducing]
            Z[j, :, :] = X[inducing_index].copy()
        for i in range(self.num_inducing):
            init_m[i] = self.cond_likelihood.map_Y_to_f(np.array([Y[inducing_index[i]]])).copy()

        return Z, init_m

    def _update_latent_kernel(self):
        """
        Updates kernels by adding a latent noise to each kernel.
        """

        self.kernels_latent = []
        for j in range(len(self.kernels)):
            self.kernels_latent.append(self.kernels[j] + GPy.kern.White(self.X.shape[1], variance=self.latent_noise))
        self.hypers_changed = True

    def init_mog(self, init_m):
        """
        Initialised MoG (posterior distribution).

        Parameters
        ----------
        init_m : ndarray
         a matrix of size M * Q, which is mean of posterior for each latent process
        """

        for j in range(self.num_latent_proc):
            self.MoG.update_mean(j, init_m[:, j])

    def rand_init_mog(self):
        """
        Randomly initialises the posterior distribution
        """
        self.MoG.random_init()

    def _get_mog(self):
        """
        :returns: the MoG used for representing the posterior. It should be implemented by sub-classes.
        """

        raise NotImplementedError

    def get_param_names(self):
        """
        :returns: an array containing name of the parameters of the class given the current configuration.
        Useful for example when checking gradients.
        """

        if Configuration.MoG in self.config_list:
            self.param_names += ['m'] * self.MoG.get_m_size() + ['s'] * \
                                                                self.MoG.get_s_size() + ['pi'] * self.num_mog_comp

        if Configuration.HYPER in self.config_list:
            self.param_names += ['k'] * self.num_latent_proc * self.num_hyper_params

        if Configuration.LL in self.config_list:
            self.param_names += ['ll'] * self.num_like_params

        if Configuration.INDUCING in self.config_list:
            self.param_names += ['indu'] * self.num_latent_proc * self.num_inducing * self.input_dim

        return self.param_names

    def get_all_param_names(self):
        param_names = []
        param_names += ['m'] * self.MoG.get_m_size() + ['s'] * \
                                                       self.MoG.get_s_size() + ['pi'] * self.num_mog_comp
        param_names += ['k'] * self.num_latent_proc * self.num_hyper_params
        param_names += ['ll'] * self.num_like_params

        return param_names

    def image(self):
        """
        :returns: a dictionary containing an image of the class which can be used to init the model from.
        """

        return {'params': self.get_all_params(), 'Z': self.Z}

    def _update_inverses(self):
        """
        Calculates and stores kernel, and its inverses.
        """

        for j in range(self.num_latent_proc):
            self.Kzz[j, :, :] = self.kernels_latent[j].K(self.Z[j, :, :])
            self.chol[j, :, :] = jitchol(self.Kzz[j, :, :])
            self.invZ[j, :, :] = inv_chol(self.chol[j, :, :])
            self.log_detZ[j] = pddet(self.chol[j, :, :])
        self.hypers_changed = False
        self.inducing_changed = False

    def kernel_hyp_params(self):
        """
        :return: a matrix of dimension Q * |H|, containing hyper-parameters of all kernels.
        """

        hyper_params = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            hyper_params[j] = self.kernels[j].param_array[:].copy()
        return hyper_params

    def _update(self):
        """
        Updates objective function and its gradients under current configuration and stores them in the corresponding
        variables for future uses.
        """

        self.ll = 0

        if Configuration.MoG in self.config_list:
            grad_m = np.zeros((self.MoG.m_dim()))
            grad_s = np.zeros((self.MoG.get_s_size()))
            grad_pi = np.zeros((self.MoG.pi_dim()))

        if Configuration.HYPER in self.config_list:
            self.hyper_params = self.kernel_hyp_params()
            grad_hyper = np.zeros(self.hyper_params.shape)

        if Configuration.INDUCING in self.config_list:
            grad_inducing = np.zeros((self.num_latent_proc, self.num_inducing, self.input_dim))

        if self.hypers_changed or self.inducing_changed:
            self._update_inverses()

        if Configuration.ENTROPY in self.config_list or (self.cached_ent is None):
            self.cached_ent = self._l_ent()
            if Configuration.MoG in self.config_list:
                grad_m += self._d_ent_d_m()
                grad_s += self._transformed_d_ent_d_S()
                grad_pi += self._d_ent_d_pi()
            if Configuration.HYPER in self.config_list:
                grad_hyper += self._dent_dhyper()
        self.ll += self.cached_ent

        if Configuration.CROSS in self.config_list or (self.cached_cross is None):
            xcross, xdcorss_dpi = self._cross_dcorss_dpi(0)
            self.cached_cross = xcross
            if Configuration.MoG in self.config_list:
                grad_m += self._dcorss_dm()
                grad_s += self.transform_dcorss_dS()
                grad_pi += xdcorss_dpi
            if Configuration.HYPER in self.config_list:
                grad_hyper += self._dcross_dhyper()
            if Configuration.INDUCING in self.config_list:
                grad_inducing += self._dcross_dinducing()

        self.ll += self.cached_cross

        if Configuration.ELL in self.config_list:
            xell, xdell_dm, xdell_ds, xdell_dpi, xdell_hyper, xdell_dll, xdell_dinduc = self._ell()
            self.cached_ell = xell
            self.ll += xell
            if Configuration.MoG in self.config_list:
                grad_m += xdell_dm
                grad_s += self.MoG.transform_S_grad(xdell_ds)
                grad_pi += xdell_dpi
            if Configuration.HYPER in self.config_list:
                grad_hyper += xdell_hyper
            if Configuration.INDUCING in self.config_list:
                grad_inducing += xdell_dinduc

        self.grad_ll = np.array([])
        if Configuration.MoG in self.config_list:
            self.grad_ll = np.hstack([grad_m.flatten(),
                                      grad_s,
                                      self.MoG.transform_pi_grad(grad_pi),
            ])

        if Configuration.HYPER in self.config_list:
            self.grad_ll = np.hstack([self.grad_ll,
                                      (grad_hyper.flatten()) * self.hyper_params.flatten()
                                      ])

        if Configuration.LL in self.config_list:
            self.grad_ll = np.hstack([self.grad_ll,
                                      xdell_dll
                                      ])

        if Configuration.INDUCING in self.config_list:
            self.grad_ll = np.hstack([self.grad_ll,
                                      grad_inducing.flatten()
                                      ])

    def set_configuration(self, config_list):
        self.config_list = config_list
        self._clear_cache()
        self._update()

    def _clear_cache(self):
        self.cached_ell = None
        self.cached_cross = None
        self.cached_ent = None

    def set_params(self, p):
        """
        Sets the internal parameters of the model.

        :param p: input parameters. ``p`` should contain parameters specified in the configuration.
        """
        self.last_param = p
        index = 0
        if Configuration.MoG in self.config_list:
            self.MoG.update_parameters(p[:self.MoG.num_parameters()])
            index = self.MoG.num_parameters()
        if Configuration.HYPER in self.config_list:
            self.hyper_params = np.exp(p[index:(index + self.num_latent_proc * self.num_hyper_params)].
                                       reshape((self.num_latent_proc, self.num_hyper_params)))
            for j in range(self.num_latent_proc):
                self.kernels[j].param_array[:] = self.hyper_params[j]
            index += self.num_latent_proc * self.num_hyper_params
            self._update_latent_kernel()

        if Configuration.LL in self.config_list:
            self.cond_likelihood.set_params(p[index:index + self.num_like_params])
            index += self.num_like_params

        if Configuration.INDUCING in self.config_list:
            self.Z = p[index:].reshape((self.num_latent_proc, self.num_inducing, self.input_dim))
            self.inducing_changed = True

        self._update()

    def set_all_params(self, p):
        """
        Sets all the parameters of the model (not only those specified by the configuration).
        """
        self.last_param = p
        self.MoG.update_parameters(p[:self.MoG.num_parameters()])
        index = self.MoG.num_parameters()
        self.hyper_params = np.exp(p[index:(index + self.num_latent_proc * self.num_hyper_params)].
                                   reshape((self.num_latent_proc, self.num_hyper_params)))
        for j in range(self.num_latent_proc):
            self.kernels[j].param_array[:] = self.hyper_params[j]
        index += self.num_latent_proc * self.num_hyper_params
        self._update_latent_kernel()
        self.cond_likelihood.set_params(p[index:index + self.num_like_params])
        self._clear_cache()
        self._update()

    def get_params(self):
        """
        Returns parameters of the model according to the configuration.
        """
        params = np.array([])
        if Configuration.MoG in self.config_list:
            params = self.MoG.parameters
        if Configuration.HYPER in self.config_list:
            params = np.hstack([params, np.log(self.hyper_params.flatten())])
        if Configuration.LL in self.config_list:
            params = np.hstack([params, self.cond_likelihood.get_params()])
        if Configuration.INDUCING in self.config_list:
            params = np.hstack([params, self.Z.flatten()])
        return params.copy()

    def get_posterior_params(self):
        return self.MoG.get_m_S_params()

    def get_all_params(self):
        """
        Returns all internal parameters of the model.
        """
        params = self.MoG.parameters
        params = np.hstack([params, np.log(self.kernel_hyp_params().flatten())])
        params = np.hstack([params, self.cond_likelihood.get_params()])
        params = np.hstack([params, self.Z.flatten()])
        return params

    def log_likelihood(self):
        return self.ll

    def _log_likelihood_gradients(self):
        return self.grad_ll

    def _A(self, j, K):
        """
        calculates A for latent process ``j`` (see paper for the definition of A)
        """
        return cho_solve((self.chol[j, :, :], True), K).T

    def _Kdiag(self, p_X, K, A, j):
        """
        calculates diagonal terms of K_tilda for latent process ``j`` (see paper for the definition of Ktilda)
        """
        return self.kernels_latent[j].Kdiag(p_X) - mdiag_dot(A, K)


    def _b(self, k, j, Aj, Kzx):
        """
        calculating [b_k]j for latent process ``j`` for all ``k``

        :returns: an ndarray of dimension N * 1

        """
        return mdot(Aj, self.MoG.m[k, j, :].T)

    def _sigma(self, k, j, Kj, Aj, Kzx):
        """
        calculates [sigma_k]j,j for latent process ``j`` and component ``k``

        :returns: an ndarray of dimension N * 1
        """
        return Kj + self.MoG.aSa(Aj, k, j)

    # @profile
    def _get_A_K(self, p_X):
        """
        Calculates A, Ktilda, and Kzx for partition ``p_X``

        Parameters
        ----------
        p_X : ndarray
         input of dimension P * D

        Returns
        -------
        A : ndarray
         dimensions: Q * P * M

        Kzx : ndarray
         dimensions: Q * M * P

        K : ndarray
         dimensions: Q * P
        """

        A = np.empty((self.num_latent_proc, p_X.shape[0], self.num_inducing))
        K = np.empty((self.num_latent_proc, p_X.shape[0]))
        Kzx = np.empty((self.num_latent_proc, self.num_inducing, p_X.shape[0]))
        for j in range(self.num_latent_proc):
            Kzx[j, :, :] = self.kernels_latent[j].K(self.Z[j, :, :], p_X)
            A[j] = self._A(j, Kzx[j, :, :])
            K[j] = self._Kdiag(p_X, Kzx[j, :, :], A[j], j)
        return A, Kzx, K

    def _dell_ds(self, k, j, cond_ll, A, n_sample, sigma_kj):
        """
        Returns gradient of ell wrt to the posterior covariance for component ``k`` and latent process ``j``.
        """
        raise Exception("method not implemented")

    def _ell(self):
        """
        Calculates ell and its gradient for each partition of data, and adds them together to build the ell and gradients
        over all data. Each partition
        is ran in a separate thread, with maximum of ``self.n_threads`` threads.
        """

        threadLimiter = threading.BoundedSemaphore(self.n_threads)

        lock = threading.Lock()

        class MyThread(threading.Thread):
            def __init__(self, savigp, X, Y, output):
                super(MyThread, self).__init__()
                self.output = output
                self.X = X
                self.Y = Y
                self.savigp = savigp

            def run(self):
                threadLimiter.acquire()
                try:
                    self.Executemycode()
                finally:
                    threadLimiter.release()

            def Executemycode(self):
                out = self.savigp._parition_ell(self.X, self.Y)
                lock.acquire()
                try:
                    if not self.output:
                        self.output.append(list(out))
                    else:
                        for o in range(len(out)):
                            self.output[0][o] += out[o]
                finally:
                    lock.release()


        total_out = []
        threads = []
        for p in range(0, self.n_partitions):
            t = MyThread(self, self.X_partitions[p], self.Y_partitions[p], total_out)
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()

        return total_out[0]

    def _parition_ell(self, X, Y):
        """
        calculating expected log-likelihood, and it's derivatives for input ``X`` and output ``Y``.

        Returns
        -------
        total_ell : float
         expected log likelihood, calculated either using sampling or exact likelihood

        d_ell_dm : ndarray
         gradient of ell wrt to the mean posterior. Dimensions: K, Q, M; where K is the number of
         mixture components.

        d_ell_ds : ndarray
         gradient of ell wrt to the covariance posterior. Dimensions: K, Q, ``self.MoG.S_dim()``

        d_ell_dPi : ndarray
         gradient of ell wrt to mixture component weights. Dimensions: Q * 1

        d_ell_d_hyper : ndarray
         gradient wrt to the hyper-parameters. Dimensions: Q * |H|; where |H| is the number of
         hyper-parameters.

        d_ell_d_ll : ndarray
         gradient wrt to the likelihood parameters. Dimensions: |L|; where |L| is the number of
         likelihood parameters.
        """

        # print 'ell started'
        total_ell = self.cached_ell
        d_ell_dm = np.zeros((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        d_ell_ds = np.zeros((self.num_mog_comp, self.num_latent_proc) + self.MoG.S_dim())
        d_ell_dPi = np.zeros(self.num_mog_comp)
        if Configuration.HYPER in self.config_list:
            d_ell_d_hyper = np.zeros((self.num_latent_proc, self.num_hyper_params))
        else:
            d_ell_d_hyper = 0

        if Configuration.INDUCING in self.config_list:
            d_ell_d_induc = np.zeros((self.num_latent_proc, self.num_inducing, self.input_dim))

        if Configuration.LL in self.config_list:
            d_ell_d_ll = np.zeros(self.num_like_params)
        else:
            d_ell_d_ll = 0

        if Configuration.MoG in self.config_list or \
                        Configuration.LL in self.config_list or \
                        self.cached_ell is None or \
                self.calculate_dhyper() or \
                Configuration.INDUCING in self.config_list:
            total_ell = 0
            A, Kzx, K = self._get_A_K(X)
            mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0]))
            sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0]))
            F = np.empty((self.n_samples, X.shape[0], self.num_latent_proc))
            for k in range(self.num_mog_comp):
                for j in range(self.num_latent_proc):
                    norm_samples = self.normal_samples[j, :, :X.shape[0]]
                    mean_kj[k, j] = self._b(k, j, A[j], Kzx[j].T)
                    sigma_kj[k, j] = self._sigma(k, j, K[j], A[j], Kzx[j].T)
                    F[:, :, j] = (norm_samples * np.sqrt(sigma_kj[k, j]))
                    F[:, :, j] = F[:, :, j] + mean_kj[k, j]
                cond_ll, grad_ll = self.cond_likelihood.ll_F_Y(F, Y)
                for j in range(self.num_latent_proc):
                    norm_samples = self.normal_samples[j, :, :X.shape[0]]
                    m = self._average(cond_ll, norm_samples / np.sqrt(sigma_kj[k, j]), True)
                    d_ell_dm[k, j] = self._proj_m_grad(j, mdot(m, Kzx[j].T)) * self.MoG.pi[k]
                    d_ell_ds[k, j] = self._dell_ds(k, j, cond_ll, A, sigma_kj, norm_samples)
                    if self.calculate_dhyper():
                        ds_dhyp = self._dsigma_dhyp(j, k, A[j], Kzx, X)
                        db_dhyp = self._db_dhyp(j, k, A[j], X)
                        for h in range(self.num_hyper_params):
                            d_ell_d_hyper[j, h] += -1. / 2 * self.MoG.pi[k] * (
                                self._average(cond_ll,
                                              np.ones(cond_ll.shape) / sigma_kj[k, j] * ds_dhyp[:, h] +
                                              -2. * norm_samples / np.sqrt(sigma_kj[k, j]) * db_dhyp[:, h]
                                              - np.square(norm_samples) / sigma_kj[k, j] * ds_dhyp[:, h], True)).sum()

                    if Configuration.INDUCING in self.config_list:
                        db_dinduc = self._db_dinduc(j, k, A[j], X)
                        ds_dinduc = self._dsigma_dinduc(j, k, A[j], Kzx, X)
                        ds_dinduc = ds_dinduc.reshape(ds_dinduc.shape[0], ds_dinduc.shape[1] * ds_dinduc.shape[2])
                        db_dinduc = db_dinduc.reshape(db_dinduc.shape[0], db_dinduc.shape[1] * db_dinduc.shape[2])
                        tmp_induc = np.empty(ds_dinduc.shape[1])
                        for q in range(ds_dinduc.shape[1]):
                            tmp_induc[q] += -1. / 2 * self.MoG.pi[k] * (
                                self._average(cond_ll,
                                              np.ones(cond_ll.shape) / sigma_kj[k, j] * ds_dinduc[:, q] +
                                              -2. * norm_samples / np.sqrt(sigma_kj[k, j]) * db_dinduc[:, q]
                                              - np.square(norm_samples) / sigma_kj[k, j] * ds_dinduc[:, q], True)).sum()
                        d_ell_d_induc[j, :, :] = tmp_induc.reshape((self.num_inducing, self.input_dim))

                sum_cond_ll = cond_ll.sum() / self.n_samples
                total_ell += sum_cond_ll * self.MoG.pi[k]
                d_ell_dPi[k] = sum_cond_ll

                if Configuration.LL in self.config_list:
                    d_ell_d_ll += self.MoG.pi[k] * grad_ll.sum() / self.n_samples

            if self.is_exact_ell:
                total_ell = 0
                for n in range(len(X)):
                    for k in range(self.num_mog_comp):
                        total_ell += self.cond_likelihood.ell(np.array(mean_kj[k, :, n]), np.array(sigma_kj[k, :, n]),
                                                              Y[n, :]) * self.MoG.pi[k]

        return total_ell, d_ell_dm, d_ell_ds, d_ell_dPi, d_ell_d_hyper, d_ell_d_ll, d_ell_d_induc

    def _average(self, condll, X, variance_reduction):
        """
        calculates (condll * X).mean(axis=1) using variance reduction method.

        number of control variables = number of samples / 10

        Parameters
        ----------
        condll : ndarray
         dimensions: s * N

        X : ndarray
         dimensions: s * N

        Returns
        -------
        :returns: a matrix of dimension N
        """
        if variance_reduction:
            X = X.T
            condll = condll.T
            cvsamples = self.n_samples / 10
            pz = X[:, 0:cvsamples]
            py = np.multiply(condll[:, 0:cvsamples], pz)
            above = np.multiply((py.T - py.mean(1)), pz.T).sum(axis=0) / (cvsamples - 1)
            below = np.square(pz).sum(axis=1) / (cvsamples - 1)
            cvopt = np.divide(above, below)
            cvopt = np.nan_to_num(cvopt)

            grads = np.multiply(condll, X) - np.multiply(cvopt, X.T).T
        else:
            grads = np.multiply(condll.T, X.T)
        return grads.mean(axis=1)

    def calculate_dhyper(self):
        """
        whether to calculate gradients of ell wrt to the hyper parameters. Note that when the model is not sparse
        gradients of ell wrt to the hyper-parameters are zero.
        """
        return self.sparse and Configuration.HYPER in self.config_list

    def _proj_m_grad(self, j, dl_dm):
        r"""
        Projects gradients to the kernel space, i.e,. returns

        :returns K^-1 dl\\dm
        """
        return cho_solve((self.chol[j, :, :], True), dl_dm)

    def _dsigma_dhyp(self, j, k, Aj, Kzx, X):
        """
        calculates gradient of ``sigma`` for component ``k`` and latent process ``j`` wrt to the
        hyper parameters. ``sigma`` is as follows:

         sigma = Kj(X, X) - Aj Kzx + Aj Skj Aj

        """
        return self.kernels[j].get_gradients_Kdiag(X) \
               - self.kernels[j].get_gradients_AK(Aj, X, self.Z[j]) + \
               2. * self.dA_dhyper_mult_x(j, X, Aj,
                                          self.MoG.Sa(Aj.T, k, j) - Kzx[j] / 2)

    def _dsigma_dinduc(self, j, k, Aj, Kzx, X):
        """
        calculates gradient of ``sigma`` for component ``k`` and latent process ``j`` wrt to the
        location of inducing points. ``sigma`` is as follows:

         sigma = Kj(X, X) - Aj Kzx + Aj Skj Aj

        """
        return -self.kernels[j].get_gradients_X_AK(Aj.T, self.Z[j], X) + \
               2. * self.dA_dinduc_mult_x(j, X, Aj,
                                          self.MoG.Sa(Aj.T, k, j) - Kzx[j] / 2)

    def _db_dhyp(self, j, k, Aj, X):
        """
        calculates gradients of ``b`` for latent process ``j`` and component ``k`` wrt to the
        hyper-parameters. ``b`` is as follows:

         b = Aj mkj

        """
        return self.dA_dhyper_mult_x(j, X, Aj, np.repeat(self.MoG.m[k, j][:, np.newaxis], X.shape[0], axis=1))

    def _db_dinduc(self, j, k, Aj, X):
        """
        calculates gradients of ``b`` for latent process ``j`` and component ``k`` wrt to the
        location of inducing points. ``b`` is as follows:

         b = Aj mkj

        """
        return self.dA_dinduc_mult_x(j, X, Aj, np.repeat(self.MoG.m[k, j][:, np.newaxis], X.shape[0], axis=1))

    def dA_dhyper_mult_x(self, j, X, Aj, m):
        r"""

        Assume:

         dfn \\ dH = dAn \\ dH * m

        where:

         dAn \\ dH = (dK(X[n, :], Z[j]) \\ dH  - An d K(Z[j], Z[j]) \\ dH) K(Z[j], Z[j]) ^ -1

        and
         An = A[n, :]

        then this function returns
         dfn \\ dH for all `n`s:

        :returns dF \\dH where (dF \\dH)[n] = dfn \\ dH
        """
        w = mdot(self.invZ[j], m)
        return self.kernels[j].get_gradients_AK(w.T, X, self.Z[j]) - \
               self.kernels[j].get_gradients_SKD(Aj, w, self.Z[j])


    def dA_dinduc_mult_x(self, j, X, Aj, m):
        r"""

        Assume:

         dfn \\ dH = dAn \\ dH * m

        where:

         dAn \\ dH = (dK(X[n, :], Z[j]) \\ dH  - An d K(Z[j], Z[j]) \\ dH) K(Z[j], Z[j]) ^ -1

        and
         An = A[n, :]

        then this function returns
         dfn \\ dH for all `n`s:

        :returns dF \\dH where (dF \\dH)[n] = dfn \\ dH
        """
        w = mdot(self.invZ[j], m)
        return self.kernels[j].get_gradients_X_AK(w, self.Z[j], X) - \
               self.kernels[j].get_gradients_X_SKD(Aj, w, self.Z[j])

    def _dcorss_dm(self):
        r"""
        calculates d corss \\ dm

        :returns a matrix of dimension K * Q * M, where K is the number of mixture components
        """

        dcdm = np.empty((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        for j in range(self.num_latent_proc):
            dcdm[:, j, :] = -cho_solve((self.chol[j, :, :], True), self.MoG.m[:, j, :].T).T * self.MoG.pi[:, np.newaxis]
        return dcdm

    def _dcross_ds(self):
        """
        calculates gradient of the cross term of ELBO wrt to the posterior covariance and transforms the covariance to
        the raw space.

        Returns
        -------
        output : ndarray
         dim(output) = K * Q * ``self.MoG.get_sjk_size()``
        """

        dc_ds = np.empty((self.num_mog_comp, self.num_latent_proc, self.MoG.get_sjk_size()))
        for j in range(self.num_latent_proc):
            dc_ds[:, j] = -1. / 2 * np.array(
                [self.MoG.dAinvS_dS(self.chol[j, :, :], k, j) * self.MoG.pi[k] for k in range(self.num_mog_comp)])
        return dc_ds

    def transform_dcorss_dS(self):
        r"""
        calculates dcross \\ dL, where L is the representation of the covariance matrix. For example, in the case of
        full posterior covariance matrix, L is the lower triangular elements of the Cholesky decomposition of
        posterior covariance matrix.
        """

        return self._dcross_ds().flatten()

    # def _cross_dcorss_dpi(self, N):
    # """
    #     calculating L_corss by pi_k, and also calculates the cross term
    #     :returns d cross / d pi, cross
    #     """
    #     cross = 0
    #     d_pi = np.zeros(self.num_mog_comp)
    #     for j in range(self.num_latent_proc):
    #         for k in range(self.num_mog_comp):
    #             d_pi[k] += \
    #                 N * math.log(2 * math.pi) + \
    #                 self.log_detZ[j] + \
    #                 mdot(self.MoG.m[k, j, :].T, cho_solve((self.chol[j, :, :], True), self.MoG.m[k, j, :])) + \
    #                 self.MoG.tr_A_mult_S(self.chol[j, :, :], k, j)
    #     for k in range(self.num_mog_comp):
    #         cross += self.MoG.pi[k] * d_pi[k]
    #
    #     d_pi *= -1. / 2
    #     cross *= -1. / 2
    #     return cross, d_pi

    def _cross_dcorss_dpi(self, N):
        """
        Returns
        --------
        cross : float
         the cross term of ELBO

        d_pi : ndarray
         dcross \\ dpi
        """
        cross = 0
        d_pi = np.zeros(self.num_mog_comp)
        for j in range(self.num_latent_proc):
            for k in range(self.num_mog_comp):
                a = solve_triangular(self.chol[j, :, :], self.MoG.m[k, j, :], lower=True)
                d_pi[k] += \
                    N * math.log(2 * math.pi) + \
                    self.log_detZ[j] + \
                    + np.dot(a, a.T) + \
                    self.MoG.tr_AinvS(self.chol[j, :, :], k, j)
        for k in range(self.num_mog_comp):
            cross += self.MoG.pi[k] * d_pi[k]

        d_pi *= -1. / 2
        cross *= -1. / 2
        return cross, d_pi

    def _dcross_K(self, j):
        r"""
        Gradient of the cross term of ELBO wrt to the kernel of latent process ``j``.

        Returns
        -------
        :returns: dcross \\ dK(Z[j], Z[j]). Dimensions: M * M
        """

        dc_dK = np.zeros((self.num_inducing, self.num_inducing))
        for k in range(self.num_mog_comp):
            dc_dK += -0.5 * self.MoG.pi[k] * (self.invZ[j]
                                              - mdot(self.invZ[j], self.MoG.mmTS(k, j), self.invZ[j])
                                              )
        return dc_dK

    def _dcross_dhyper(self):
        r"""
        Gradient of the cross term of ELBO wrt to the hyper-parameters (H).

        Returns
        -------
        :returns: dcross \\ dH. Dimensions: Q * |H|
        """

        dc_dh = np.empty((self.num_latent_proc, self.num_hyper_params))
        for j in range(self.num_latent_proc):
            self.kernels_latent[j].update_gradients_full(self._dcross_K(j), self.Z[j])
            dc_dh[j] = self.kernels[j].gradient.copy()

        return dc_dh

    def _dcross_dinducing(self):
        r"""
        Gradient of the cross term of ELBO wrt to the location of inducing points (Z).

        Returns
        -------
        :returns: dcross \\ dH. Dimensions: Q * M * D
        """

        dc_dindu = np.empty((self.num_latent_proc, self.num_inducing, self.input_dim))
        for j in range(self.num_latent_proc):
            dc_dindu[j] = self.kernels_latent[j].gradients_X(self._dcross_K(j), self.Z[j])

        return dc_dindu

    def _dent_dhyper(self):
        r"""
        Gradient of the entropy term of ELBO wrt to the hyper-parameters, which is always zero in the case of
        this model.

        Returns
        -------
        :returns: a zero matrix of dimensions: Q * |H|
        """

        return np.zeros((self.num_latent_proc, self.num_hyper_params))

    def _d_ent_d_m_kj(self, k, j):
        r"""
        Gradient of the entropy term of ELBO wrt to the posterior mean for component ``k`` and latent process ``j``.

        Returns
        -------
        d ent \\ dm[k,j]. Dimensions: M * 1
        """
        m_k = np.zeros(self.num_inducing)
        for l in range(self.num_mog_comp):
            m_k += self.MoG.pi[k] * self.MoG.pi[l] * (np.exp(self.log_N_kl[k, l] - self.log_z[k]) +
                                                      np.exp(self.log_N_kl[k, l] - self.log_z[l])) * \
                   (self.MoG.C_m(j, k, l))
        return m_k

    def _d_ent_d_m(self):
        r"""
        Calculates gradients of the entropy term of ELBO wrt to the posterior mean.

        Returns
        -------
        :returns: d ent \\ dm. Dimensions: K * Q * M
        """
        raise NotImplementedError

    def _d_ent_d_pi(self):
        r"""
        Calculates gradient of the entropy term of ELBO wrt to the mixture weights (p)

        Returns
        -------
        dpi : ndarray
         d ent \\ dpi. Dimensions : K * 1
        """

        raise NotImplementedError

    def _l_ent(self):
        """ returns entropy term of the ELBO. """
        raise NotImplementedError

    def _transformed_d_ent_d_S(self):
        r"""
        Calculates gradient of the entropy term wrt to the posterior covariance, and transforms it to the raw space
        and returns a flatten array.
        """
        raise NotImplementedError

    def _predict_comp(self, Xs, Ys):
        """
        Predicts output for test points ``Xs``, and also calculates NLPD if ``Ys`` is provided. The prediction is
        made for each mixture component separately.

        P(Ys|Xs) = \integral P(Ys|f) N(f|b, sigma) df

        Parameters
        ----------
        Xs : ndarray
         test points. Dimension : N * D, where N is the number of test points.

        Ys : ndarray (or None)
         output at test points. Dimensions : N * O

        Returns
        -------
        predicted_mu : ndarray
         returns E[P(Ys|Xs)]. Dimensions N * K * O, where K is the number of mixture components

        predicted_var : ndarray
         returns var[P(Ys|Xs)]. Dimensions N * K * O, where K is the number of mixture components

        NLPD : ndarray
         returns -log [P(Ys|Xs)]. Dimensions N * |NLPD|, wheree |NLPD| is the number of NLPDs returned
         by the likelihood. |NLPD| is generally 1, but likelihood is allowed to return multiple NLPD for example
         for each output in the case of multi-output models.
        """

        A, Kzx, K = self._get_A_K(Xs)

        predicted_mu = np.empty((Xs.shape[0], self.num_mog_comp, self.cond_likelihood.output_dim()))
        predicted_var = np.empty((Xs.shape[0], self.num_mog_comp, self.cond_likelihood.output_dim()))
        nlpd = None
        if not (Ys is None):
            nlpd = np.empty((Xs.shape[0], self.cond_likelihood.nlpd_dim(), self.num_mog_comp))

        mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0]))
        sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0]))
        for k in range(self.num_mog_comp):
            for j in range(self.num_latent_proc):
                mean_kj[k, j] = self._b(k, j, A[j], Kzx[j].T)
                sigma_kj[k, j] = self._sigma(k, j, K[j], A[j], Kzx[j].T)

            if not (Ys is None):
                predicted_mu[:, k, :], predicted_var[:, k, :], nlpd[:, :, k] = \
                    self.cond_likelihood.predict(mean_kj[k, :].T, sigma_kj[k, :].T, Ys, self)
            else:
                predicted_mu[:, k, :], predicted_var[:, k, :], _ = \
                    self.cond_likelihood.predict(mean_kj[k, :].T, sigma_kj[k, :].T, Ys, self)

        return predicted_mu, predicted_var, -logsumexp(nlpd, 2, self.MoG.pi)

    def predict(self, Xs, Ys=None):
        """
        Makes prediction for test points ``Xs``, and calculates NLPD for ``Ys`` if it is provided.

        Parameters
        ----------
        Xs : ndarray
         test points. Dimensions N * D, where N is the number of test points, and D is input dimensionality.

        Ys : ndarray (optional)
         output at test points. Dimensions N * O, where O is the output dimensionality.

        Returns
        -------
        mean : ndarray
         mean of the prediction at the test point. Dimensions : N * O

        var : ndarray
         variance of the prediction at the test point. Dimensions : N * O (?)

        NLPD : ndarray
         NLPD at the test points in the case that ``Ys`` is provided.


        """

        X_partitions, Y_partitions, n_partitions, partition_size = self._partition_data(Xs, Ys)

        mu, var, nlpd = self._predict_comp(X_partitions[0], Y_partitions[0])
        for p in range(1, len(X_partitions)):
            p_mu, p_var, p_nlpd = self._predict_comp(X_partitions[p], Y_partitions[p])
            mu = np.concatenate((mu, p_mu), axis=0)
            var = np.concatenate((var, p_var), axis=0)
            nlpd = np.concatenate((nlpd, p_nlpd), axis=0)

        predicted_mu = np.average(mu, axis=1, weights=self.MoG.pi)
        predicted_var = np.average(mu ** 2, axis=1, weights=self.MoG.pi) \
                        + np.average(var, axis=1, weights=self.MoG.pi) - predicted_mu ** 2

        return predicted_mu, predicted_var, nlpd
