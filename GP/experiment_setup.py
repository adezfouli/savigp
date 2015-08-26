import Image

__author__ = 'AT'

from data_source import DataSource
from model_learn import ModelLearn
from likelihood import *
from data_transformation import *
from ExtRBF import ExtRBF


class ExperimentSetup:
    """
    Sets up settings for running each experiment.
    """

    def __init__(self):
        pass

    @staticmethod
    def boston_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.boston_data()
        d = data[config['run_id'] - 1]
        names = []
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'boston'
        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 1, True)
        # gaussian_sigma = np.var(Ytrain)/4 + 1e-4
        gaussian_sigma = 1.0
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        cond_ll = UnivariateGaussian(np.array(gaussian_sigma))
        num_samples = 2000

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['hyp', 'mog', 'll'], MeanTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200))
        return names

    @staticmethod
    def wisconsin_breast_cancer_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.wisconsin_breast_cancer_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'breast_cancer'

        # uncomment these lines to use softmax
        # kernel = Experiments.get_kernels(Xtrain.shape[1], 2, False)
        # Ytrain = np.array([(Ytrain[:,0] + 1) / 2, (-Ytrain[:,0] + 1) / 2]).T
        # Ytest = np.array([(Ytest[:,0] + 1) / 2, (-Ytest[:,0] + 1) / 2]).T
        # cond_ll = SoftmaxLL(2)

        # uncomment these lines to use logistic
        cond_ll = LogisticLL()
        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 1, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200))
        return names


    @staticmethod
    def mining_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.mining_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'mining'
        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 1, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = LogGaussianCox(math.log(191. / 811))
        kernel[0].variance = 1.0
        kernel[0].lengthscale = 13516.

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog'], IdentityTransformation, True,
                                 config['log_level'], True, latent_noise=0.001,
                                 opt_per_iter={'mog': 15000, 'hyp': 25, 'll': 25},
                                 max_iter=1))
        return names


    @staticmethod
    def USPS_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.USPS_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'USPS'
        kernel = [ExtRBF(Xtrain.shape[1], variance=2, lengthscale=np.array((4.,)), ARD=False) for j in range(3)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = SoftmaxLL(3)

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=300))


    @staticmethod
    def abalone_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.abalone_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'abalone'
        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 1, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll = WarpLL(np.array([-2.0485, 1.7991, 1.5814]),
                         np.array([2.7421, 0.9426, 1.7804]),
                         np.array([0.1856, 0.7024, -0.7421]),
                         np.log(0.1))

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp', 'll'], MinTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200))


    @staticmethod
    def creep_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.creep_data()

        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'creep'
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 1, True)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll = WarpLL(np.array([3.8715, 3.8898, 2.8759]),
                         np.array([1.5925, -1.3360, -2.0289]),
                         np.array([0.7940, -4.1855, -3.0289]),
                         np.log(0.01))

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp', 'll'], MinTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200))


    @staticmethod
    def MNIST_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.mnist_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'mnist'

        # uncomment these lines to delete unused features
        # features_rm = np.array([])
        # for n in range(Xtrain.shape[1]):
        # if Xtrain[:, n].sum() ==0:
        #         features_rm = np.append(features_rm, n)
        # Xtrain = np.delete(Xtrain, features_rm.astype(int), 1)
        # Xtest = np.delete(Xtest, features_rm.astype(int), 1)


        # uncomment these lines to change the resolution
        # res = 13
        # current_res = int(np.sqrt(Xtrain.shape[1]))
        # X_train_resized = np.empty((Xtrain.shape[0], res * res))
        # X_test_resized = np.empty((Xtest.shape[0], res * res))
        # for n in range(Xtrain.shape[0]):
        #     im = Image.fromarray(Xtrain[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_train_resized[n] = np.array(im).flatten()
        #
        # for n in range(Xtest.shape[0]):
        #     im = Image.fromarray(Xtest[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_test_resized[n] = np.array(im).flatten()
        #
        #
        # Xtrain = X_train_resized
        # Xtest = X_test_resized

        kernel = [ExtRBF(Xtrain.shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(10)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = SoftmaxLL(10)

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000

        image = None
        if 'image' in config.keys():
            image = config['image']

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, False,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 50, 'hyp': 10},
                                 max_iter=300, n_threads=n_threads, ftol=10,
                                 model_image_file=image, partition_size=partition_size))


    @staticmethod
    def MNIST_binary_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.mnist_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain_full = d['train_Y']
        Xtest = d['test_X']
        Ytest_full = d['test_Y']
        name = 'mnist_binary'

        # uncomment these lines to delete unused features
        # features_rm = np.array([])
        # for n in range(Xtrain.shape[1]):
        # if Xtrain[:, n].sum() ==0:
        #         features_rm = np.append(features_rm, n)
        # Xtrain = np.delete(Xtrain, features_rm.astype(int), 1)
        # Xtest = np.delete(Xtest, features_rm.astype(int), 1)


        # uncomment these lines to change the resolution
        # res = 13
        # current_res = int(np.sqrt(Xtrain.shape[1]))
        # X_train_resized = np.empty((Xtrain.shape[0], res * res))
        # X_test_resized = np.empty((Xtest.shape[0], res * res))
        # for n in range(Xtrain.shape[0]):
        #     im = Image.fromarray(Xtrain[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_train_resized[n] = np.array(im).flatten()
        #
        # for n in range(Xtest.shape[0]):
        #     im = Image.fromarray(Xtest[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_test_resized[n] = np.array(im).flatten()
        #
        #
        # Xtrain = X_train_resized
        # Xtest = X_test_resized

        Ytrain = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, Ytrain_full).astype(int)[:, np.newaxis]
        Ytest = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, Ytest_full).astype(int)[:, np.newaxis]

        kernel = [ExtRBF(Xtrain.shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(1)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = LogisticLL()

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000

        image = None
        if 'image' in config.keys():
            image = config['image']

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, False,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 60, 'hyp': 15},
                                 max_iter=300, n_threads=n_threads, ftol=10,
                                 model_image_file=image, partition_size=partition_size))

    @staticmethod
    def MNIST_binary_inducing_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.mnist_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain_full = d['train_Y']
        Xtest = d['test_X']
        Ytest_full = d['test_Y']
        name = 'mnist_binary'

        # uncomment these lines to change the resolution
        # res = 13
        # current_res = int(np.sqrt(Xtrain.shape[1]))
        # X_train_resized = np.empty((Xtrain.shape[0], res * res))
        # X_test_resized = np.empty((Xtest.shape[0], res * res))
        # for n in range(Xtrain.shape[0]):
        #     im = Image.fromarray(Xtrain[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_train_resized[n] = np.array(im).flatten()
        #
        # for n in range(Xtest.shape[0]):
        #     im = Image.fromarray(Xtest[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_test_resized[n] = np.array(im).flatten()
        #
        #
        # Xtrain = X_train_resized
        # Xtest = X_test_resized

        Ytrain = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, Ytrain_full).astype(int)[:, np.newaxis]
        Ytest = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, Ytest_full).astype(int)[:, np.newaxis]

        kernel = [ExtRBF(Xtrain.shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(1)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = LogisticLL()

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000

        image = None
        if 'image' in config.keys():
            image = config['image']

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'hyp', 'inducing'], IdentityTransformation, False,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 60, 'hyp': 15, 'inducing': 6},
                                 max_iter=9, n_threads=n_threads, ftol=10,
                                 model_image_file=image, partition_size=partition_size))


    @staticmethod
    def sarcos_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.sarcos_data()

        names = []
        d = data[0]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'sarcos'
        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 3, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll = CogLL(0.1, 2, 1)

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000

        image = None
        if 'image' in config.keys():
            image = config['image']

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'll', 'hyp'], MeanStdYTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 50, 'hyp': 10, 'll': 10},
                                 max_iter=200,
                                 partition_size=partition_size,
                                 n_threads=n_threads,
                                 model_image_file=image))


    @staticmethod
    def sarcos_all_joints_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.sarcos_all_joints_data()

        names = []
        d = data[0]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'sarcos_all_joints'

        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)

        kernel = ExperimentSetup.get_kernels(Xtrain.shape[1], 8, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll = CogLL(0.1, 7, 1)

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000

        image = None
        if 'image' in config.keys():
            image = config['image']

        names.append(
            ModelLearn.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                 num_samples, sparsify_factor, ['mog', 'll', 'hyp'], MeanStdYTransformation, True,
                                 config['log_level'], False, latent_noise=0.001,
                                 opt_per_iter={'mog': 50, 'hyp': 10, 'll': 10},
                                 max_iter=200,
                                 partition_size=partition_size,
                                 ftol=10,
                                 n_threads=n_threads,
                                 model_image_file=image))


    @staticmethod
    def get_kernels(input_dim, num_latent_proc, ARD):
        return [ExtRBF(input_dim, variance=1, lengthscale=np.array((1.,)), ARD=ARD) for j in range(num_latent_proc)]

    @staticmethod
    def get_train_test(X, Y, n_train):
        """
        Divides inputs ``X`` and ``Y`` into a training set of size ``n_train``, and a test set which contains rest of
        data points.

        Returns
        -------
        Xn : ndarray
         training X

        Yn : ndarray
         training Y

        Xs : ndarray
         test X

        Ys : ndarray
         test Y
        """
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        Xn = data[:, :X.shape[1]]
        Yn = data[:, X.shape[1]:]
        return Xn[:n_train], Yn[:n_train], Xn[n_train:], Yn[n_train:]

