import Image

__author__ = 'AT'

from data_source import DataSource
from data_transformation import *
from ExtRBF import ExtRBF
from likelihood import *
import model_learn
import numpy as np


def boston_experiment(method, sparsity_factor, run_id, log_level):
    name = 'boston'
    data = DataSource.boston_data()[run_id - 1]
    kernel = get_kernels(data['train_X'].shape[1], 1, True)
    cond_ll = UnivariateGaussian(np.array(1.0))

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['hyp', 'mog', 'll'],
                                 MeanTransformation,
                                 True,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200)

def wisconsin_experiment(method, sparsity_factor, run_id, log_level):
    name = 'breast_cancer'
    data = DataSource.wisconsin_breast_cancer_data()[run_id - 1]

    # uncomment these lines to use softmax
    # kernel = Experiments.get_kernels(Xtrain.shape[1], 2, False)
    # Ytrain = np.array([(Ytrain[:,0] + 1) / 2, (-Ytrain[:,0] + 1) / 2]).T
    # Ytest = np.array([(Ytest[:,0] + 1) / 2, (-Ytest[:,0] + 1) / 2]).T
    # cond_ll = SoftmaxLL(2)

    # uncomment these lines to use logistic
    cond_ll = LogisticLL()
    kernel = get_kernels(data['train_X'].shape[1], 1, False)

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp'],
                                 IdentityTransformation,
                                 True,
                                 log_level,
                                 False, 
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200)

def mining_experiment(method, sparsity_factor, run_id, log_level):
    name = 'mining'
    data = DataSource.mining_data()[run_id - 1]
    kernel = get_kernels(data['train_X'].shape[1], 1, False)

    cond_ll = LogGaussianCox(math.log(191. / 811))
    kernel[0].variance = 1.0
    kernel[0].lengthscale = 13516.

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog'],
                                 IdentityTransformation,
                                 True,
                                 log_level, True,
                                 opt_per_iter={'mog': 15000},
                                 max_iter=1)

def USPS_experiment(method, sparsity_factor, run_id, log_level):
    name = 'usps'
    data = DataSource.USPS_data()[run_id - 1]
    kernel = [ExtRBF(data['train_X'].shape[1], variance=2, lengthscale=np.array((4.,)), ARD=False) for j in range(3)]
    cond_ll = SoftmaxLL(3)

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp'],
                                 IdentityTransformation,
                                 True,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=300)


def abalone_experiment(method, sparsity_factor, run_id, log_level):
    name = 'abalone'
    data = DataSource.abalone_data()[run_id - 1]
    kernel = get_kernels(data['train_X'].shape[1], 1, False)
    cond_ll = WarpLL(np.array([-2.0485, 1.7991, 1.5814]),
                     np.array([2.7421, 0.9426, 1.7804]),
                     np.array([0.1856, 0.7024, -0.7421]),
                     np.log(0.1))

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp', 'll'],
                                 MinTransformation,
                                 True,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200)


def creep_experiment(method, sparsity_factor, run_id, log_level):
    name = 'creep'
    data = DataSource.creep_data()[run_id - 1]

    scaler = preprocessing.StandardScaler().fit(data['train_X'])
    data['train_X'] = scaler.transform(data['train_X'])
    data['test_X'] = scaler.transform(data['test_X'])
    kernel = get_kernels(data['train_X'].shape[1], 1, True)

    cond_ll = WarpLL(np.array([3.8715, 3.8898, 2.8759]),
                     np.array([1.5925, -1.3360, -2.0289]),
                     np.array([0.7940, -4.1855, -3.0289]),
                     np.log(0.01))

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp', 'll'],
                                 MinTransformation,
                                 True,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                 max_iter=200)


def MNIST_experiment(method, sparsity_factor, run_id, log_level,
                     image=None, n_threads=1, partition_size=3000):
    name = 'mnist'
    data = DataSource.mnist_data()[run_id - 1]

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

    kernel = [ExtRBF(data['train_X'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(10)]
    cond_ll = SoftmaxLL(10)

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp'],
                                 IdentityTransformation,
                                 False,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 50, 'hyp': 10},
                                 max_iter=300,
                                 n_threads=n_threads,
                                 ftol=10,
                                 model_image_file=image,
                                 partition_size=partition_size)

def MNIST_binary_experiment(method, sparsity_factor, run_id, log_level,
                            image=None, n_threads=1, partition_size=3000):
    name = 'mnist_binary'
    data = DataSource.mnist_data()[run_id - 1]

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

    data['train_Y'] = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, data['train_Y']).astype(int)[:, np.newaxis]
    data['test_Y'] = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, data['test_Y']).astype(int)[:, np.newaxis]

    kernel = [ExtRBF(data['train_X'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(1)]
    cond_ll = LogisticLL()

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp'],
                                 IdentityTransformation,
                                 False,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 60, 'hyp': 15},
                                 max_iter=300,
                                 n_threads=n_threads,
                                 ftol=10,
                                 model_image_file=image,
                                 partition_size=partition_size)

def MNIST_binary_inducing_experiment(method, sparsity_factor, run_id, log_level,
                                     image=None, n_threads=1, partition_size=3000):
    name = 'mnist_binary'
    data = DataSource.mnist_data()[run_id - 1]

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

    data['train_Y'] = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, data['train_Y']).astype(int)[:, np.newaxis]
    data['test_Y'] = np.apply_along_axis(lambda x: x[1:10:2].sum() - x[0:10:2].sum(), 1, data['test_Y']).astype(int)[:, np.newaxis]

    kernel = [ExtRBF(data['train_X'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(1)]
    cond_ll = LogisticLL()

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'hyp', 'inducing'],
                                 IdentityTransformation,
                                 False,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 60, 'hyp': 15, 'inducing': 6},
                                 max_iter=9,
                                 n_threads=n_threads,
                                 ftol=10,
                                 model_image_file=image,
                                 partition_size=partition_size)

def sarcos_experiment(method, sparsity_factor, run_id, log_level,
                      image=None, n_threads=1, partition_size=3000):
    name = 'sarcos'
    data = DataSource.sarcos_data()[0]

    kernel = get_kernels(data['train_X'].shape[1], 3, False)

    cond_ll = CogLL(0.1, 2, 1)

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'll', 'hyp'],
                                 MeanStdYTransformation,
                                 True,
                                 log_level,
                                 False,
                                 opt_per_iter={'mog': 50, 'hyp': 10, 'll': 10},
                                 max_iter=200,
                                 partition_size=partition_size,
                                 n_threads=n_threads,
                                 model_image_file=image)

def sarcos_all_joints_experiment(method, sparsity_factor, run_id, log_level,
                                 image=None, n_threads=1, partition_size=3000):
    name = 'sarcos_all_joints'
    data = DataSource.sarcos_all_joints_data()[0]


    scaler = preprocessing.StandardScaler().fit(data['train_X'])
    data['train_X'] = scaler.transform(data['train_X'])
    data['test_X'] = scaler.transform(data['test_X'])

    kernel = get_kernels(data['train_X'].shape[1], 8, False)

    cond_ll = CogLL(0.1, 7, 1)

    return model_learn.run_model(data['train_X'],
                                 data['train_Y'],
                                 data['test_X'],
                                 data['test_Y'],
                                 cond_ll,
                                 kernel,
                                 method,
                                 name,
                                 data['id'],
                                 sparsity_factor,
                                 ['mog', 'll', 'hyp'],
                                 MeanStdYTransformation,
                                 True,
                                 log_level, False,
                                 opt_per_iter={'mog': 50, 'hyp': 10, 'll': 10},
                                 max_iter=200,
                                 partition_size=partition_size,
                                 ftol=10,
                                 n_threads=n_threads,
                                 model_image_file=image)

def get_kernels(input_dim, num_latent_proc, ARD):
    return [ExtRBF(input_dim, variance=1, lengthscale=np.array((1.,)), ARD=ARD) for j in range(num_latent_proc)]

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
