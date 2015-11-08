import logging
from ExtRBF import ExtRBF
from model_learn import ModelLearn
from data_transformation import MeanTransformation
from likelihood import UnivariateGaussian
from data_source import DataSource
import numpy as np

# defining model type. It can be "mix1", "mix2", or "full"
method = "full"

# number of inducing points
num_inducing = 30

# loading data
data = DataSource.boston_data()

d = data[0]
Xtrain = d['train_X']
Ytrain = d['train_Y']
Xtest = d['test_X']
Ytest = d['test_Y']

# is is just of name that will be used for the name of folders and files when exporting results
name = 'boston'

# defining the likelihood function
cond_ll = UnivariateGaussian(np.array(1.0))

# number of samples used for approximating the likelihood and its gradients
num_samples = 2000

# defining the kernel
kernels = [ExtRBF(Xtrain.shape[1], variance=1, lengthscale=np.array((1.,)), ARD = False)]

ModelLearn.run_model(Xtest,
                     Xtrain,
                     Ytest,
                     Ytrain,
                     cond_ll,
                     kernels,
                     method,
                     name,
                     d['id'],
                     num_inducing,
                     num_samples,
                     num_inducing / Xtrain.shape[0],

                     # optimise hyper-parameters (hyp), posterior parameters (mog), and likelihood parameters (ll)
                     ['hyp', 'mog', 'll'],

                     # Transform data before training
                     MeanTransformation,

                     # place inducting points on training data. If False, they will be places using clustering
                     True,

                     # level of logging
                     logging.DEBUG,

                     # do not export training data into csv files
                     False,

                     # add a small latent noise to the kernel for stability of numerical computations
                     latent_noise=0.001,

                     # for how many iterations each set of parameters will be optimised
                     opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},

                     # total number of global optimisations
                     max_iter=200,

                     # number of threads
                     n_threads=1,

                     # size of each partition of data
                     partition_size=3000)
