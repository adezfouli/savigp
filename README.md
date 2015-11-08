**Big and Black GP**


This code is an implementation of the inference framework for Gaussian process (GP) models proposed in [1]. The framework is able to perform
inference for Gaussian process models with arbitrary likelihood function (Black) and it is scalable to large datasets (Big).


The method is also referred to as SAVIGP, which stands for Scalable Automated Variational Inference for Gaussian Process Models.

**Experiments**

Several example usages for different forms of likelihood functions are available in `GP/experiment_setup.py` file. 

In order to replicate the experiments reported in the paper, the corresponding line in `GP/experiment_run.py` file should be uncommented. For example, 
the following line in the file will run the experiment using Boston dataset:
```python
ExperimentRunner.boston_experiment()
```

The experiments also can be ran concurrently, as it is shown in the file.

**Output**

After running an experiment is finished, the results will be saved in a directory called `resutls`, which is a folder one level higher than the directory of the code. 
The result of each experiment will be saved in a separate directory which contains several files, as follows:

| File        | content|
| ------------- |:-----|
| test_.csv      | Result of the prediction on test data |
| train_.csv      | Training data used. Model can be configured to not save these data, in the case the training dataset is large|
| model.dump | An image of the model. After each iteration of the optimisation this image will be updated, and it can be used to initialize the model|
|opt.dump| Last state of the optimiser that can be used to continue the optimisation from the last iteration|
|config_.csv| Configuration of the model|
|\*.log|Log file|

**Dependences**

Following packages are required:
* Python 2.7 (2.7.6)
* Scipy (0.15.1)
* Numpy (1.9.1)
* GPy (0.6.0)
* pandas (0.16.0)
* scikit-learn (0.14.1)

Following are required for tests:
* DerApproximator (0.52)
* texttable (0.8.2)

Numbers in the paranthesis indicate the tested version.

**Visualization**

In order to plot the results generated, the the last line in the `GP/experiment_run.py` file should be uncommented:

```Python
ExperimentRunner.plot()
```

In the case that the results of several experiments are in the result folder, then this method will plot average of the results. The type of plots
depends on the likelihood function. For example bar-charts are generated in the case of classification, and box-plots in the case of regression models. The 
type of likelihood is extracted from the `config_.csv` file. The plot function also exports the data used for graphs (e.g., the height of the bar charts and 
size of error-bars) into a separate folder called `graph_data`, which is one lever higher than the code folder. These data can be used to regenerate the plots
using other tools. I used an R code for the plots in the paper. The code is in the file `graphs_R/graphs.R`.

**Likelihood functions**

The code comes with a set of pre-defined likelihood functions as follows:

| Likelihood class       |problem type|
| ------------- |:-----|
|Likelihood.UnivariateGaussian|Normal Gaussian process regression with single output|
|Likelihood.MultivariateGaussian|Normal Gaussian process regression with multi-dimensional output|
|Likelihood.LogGaussianCox|Log Gaussian Cox process. Can be used for example for the prediction of rate of incidents|
|Likelihood.LogisticLL|Logistic likelihood function. Can be used for binary classification|
|Likelihood.SoftmaxLL|SoftmaxLL likelihood function. Can be used for multi-class classification|
|Likelihood.WarpLL|Likelihood corresponding to Warp Gaussian process|
|Likelihood.CogLLL|Likelihood corresponding to Gaussian process networks|


For defining a new likelihood function the class `Likelihood.likelihood` should be extended and the required functions should be implemented.
See the class documentation for more details.

**Example**

Below is a short example using Boston dataset:

```python
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

```
                     
The code shows how to configure the model. There are two options that can significantly affect the speed of the code and 
the amount of memory usage: `n_threads` and `partition_size`. The whole dataset is divided into partitions of size
 `partition_size` and calculations on each partition is performed on a separate thread, where the maximum number of threads is `n_threads`. 

References
----------

**[1]** A. Dezfouli, E. V. Bonilla. Scalable Inference for Gaussian Process Models with Black-Box Likelihoods
    Processes, Advances in Neural and Information Processing Systems (NIPS),
    Montreal, December 2015
