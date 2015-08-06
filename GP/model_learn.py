import logging
import pickle
import csv

import GPy
import numpy as np

from savigp import SAVIGP
from savigp_diag import SAVIGP_Diag
from savigp_single_comp import SAVIGP_SingleComponent
from optimizer import Optimizer
from util import id_generator, check_dir_exists, get_git


class ModelLearn:
    """
    Provides facility for fitting models to data, making predictions and exporting results to csv files.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_output_path():
        """
        Returns
        -------
        path : string
         the path in which results of the experiment will be saved
        """
        return '../results/'

    @staticmethod
    def get_logger_path():
        """
        Returns
        -------
        path : string
         the path in which log files will be created
        """
        return '../logs/'

    @staticmethod
    def get_logger(name, level):
        """
        Creates loggers

        Parameters
        ----------
        name : string
         name of the log file

        level : string
         level of debugging

        Returns
        -------
        logger : logger
         created loggers
        """

        logger = logging.getLogger(name)
        logger.setLevel(level)
        check_dir_exists(ModelLearn.get_logger_path())
        fh = logging.FileHandler(ModelLearn.get_logger_path() + name + '.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def export_train(name, Xtrain, Ytrain, export_X=False):
        """
        Exports training data into a csv file

        Parameters
        ----------
        name : string
         name of file

        Xtrain : ndarray
         X of training data

        Ytrain : ndarray
         Y of training data

        export_X : boolean
         whether to export 'X'. If False, only ``Ytrain`` will be exported

        :return:
         None
        """
        path = ModelLearn.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'train_'
        header = ['Y%d,' % (j) for j in range(Ytrain.shape[1])]
        data = None
        if export_X:
            data = np.hstack((Ytrain, Xtrain))
            header += ['X%d,' % (j) for j in range(Xtrain.shape[1])]
        else:
            data = Ytrain
        np.savetxt(path + file_name + '.csv', data, header=''.join(header), delimiter=',', comments='')


    @staticmethod
    def export_track(name, track):
        """
        exports trajectory of the objective function

        Parameters
        ----------
        name : string
         name of the file to which track will be exported

        track : list
         trajectory of the objective function

        """
        path = ModelLearn.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'obj_track_'
        np.savetxt(path + file_name + '.csv', np.array([track]).T,
                   header='objective'
                   , delimiter=',', comments='')

    @staticmethod
    def export_model(model, name):
        """
        exports Model into a csv file

        Parameters
        ----------
        model : model
         the model to be exported

        name : string
         name of the csv file
        """

        path = ModelLearn.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'model_'
        if model is not None:
            with open(path + file_name + '.csv', 'w') as fp:
                f = csv.writer(fp, delimiter=',')
                f.writerow(['#model', model.__class__])
                params = model.get_all_params()
                param_names = model.get_all_param_names()
                for j in range(len(params)):
                    f.writerow([param_names[j], params[j]])


    @staticmethod
    def export_test(name, X, Ytrue, Ypred, Yvar_pred, nlpd, pred_names=[''], export_X=False, post_fix= ''):
        """
        Exports test data and the predictions into a csv file

        Parameters
        ----------
        name : string
         name of the file

        X : ndarray
         X test for which prediction have been made

        Ytrue : ndarray
         The true values of 'Y'

        Ypred : ndarray
         Predictions at the test points

        Yvar_pred : ndarray
         Variance of the prediction

        nlpd : ndarray
         NLPD of the predictions

        pred_names : list
         not necessary. It should be ['']

        export_X : boolean
         Whether to export 'X' to the csv file. If False, 'X' will not be exported into the csv file
         (useful in large datasets).
        """

        path = ModelLearn.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'test_' + post_fix
        out = []
        out.append(Ytrue)
        out += Ypred
        out += Yvar_pred
        out += [nlpd]
        header = ['Ytrue%d,' % (j) for j in range(Ytrue.shape[1])] + \
                 ['Ypred_%s_%d,' % (m, j) for m in pred_names for j in range(Ypred[0].shape[1])] + \
                 ['Yvar_pred_%s_%d,' % (m, j) for m in pred_names for j in range(Yvar_pred[0].shape[1])] + \
                 ['nlpd,'] + ['NLPD_%d,' % (j) for j in range(nlpd.shape[1] - 1)]

        if export_X:
            out.append(X)
            header += ['X%d,' % (j) for j in range(X.shape[1])]

        header = ''.join(header)
        out = np.hstack(out)
        np.savetxt(path + file_name + '.csv', out
                   , header=header
                   , delimiter=',', comments='')


    @staticmethod
    def export_configuration(name, config):
        """
        Exports configuration of the model as well as optimisation parameters to a csv file

        Parameters
        ----------
        name : string
         Name of the file

        config : dictionary
         Configuration to be exported
        """

        path = ModelLearn.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = path + 'config_' + '.csv'
        with open(file_name, 'wb') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())
            writer.writeheader()
            writer.writerow(config)

    @staticmethod
    def get_ID():
        """
        :return: A random ID
        """
        return id_generator(size=6)


    @staticmethod
    def opt_callback(name):
        """
        A callback function which will be called by the optimiser to save the model

        Parameters
        ----------
        name : string
         name of the folder to save the model in.
        """

        def callback(model, current_iter, total_evals, delta_m, delta_s, obj_track):
            path = ModelLearn.get_output_path() + name + '/'
            check_dir_exists(path)
            pickle.dump(model.image(), open(path + 'model.dump', 'w'))
            pickle.dump({
                'current_iter': current_iter,
                'total_evals': total_evals,
                'delta_m': delta_m,
                'delta_s': delta_s,
                'obj_track': obj_track,
                'obj_fun': model.objective_function()
            },
                open(path + 'opt.dump', 'w'))

        return callback


    @staticmethod
    def run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, run_id, num_inducing, num_samples,
                  sparsify_factor, to_optimize, trans_class, random_Z, logging_level, export_X,
                  latent_noise=0.001, opt_per_iter=None, max_iter=200, n_threads=1, model_image_file=None,
                  xtol=1e-3, ftol=1e-5, partition_size=3000, opt_callback = None):
        """
        Fits a model to the data (Xtrain, Ytrain) using the method provided by 'method', and makes predictions on
         'Xtest' and 'Ytest', and exports the result to csv files.

        Parameters
        ----------
        Xtest : ndarray
         X of test points

        Xtrain : ndarray
         X of training points

        Ytest : ndarray
         Y of test points

        Ytrain : ndarray
         Y of traiing points

        cond_ll : subclass of likelihood/Likelihood
         Conditional log likelihood function used to build the model.

        kernel : list
         The kernel that the model uses. It should be an array, and size of the array should be same as the
         number of latent processes. Each element should provide interface similar to ``ExtRBF`` class

        method : string
         The method to use to learns the model. It can be 'full', 'mix1', and 'mix2'

        name : string
         The name that will be used for logger file names, and results files names

        run_id : object
         ID of the experiment, which can be anything, and it will be included in the configuration file. It can provdie
         for example a number referring to a particular test and train partition.

        num_inducing : integer
         Number of inducing points

        num_samples : integer
         Number of samples for estimating objective function and gradients

        sparsify_factor : float
         Can be any number and will be included in the configuration file. It will not determine
         the number of inducing points

        to_optimize : list
         The set of parameters to optimize. It should be a list, and it can include 'll', 'mog', 'hyp', e.g.,
         it can be ['ll', 'mog'] in which case posterior and ll will be optimised.

        trans_class : subclass of DataTransformation
         The class which will be used to transform data.

        random_Z : boolean
         Whether to initialise inducing points randomly on the training data. If False, inducing points
         will be placed using k-means (or mini-batch k-mean) clustering. If True, inducing points will be placed randomly
         on the training data.

        logging_level : string
         The logging level to use.

        export_X : boolean
         Whether to export X to csv files.

        latent_noise : integer
         The amount of latent noise to add to the kernel. A white noise of amount latent_noise will be
         added to the kernel.

        opt_per_iter: integer
         Number of update of each subset of parameters in each iteration, e.g., {'mog': 15000, 'hyp': 25, 'll': 25}

        max_iter: integer
         Maximum of global iterations used on optimization.

        n_threads: integer
         Maximum number of threads used.

        model_image_file: string
         The image file from the which the model will be initialized.

        xtol: float
         Tolerance of 'X' below which the optimization is determined as converged.

        ftol: float
         Tolerance of 'f' below which the optimization is determined as converged.

        partition_size: integer
         The size which is used to partition training data (This is not the partition used for SGD).
         Training data will be split to the partitions of size ``partition_size`` and calculations will be done on each
         partition separately. This aim of this partitioning of data is to make algorithm memory efficient.

        opt_callback: callable
         A function which will be called at the end of each optimisation global iteration. If it is None, then it will be
         set to the ModelLearn.opt_callback

        Returns
        -------
        folder : string
         the name of the folder in which results are stored

        model : model
         the fitted model itself.
        """

        if opt_per_iter is None:
            opt_per_iter = {'mog': 40, 'hyp': 40, 'll': 40}
        folder_name = name + '_' + ModelLearn.get_ID()
        logger = ModelLearn.get_logger(folder_name, logging_level)
        transformer = trans_class.get_transformation(Ytrain, Xtrain)
        Ytrain = transformer.transform_Y(Ytrain)
        Ytest = transformer.transform_Y(Ytest)
        Xtrain = transformer.transform_X(Xtrain)
        Xtest = transformer.transform_X(Xtest)

        opt_max_fun_evals = None
        total_time = None
        timer_per_iter = None
        tracker = None
        export_model = False
        git_hash, git_branch = get_git()

        properties = {'method': method,
                      'sparsify_factor': sparsify_factor,
                      'sample_num': num_samples,
                      'll': cond_ll.__class__.__name__,
                      'opt_max_evals': opt_max_fun_evals,
                      'opt_per_iter': opt_per_iter,
                      'xtol': xtol,
                      'ftol': ftol,
                      'run_id': run_id,
                      'experiment': name,
                      'max_iter': max_iter,
                      'git_hash': git_hash,
                      'git_branch': git_branch,
                      'random_Z': random_Z,
                      'latent_noise:': latent_noise,
                      'model_init': model_image_file
                      }
        logger.info('experiment started for:' + str(properties))

        if opt_callback is None:
            opt_callback = ModelLearn.opt_callback

        model_image = None
        current_iter = None
        if model_image_file is not None:
            model_image = pickle.load(open(model_image_file + 'model.dump'))
            opt_params = pickle.load(open(model_image_file + 'opt.dump'))
            current_iter = opt_params['current_iter']

        if model_image:
            logger.info('loaded model - iteration started from: ' + str(opt_params['current_iter']) +
                        ' Obj fun: ' + str(opt_params['obj_fun']) + ' fun evals: ' + str(opt_params['total_evals']))
        cb = opt_callback(folder_name)
        if method == 'full':
            m = SAVIGP_SingleComponent(Xtrain, Ytrain, num_inducing, cond_ll,
                                       kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads,
                                       image=model_image, partition_size=partition_size)

            cb(m, 0, 0, 0, 0, None)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol,
                                         cb, current_iter)
        if method == 'mix1':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 1, cond_ll,
                            kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads,
                            image=model_image, partition_size=partition_size)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol,
                                         opt_callback(folder_name), current_iter)
        if method == 'mix2':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 2, cond_ll,
                            kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads,
                            image=model_image, partition_size=partition_size)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol,
                                         opt_callback(folder_name), current_iter)
        if method == 'gp':
            m = GPy.models.GPRegression(Xtrain, Ytrain, kernel[0])
            if 'll' in to_optimize and 'hyp' in to_optimize:
                m.optimize('bfgs')

        y_pred, var_pred, nlpd = m.predict(Xtest, Ytest)
        if not (tracker is None):
            ModelLearn.export_track(folder_name, tracker)
        ModelLearn.export_train(folder_name, transformer.untransform_X(Xtrain), transformer.untransform_Y(Ytrain),
                                export_X)
        ModelLearn.export_test(folder_name,
                               transformer.untransform_X(Xtest),
                               transformer.untransform_Y(Ytest),
                               [transformer.untransform_Y(y_pred)],
                               [transformer.untransform_Y_var(var_pred)],
                               transformer.untransform_NLPD(nlpd),
                               [''], export_X)

        if export_model and isinstance(m, SAVIGP):
            ModelLearn.export_model(m, folder_name)

        properties['total_time'] = total_time
        properties['time_per_iter'] = timer_per_iter
        properties['total_evals'] = total_evals
        ModelLearn.export_configuration(folder_name, properties)
        return folder_name, m

