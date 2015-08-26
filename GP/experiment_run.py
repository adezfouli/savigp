import logging
from experiment_setup import ExperimentSetup
from model_learn import ModelLearn
from plot_results import PlotOutput
from multiprocessing.pool import Pool


class ExperimentRunner:

    def __init__(self):
        pass

    @staticmethod
    def get_configs():
        """
        Builds an array of configuration for running (in parallel)
        """

        configs = []
        expr_names = ExperimentRunner.get_experiments()
        methods = ['full', 'mix1']
        sparse_factor = [1.0, 0.5, 0.2, 0.1]
        run_ids = [1]
        for e in expr_names:
            for m in methods:
                for s in sparse_factor:
                    for run_id in run_ids:
                        configs.append({'method': m,
                                        'sparse_factor': s,
                                        'method_to_run': e,
                                        'run_id': run_id,
                                        'log_level': ExperimentRunner.get_log_level()
                                        })

        return configs

    @staticmethod
    def get_experiments():
        """
        Builds an array of experiments to run in parallel (the array can contain more than one experiment)
        """

        # uncomment to run desired experiment
        # return [Experiments.boston_data.__name__]
        # return [Experiments.wisconsin_breast_cancer_data.__name__]
        # return [Experiments.USPS_data.__name__]
        # return [Experiments.creep_data.__name__]
        # return [Experiments.abalone_data.__name__]
        return [ExperimentSetup.mining_data.__name__]
    #

    @staticmethod
    def run_parallel(n_process):
        """
        Creates a process for each element in the array returned by ``get_configs()`` and the experiment corresponding
        the each element. The maximum number of processes to run in parallel is determined by ``n_process``
        """

        p = Pool(n_process)
        p.map(run_config, ExperimentRunner.get_configs())


    @staticmethod
    def get_log_level():
        """ debug level """
        # return logging.DEBUG
        return logging.INFO


    @staticmethod
    def boston_experiment():
        ExperimentSetup.boston_data({'method': 'mix2', 'sparse_factor': 0.8, 'run_id': 3, 'log_level': logging.DEBUG})

    @staticmethod
    def wisconsin_breast_experiment():
        ExperimentSetup.wisconsin_breast_cancer_data(
            {'method': 'full', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def abalone_experiment():
        ExperimentSetup.abalone_data({'method': 'full', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def creep_experiment():
        ExperimentSetup.creep_data({'method': 'full', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def USPS_experiment():
        ExperimentSetup.USPS_data({'method': 'full', 'sparse_factor': 0.1, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def mining_experiment():
        ExperimentSetup.mining_data({'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def sarcos_experiment():
        ExperimentSetup.sarcos_data({'method': 'full',
                                 'sparse_factor': 0.04,
                                 'run_id': 0,
                                 'log_level': logging.DEBUG,
                                 'n_thread': 15,
                                 'partition_size': 2000,
                                 # 'image': '../results/all/'
    })

    @staticmethod
    def sarcos_all_joins_experiment():
        ExperimentSetup.sarcos_all_joints_data({'method': 'full',
                                 'sparse_factor': 0.04,
                                 'run_id': 0,
                                 'log_level': logging.DEBUG,
                                 'n_thread': 15,
                                 'partition_size': 3000,
                                 #'image': '../results/sarcos_1/'
    })

    @staticmethod
    def mnist_experiment():
        ExperimentSetup.MNIST_data({'method': 'full',
                                'sparse_factor': 0.004,
                                'run_id': 1,
                                'log_level': logging.DEBUG,
                                'n_thread': 20,
                                'partition_size': 2000,
                                # 'image': '../results/mnist_1/'
                                })


    @staticmethod
    def mnist_binary_experiment():
        ExperimentSetup.MNIST_binary_data({'method': 'full',
                                'sparse_factor': 200. / 60000,
                                'run_id': 1,
                                'log_level': logging.DEBUG,
                                'n_thread': 20,
                                'partition_size': 2000,
                                # 'image': '../results/mnist_1/'
                                })


    @staticmethod
    def mnist_binary_inducing_experiment():
        ExperimentSetup.MNIST_binary_inducing_data({'method': 'full',
                                'sparse_factor': 200. / 60000,
                                'run_id': 1,
                                'log_level': logging.DEBUG,
                                'n_thread': 8,
                                'partition_size': 1000,
                                # 'image': '../results/mnist_1/'
                                })

    @staticmethod
    def plot():
        PlotOutput.plot_output_all('boston', ModelLearn.get_output_path(),
                                   lambda x: x['method'] == 'full', False)

        # plots all the files
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                            None, False)

        # plots for an specific experiment
        # PlotOutput.plot_output_all('abalone_graph', Experiments.get_output_path(),
        #                            lambda x: x['experiment'] == 'abalone', False)


def run_config(config):
    try:
        logger.info('started config: ' + str(config))
        getattr(ModelLearn, config['method_to_run'])(config)
        logger.info('finished config: ' + str(config))
    except Exception as e:
        logger.exception(config)


if __name__ == '__main__':
    logger = ModelLearn.get_logger('general_' + ModelLearn.get_ID(), logging.DEBUG)

    # uncomment to run experiments in parallel
    # ExperimentRunner.run_parallel(3)

    # runs an individual configuration
    # ExperimentRunner.boston_experiment()
    # ExperimentRunner.wisconsin_breast_experiment()
    # ExperimentRunner.USPS_experiment()
    # ExperimentRunner.mining_experiment()
    # ExperimentRunner.abalone_experiment()
    # ExperimentRunner.mnist_binary_inducing_experiment()
    ExperimentRunner.mnist_binary_experiment()
    # ExperimentRunner.sarcos_all_joins_experiment()
    # ExperimentRunner.sarcos_experiment()


    # uncomment to plots the outputs in results folder
    # ExperimentRunner.plot()
