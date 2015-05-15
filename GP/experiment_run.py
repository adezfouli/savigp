import logging
from experiments import Experiments
from plot_results import PlotOutput
from multiprocessing.pool import Pool

class ExperimentRunner:

    @staticmethod
    def get_configs():
        """
        Builds an array of configuration for running
        """

        configs = []
        expr_names = ExperimentRunner.get_experiments()
        methods = ['full', 'mix1', 'mix2']
        sparse_factor = [1.0, 0.2, 0.1]
        run_ids = [1, 2, 3, 4, 5]
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
        Builds an array of experiments to run
        """

        # uncomment to run desired experiment
        # return [Experiments.boston_data.__name__]
        # return [Experiments.wisconsin_breast_cancer_data.__name__]
        return [Experiments.USPS_data.__name__]
        # return [Experiments.creep_data.__name__]
        # return [Experiments.abalone_data.__name__]


    @staticmethod
    def run_parallel(n_process):
        """
        :param n_process number of processes to run in parallel
        Runs experiments in parallel.
        """

        p = Pool(n_process)
        p.map(run_config, ExperimentRunner.get_configs())

    @staticmethod
    def run_serial():
        """
        Runs experiments in serial.
        """
        for c in ExperimentRunner.get_configs():
            run_config(c)


    @staticmethod
    def get_log_level():
        # return logging.DEBUG
        return logging.INFO


    @staticmethod
    def get_expr_names():
        return str(ExperimentRunner.get_experiments())[2:6]

    @staticmethod
    def boston_experiment():
        Experiments.boston_data({'method': 'mix2', 'sparse_factor': 0.8, 'run_id': 3, 'log_level': logging.DEBUG})

    @staticmethod
    def wisconsin_breast_experiment():
        Experiments.wisconsin_breast_cancer_data(
            {'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def abalone_experiment():
        Experiments.abalone_data({'method': 'full', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def creep_experiment():
        Experiments.creep_data({'method': 'full', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def USPS_experiment():
        Experiments.USPS_data({'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 3, 'log_level': logging.DEBUG})

    @staticmethod
    def mining_experiment():
        Experiments.mining_data({'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def plot():
        PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
                                   lambda x: x['method'] == 'mix1', False)

        # plots all the files
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                            None, False)

        # plots for an specific experiment
        # PlotOutput.plot_output_all('abalone_graph', Experiments.get_output_path(),
        #                            lambda x: x['experiment'] == 'abalone', False)

def run_config(config):
    try:
        logger.info('started config: ' + str(config))
        getattr(Experiments, config['method_to_run'])(config)
        logger.info('finished config: ' + str(config))
    except Exception as e:
        logger.exception(config)


if __name__ == '__main__':
    logger = Experiments.get_logger('general_' + Experiments.get_ID(), logging.DEBUG)

    ExperimentRunner.run_parallel(30)
    # run_config_serial(ExperimentRunner.get_configs())

    # runs an individual configuration

    # ExperimentRunner.boston_experiment()
    # ExperimentRunner.wisconsin_breast_experiment()
    # ExperimentRunner.USPS_experiment()
    # ExperimentRunner.mining_experiment()
    # ExperimentRunner.abalone_experiment()

    # ExperimentRunner.plot()
