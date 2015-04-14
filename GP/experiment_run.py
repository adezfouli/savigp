from experiments import Experiments
from plot_results import PlotOutput
from multiprocessing.pool import ThreadPool, Pool


class ExperimentRunner:

    @staticmethod
    def get_configs():
        configs = []
        expr_names = [Experiments.boston_data.__name__]
        expr_names = [Experiments.wisconsin_breast_cancer_data.__name__]
        methods = ['mix1', 'mix2', 'full']
        sparse_factor = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        for e in expr_names:
            for m in methods:
                for s in sparse_factor:
                    configs.append({'method': m, 'sparse_factor': s, 'method_to_run': e})

        return configs

    @staticmethod
    def boston_experiment():
        Experiments.boston_data({'method': 'full', 'sparse_factor': 1.0})

    @staticmethod
    def wisconsin_breast_experiment():
        Experiments.wisconsin_breast_cancer_data({'method': 'full', 'sparse_factor': 1.0})

    @staticmethod
    def USPS_experiment():
        Experiments.USPS_data({'method': 'full', 'sparse_factor': 1.0})

    @staticmethod
    def plot():
        PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
                                   None, False)
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                            lambda x: (x['m'] in ['mix2']), False)
        #
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                            lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)
        #
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                        lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)

def run_config(config):
    getattr(Experiments, config['method_to_run'])(config)

if __name__ == '__main__':
    n_process = 4
    p = Pool(n_process)
    p.map(run_config, ExperimentRunner.get_configs())
    # ExperimentRunner.boston_experiment()
    # ExperimentRunner.wisconsin_breast_experiment()
