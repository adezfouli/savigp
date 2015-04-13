import multiprocessing
from experiments import Experiments
from plot_results import PlotOutput
from multiprocessing.pool import ThreadPool

class ExperimentRunner:

    @staticmethod
    def parallel_experiment(f, n_thread):
        configs = []
        methods = ['mix1', 'mix2', 'full']
        sparse_factor = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        for m in methods:
            for s in sparse_factor:
                configs.append({'method': m, 'sparse_factor': s})

        p = ThreadPool(n_thread)
        p.map(f, configs)

    @staticmethod
    def logistic_experiment():
        plots.append(Experiments.wisconsin_breast_cancer_data({'method': 'full', 'sparse_factor': 1.0}))

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

    @staticmethod
    def test(x):
        print x

if __name__ == '__main__':
    plots = []
    ExperimentRunner.parallel_experiment(Experiments.boston_data, 2)
    # ExerpiemntRunner.logistic_experiment()
    # ExerpiemntRunner.boston_experiment()


