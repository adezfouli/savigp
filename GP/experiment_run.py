from experiments import Experiments
from plot_results import PlotOutput

class ExerpiemntRunner:

    @staticmethod
    def boston_experiment():
        methods = ['mix1', 'mix2', 'full']
        for m in methods:
            plots.append(Experiments.boston_data(m, 1))

            plots.append(Experiments.boston_data(m, 0.2))

            plots.append(Experiments.boston_data(m, 0.6))

            plots.append(Experiments.boston_data(m, 0.4))

            plots.append(Experiments.boston_data(m, 0.8))

    @staticmethod
    def logistic_experiment():
        plots.append(Experiments.breast_caner_data('full', 1.0))

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

if __name__ == '__main__':
    plots = []
    ExerpiemntRunner.logistic_experiment()


