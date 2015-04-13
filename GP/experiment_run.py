from experiments import Experiments
from plot_results import PlotOutput

if __name__ == '__main__':
    plots = []

    plots.append(Experiments.boston_data('full', 1))
    # plots.append(Experiments.breast_caner_data('full', 1.0))


    # methods = ['mix1', 'mix2', 'full']
    # for m in methods:
        # plots.append(Experiments.boston_data(m, 1))

        # plots.append(Experiments.boston_data(m, 0.2))
        #
        # plots.append(Experiments.boston_data(m, 0.6))
        #
        # plots.append(Experiments.boston_data(m, 0.4))
        #
        # plots.append(Experiments.boston_data(m, 0.8))

    PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
                               None, False)


    # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
    #                            lambda x: (x['m'] in ['mix2']), False)
    #
    # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
    #                            lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)

    # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
    #                            lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)
