#!/usr/bin/env python

import argparse
import json
import logging
from multiprocessing.pool import Pool

import experiment_setup
from model_learn import ModelLearn
import plot_results

EXPERIMENTS = {
    'boston': experiment_setup.boston_experiment,
    'wisconsin': experiment_setup.wisconsin_experiment,
    'mining': experiment_setup.mining_experiment,
    'usps': experiment_setup.USPS_experiment,
    'abalone': experiment_setup.abalone_experiment,
    'creep': experiment_setup.creep_experiment,
    'mnist': experiment_setup.MNIST_experiment,
    'mnist_binary': experiment_setup.MNIST_binary_experiment,
    'mnist_binary_inducing': experiment_setup.MNIST_binary_inducing_experiment,
    'sarcos': experiment_setup.sarcos_experiment,
}

# TODO: Change methods into an enum.
METHODS = ['mix1', 'mix2', 'full']
LOG_LEVEL = logging.DEBUG

logger = ModelLearn.get_logger(ModelLearn.get_logger_path(), 'general_' + ModelLearn.get_ID(), LOG_LEVEL)

def main():
    args = setup_args()
    if 'file' in args:
        # Run multiple experiments from a configuration file.
        with open(args['file']) as config_file:
            config = json.loads(config_file.read())
            run_parallel(config) 
    elif 'experiment' in args:
        # Run a single experiment from commmandline arguments.
        experiment = EXPERIMENTS[args['experiment_name']]
        del args['experiment_name']
        args['log_level'] = LOG_LEVEL
        experiment(**args)
    else:
        print 'You must either chose an experiment (-e) or a config file (-f).'

    # uncomment to plots the outputs in results folder
    # plot()

def setup_args():
    parser = argparse.ArgumentParser(description='Experimental framework for savigp.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--experiment_name', choices=EXPERIMENTS, default=argparse.SUPPRESS,
                        help='The name of the experiment to run.')
    parser.add_argument('-f', '--file', default=argparse.SUPPRESS,
                        help='A json file containing a list of experiment configurations to run.')
    parser.add_argument('-m', '--method', choices=METHODS, default=METHODS[0],
                        help='The type of mixture of gaussians to learn.')
    parser.add_argument('-s', '--sparsity_factor', type=float, default=1.0,
                        help='The sparsity of inducing points. Value must be between 0 and 1.')
    parser.add_argument('-r', '--run_id', type=int, default=1,
                        help='The id of the experiment configuration.')
    parser.add_argument('-i', '--image', default=argparse.SUPPRESS,
                        help='A path to a partially completed large scale experiment')
    parser.add_argument('-n', '--n_threads', default=argparse.SUPPRESS,
                        help='The number of threads to run for a large scale experiment.')
    parser.add_argument('-p', '--partition_size', default=argparse.SUPPRESS,
                        help='The size of sample partitions for a large scale experiment.')
    return vars(parser.parse_args())

def run_parallel(config):
    """
    Creates a process for each element in the array returned by ``get_configs()`` and the experiment corresponding
    the each element. The maximum number of processes to run in parallel is determined by ``n_process``
    """

    # Setup an array of individual experiment configurations.
    experiment_configs = []
    for e in config['experiment_names']:
        for m in config['methods']:
            for s in config['sparsity_factors']:
                for r in config['run_ids']:
                    experiment_configs.append({'experiment_name': e,
                                               'method': m,
                                               'sparsity_factor': s,
                                               'run_id': r,
                                               'log_level': LOG_LEVEL})

    pool = Pool(config['num_processes'])
    pool.map(run_config, experiment_configs)

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
        experiment = EXPERIMENTS[config['experiment_name']]
        del config['experiment_name']
        experiment(**config)
        logger.info('finished config: ' + str(config))
    except Exception as e:
        logger.exception(config)


if __name__ == '__main__':
    main()
