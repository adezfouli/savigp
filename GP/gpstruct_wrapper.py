import logging
import numpy as np
from GPstruct.prepare_from_data_chain import prepare_from_data_chain


def gpstruct_wrapper(
    data_indices_train=np.arange(0,50),
    data_indices_test=np.arange(51,53),
    task='chunking',
    data_folder=None,
    ):
    if data_folder==None:
        data_folder = 'data/struct/%s' % task

    n_labels = None # used as flag after if/elif chain below, to check whether a correct task was indicated
    # NB can't infer n_features_x and n_labels from a dataset, cos they need to be consistent across train/ test datasets
    if (task == 'basenp'):
        n_features_x = 6438
        n_labels = 3
    elif (task == 'chunking'):
        n_features_x = 29764
        n_labels = 14
    elif (task == 'japanesene'):
        n_features_x = 102799
        n_labels = 17
    elif (task == 'segmentation'):
        n_features_x = 1386
        n_labels = 2

    if (n_labels == None):
        print('Task %s is not configured. Please use one of basenp, chunking, japanesene, segmentation as the value of the task argument (default is basenp).' % task)
    else:
       return prepare_from_data_chain(
            data_indices_train=data_indices_train,
            data_indices_test=data_indices_test,
            data_folder=data_folder,
            logger=logging.getLogger('tt'),
            n_labels=n_labels,
            n_features_x=n_features_x,
            native_implementation=False
            )

if __name__ == '__main__':
    print gpstruct_wrapper()