import numpy as np
import os
import prepare_from_data
import learn_predict

def learn_predict_gpstruct_wrapper(
    data_indices_train=np.arange(0,10), 
    data_indices_test=np.arange(10,20),
    data_folder=None, # if None, will default, cf below
    result_prefix='/tmp/pygpstruct/',
    grid_size=10, 
    n_labels=2,
    n_samples=0, 
    prediction_thinning=1,
    likelihood_exact=True,
    inference_exact=True,
    console_log=True,
    lhp_update={}
    ):
    if (data_folder == None):
        data_folder = './dags.size_%d' % grid_size
    learn_predict.learn_predict_gpstruct(lambda logger : 
                           prepare_from_data_grid(
                                data_indices_train=data_indices_train, 
                                data_indices_test=data_indices_test,
                                data_folder=data_folder,
                                grid_size=grid_size,
                                n_labels=n_labels,
                                logger=logger,
                                likelihood_exact=likelihood_exact,
                                inference_exact=inference_exact
                                ), 
                           result_prefix=result_prefix,
                           console_log=console_log,
                           n_samples=n_samples, 
                           prediction_thinning=prediction_thinning, 
                           lhp_update=lhp_update
                           )

def prepare_from_data_grid(data_indices_train, 
                           data_indices_test, 
                           data_folder, 
                           grid_size,
                           likelihood_exact,
                           inference_exact, 
                           n_labels,
                           logger = None):
    import grid_libdai_inference
    import grid_pseudo_likelihood 
    if likelihood_exact:
        log_likelihood_datapoint = lambda log_node_pot, log_edge_pot, y, object_size_n, n_labels: grid_libdai_inference.grid_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, method="JTREE") # ignore object_size_n, use grid_size always
    else:
        log_likelihood_datapoint = lambda log_node_pot, log_edge_pot, y, object_size_n, n_labels: grid_pseudo_likelihood.grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, visible_pixels=None) # ignore object_size_n, use grid_size always

    if inference_exact:
        marginals_function = lambda log_node_pot, log_edge_pot, object_size_n, n_labels : grid_libdai_inference.grid_marginals(log_node_pot, log_edge_pot, grid_size, n_labels, method="JTREE") # ignore object_size_n, use grid_size always
    else:
        marginals_function = lambda log_node_pot, log_edge_pot, object_size_n, n_labels : grid_libdai_inference.grid_marginals(log_node_pot, log_edge_pot, grid_size, n_labels, method="TRWBP") # ignore object_size_n, use grid_size always
    logger.info(locals())
    
    # order of files given by sorted filenames
    #import glob
    #data_ids = [filename[len(data_folder) + 1:len(data_folder)+8] for filename in sorted(glob.glob(os.path.join(data_folder, '*.features.npy')))]
    
    # order of files given by fold-split which comes with data
    data_ids = np.loadtxt(os.path.join(data_folder, 'dags-regions', 'fold-1-train.txt'), dtype=np.str)
    
    if (len(data_ids) == 0):
        logger.error("data_folder %s has no files *.feature.npy" % data_folder)
    data_ids = np.asarray(data_ids) # will allow integer indexing

    data_train = create_dataset(data_ids=data_ids[data_indices_train], data_folder=data_folder, grid_size=grid_size, n_labels=n_labels)
    data_test = create_dataset(data_ids=data_ids[data_indices_test], data_folder=data_folder, grid_size=grid_size, n_labels=n_labels)
    
    logger.debug("datasets were read in")

    return (lambda f : prepare_from_data.log_likelihood_dataset(f, data_train, log_likelihood_datapoint, True),  # that's ll_fun_wants_log_domain=True
            lambda f : prepare_from_data.posterior_marginals(f, data_test, marginals_function), 
            lambda marginals : compute_error_nlm(marginals, data_test),
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_test, log_likelihood_datapoint, True), # that's ll_fun_wants_log_domain=True
            prepare_from_data.average_marginals,  
            write_marginals,
            lambda marginals_file : read_marginals(marginals_file, data_test),
            data_train.n_labels, np.vstack(data_train.X), np.vstack(data_test.X)) 
 

def load_datafiles(filename_component, data_ids, data_folder, dtype=None):
    result = []
    for file_id in data_ids:
        result.append(np.load('%s/%s.%s.npy' % (data_folder, file_id, filename_component)))
    return result 
    
def create_dataset(data_ids, data_folder, grid_size, n_labels):
    class dataset:
        pass
    dataset.N = len(data_ids)
    dataset.object_size = np.empty(dataset.N, dtype=np.int16); dataset.object_size.fill(np.int16(grid_size) ** 2)
    dataset.grid_size = grid_size
    dataset.n_points = grid_size * grid_size * dataset.N
    dataset.n_labels = np.int16(n_labels)
    all_files = load_datafiles('features', data_ids, data_folder)
    n_features = all_files[0].shape[2] # should be 8
    X_list = []
    for x_image in all_files: # assume size 5,6,8
#        reshaped_features = np.rollaxis(x_image, 2) # will be size 8, 5, 6
#        feature_array = reshaped_features.reshape((n_features, dataset.object_size * dataset.object_size),) # will be size 8, 5*6
#        feature_array = feature_array.T # shape 5*6, 8, as expected
#        X_list.append(feature_array)
        X_list.append(x_image.reshape((grid_size * grid_size, n_features), order='F'))

    dataset.X = np.vstack(X_list)
    
    dataset.Y = load_datafiles('labels', data_ids, data_folder)
    
    # fill dataset.unaries
    # each set of unaries corresponds to one log_node_pot array of shape 
    # (grid_size, grid_size, n_labels)
    log_node_pot_size = grid_size * grid_size * dataset.n_labels
    a=np.arange(n_labels * dataset.N * grid_size * grid_size).reshape((n_labels, dataset.N, grid_size, grid_size),) # have all the indices in shape
    a=np.split(a, dataset.N, axis=1) # split along the axis dataset.N (axis 1 above), obtain a list of len dataset.N
    a=[np.squeeze(e) for e in a] # split will leave an axis of size 1, squeeze it out
    a = [np.rollaxis(e, 0, 3) for e in a ] # move the axis n_labels to the end, so that every element has shape (grid_size, grid_size, n_labels)
    dataset.unaries = [np.swapaxes(e, 0, 1) for e in a] # swap grid axes to reflect order='F' in X_train 
#    dataset.unaries = [np.arange(image_index * log_node_pot_size,
#                                 (image_index + 1) * log_node_pot_size)
#                       .reshape(dataset.object_size, dataset.object_size, dataset.n_labels) 
#                       for image_index in range(dataset.N)
#                       ]
#    
    # fill dataset.binaries
    # corresponds to log_edge_pot of shape (n_labels * n_labels)
    log_edge_pot_size = dataset.n_labels * dataset.n_labels
    dataset.binaries = np.arange(dataset.N * log_node_pot_size, 
                                 dataset.N * log_node_pot_size + log_edge_pot_size).reshape(dataset.n_labels, dataset.n_labels, order='F')
    return dataset
     
def write_marginals(marginals_f, marginals_file):
    #print(marginals_f)
    np.array(np.hstack([marginals_object.ravel() for marginals_object in marginals_f]), dtype=np.float32).tofile(marginals_file) # can hstack cos all elements have #labels rows
    
def read_marginals(marginals_file, dataset):
    result = np.fromfile(marginals_file, dtype=np.float32)
    result = result.reshape((-1, dataset.n_points * dataset.n_labels))
    result = np.split(result, result.shape[0], axis=0) # make list of rows
    # each element now contains a 1D array containing all the marginals for all data points. need to turn that into a list of n elements.
    result = [np.split(marginals_f, dataset.N, axis=1) for marginals_f in result] 
    result = [[marginals_object.reshape((dataset.grid_size, dataset.grid_size, dataset.n_labels)) for marginals_object in marginals_f] for marginals_f in result]
    #print(result)
    return result

def compute_error_nlm(marginals, data_test):
    stats_per_object = np.empty((data_test.N,2))
    for n, marginals_n in enumerate(marginals):
        ampm = np.argmax(marginals_n, axis = 2) # argmax posterior marginals
        stats_per_object[n,0] = (ampm != data_test.Y[n]).mean()

        # compute average (per pixel) negative log posterior marginal
        avg_nlpm_object = np.empty_like(data_test.Y[n])
        for i in range(marginals[n].shape[0]): # to generalize across grid+chain, must find a way to select from array ar, along axis ax, using selector array s (while keeping all the other axes unchanged)
            for j in range(marginals[n].shape[1]):
                avg_nlpm_object[i,j] = -np.log(marginals_n[i,j,data_test.Y[n][i,j]])
        stats_per_object[n,1] = avg_nlpm_object.mean()
    return stats_per_object.mean(axis=0)    