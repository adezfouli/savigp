import numpy as np
import os
import scipy.sparse
import chain_forwards_backwards_logsumexp
import prepare_from_data
import learn_predict
import kernels
import numba

try:
    import chain_forwards_backwards_native
    native_implementation_found = True
    @numba.jit # not even sure this speeds anything up. if it doesn't speed up anything, could remove this function and just define it 
    # inline as a lambda function
    def log_likelihood_function_native(log_node_pot, log_edge_pot, dataset_Y_n, object_size, n_labels):
        return chain_forwards_backwards_native.log_likelihood(log_edge_pot, log_node_pot, dataset_Y_n)
except ImportError as ie:
    native_implementation_found = False

def learn_predict_gpstruct_wrapper(
    data_indices_train=np.arange(0,10), 
    data_indices_test=np.arange(10,20),
    task='basenp',
    data_folder=None,
    result_prefix='/tmp/pygpstruct/',
    console_log=True, # log to console as well as to file ?
    n_samples=0, 
    prediction_thinning=1, # how often (in terms of MCMC iterations) to carry out prediction, ie compute f*|f and p(y*)
    lhp_update={},
    kernel=kernels.kernel_linear_unary,
    random_seed=0,
    stop_check=None,
    native_implementation=False
    ):
    if data_folder==None:
        data_folder = './data/%s' % task
        
    n_labels = None # used as flag after if/elif chain below, to check whether a correct task was indicated
    # NB can't infer n_features_x and n_labels from a dataset, cos they need to be consistent across train/ test datasets
    if (task == 'basenp'):
        n_features_x = 6438;
        n_labels = 3;
    elif (task == 'chunking'):
        n_features_x = 29764;
        n_labels = 14;
    elif (task == 'japanesene'):
        n_features_x = 102799;
        n_labels = 17;
    elif (task == 'segmentation'):
        n_features_x = 1386;
        n_labels = 2;
        
    if (n_labels == None):
        print('Task %s is not configured. Please use one of basenp, chunking, japanesene, segmentation as the value of the task argument (default is basenp).' % task)
    else:
        learn_predict.learn_predict_gpstruct(lambda logger : 
                           prepare_from_data_chain(
                                data_indices_train=data_indices_train, 
                                data_indices_test=data_indices_test,
                                data_folder=data_folder,
                                logger=logger,
                                n_labels=n_labels,
                                n_features_x=n_features_x,
                                native_implementation=native_implementation
                                ), 
                           result_prefix=result_prefix,
                           console_log=console_log,
                           n_samples=n_samples, 
                           prediction_thinning=prediction_thinning, 
                           lhp_update=lhp_update,
                           kernel=kernel,
                           random_seed=random_seed,
                           stop_check=stop_check
                           )

def prepare_from_data_chain(data_indices_train, data_indices_test, data_folder, logger, n_labels, n_features_x, native_implementation):
    logger.info("prepare_from_data_chain started with arguments: " + str(locals()))
    
    data_train = loadData(data_folder, n_labels, data_indices_train, n_features_x)
    data_test = loadData(data_folder, n_labels, data_indices_test, n_features_x)

    # pre-assigning for speed so that there's no memory assingment in the most inner loop of the likelihood computation
    max_T = np.max([data_train.object_size.max(), data_test.object_size.max()])
    
    if native_implementation:
        if native_implementation_found:
            chain_forwards_backwards_native.init_kappa(max_T)
            return (
                lambda f : prepare_from_data.log_likelihood_dataset(f, data_train, log_likelihood_function_native, False),  # that's ll_fun_wants_log_domain=False
                lambda f : prepare_from_data.posterior_marginals(f, data_test, marginals_function), 
                lambda marginals : compute_error_nlm(marginals, data_test),
                lambda f : prepare_from_data.log_likelihood_dataset(f, data_test, log_likelihood_function_native, False), # that's ll_fun_wants_log_domain=False
                prepare_from_data.average_marginals, 
                write_marginals,
                lambda marginals_file : read_marginals(marginals_file, data_test),
                n_labels, data_train.X, data_test.X, data_train, data_test)
        else:
            raise Exception("You have set native_implementation=True, but there has been an ImportError on import chain_forwards_backwards_native, and so I can't find the native implementation.")
    else:
        log_likelihood_function_numba.log_alpha = np.empty((max_T, n_labels))
        log_likelihood_function_numba.log_kappa = np.empty((max_T))    
        log_likelihood_function_numba.temp_array_1 = np.empty((n_labels))
        log_likelihood_function_numba.temp_array_2 = np.empty((n_labels))
        return (
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_train, log_likelihood_function_numba, True),
            lambda f : prepare_from_data.posterior_marginals(f, data_test, marginals_function), 
            lambda marginals : compute_error_nlm(marginals, data_test),
            lambda f : prepare_from_data.log_likelihood_dataset(f, data_test, log_likelihood_function_numba, True),
            prepare_from_data.average_marginals, 
            write_marginals,
            lambda marginals_file : read_marginals(marginals_file, data_test),
            n_labels, data_train.X, data_test.X, data_train, data_test)

def loadData(dirName, n_labels, indexData, n_features_x):
    # .Y list of ndarrays
    class dataset:
        pass
    dataset.N = indexData.shape[0]
    dataset.n_labels = n_labels;
    dataset.Y = []
    dataset.unaries = []

    # read data from sparse representation in file
    # -----
    # to initialize a sparse matrix X containing (rows = position in sequence, all seq concatenated, cols = feature, value=binaries)
    # first create a large IJV array, with the I indices corresponding to rows in the vstack'ed X
    # if I knew how to append to a COO sparse matrix, I could avoid all this, and just append rows as I go along
    ijv_array_list = []     # ijv_array_list is a list, with one ijv array per sentence
    dataset.n_points = 0 # how many rows does X have so far?
    dataset.object_size = np.zeros((dataset.N), dtype=np.int16) # contains length of sentence n
    for n in range(indexData.shape[0]):
        # set X part
        this_x = np.loadtxt(os.path.join(dirName, str(indexData[n]+1) + ".x"), dtype=np.int32) # int32: some tasks have up to 102799 features (japanesene)
        dataset.object_size[n] = this_x[-1,0] # the last element of the (Matlab, row-ordered) sparse format representation will contain an index into the last row
        this_x[:,[0,1]] -= 1 # from Matlab-indexing to Numpy-indexing: remove 1 from i and j indices
        this_x[:,0] += dataset.n_points # correct all row indices so that they correspond to a vstack'ed X
        ijv_array_list.append(this_x)
        dataset.n_points += dataset.object_size[n]
        
        # set other parts
        this_y = np.loadtxt(os.path.join(dirName, str(indexData[n]+1) + '.y'), dtype=np.int8) # labels start with 0 in data files
        this_y[this_y < 0] = 0 # remove the three -1 labels found in task JapaneseNE for unknown reason
        dataset.Y.append(this_y) 
        dataset.unaries.append(np.zeros((dataset.object_size[n], dataset.n_labels), dtype=np.int))

    assert(dataset.n_points == dataset.object_size.sum())
    # stack the ijv lists vertically, to create a complete ijv array
    ijv_array=np.vstack(tuple(ijv_array_list))
    # create a sparse matrix from the ijv array (coo_matrix expects input in order vij)
    dataset.X = scipy.sparse.coo_matrix((ijv_array[:,2],(ijv_array[:,0],ijv_array[:,1])), shape=(dataset.n_points, n_features_x))

    # .unaries has the most basic (assumes no tying) shape compatible with the linear CRF: f~(n, t, y)
    f_index_max = 0 # contains largest index in f
    for yt in range(dataset.n_labels):
        for n in range(dataset.N):
            for t in range(dataset.object_size[n]):
                dataset.unaries[n][t, yt] = f_index_max
                f_index_max = f_index_max + 1

    dataset.binaries = np.arange(f_index_max, f_index_max + n_labels**2).reshape((dataset.n_labels, dataset.n_labels), order='F')
    #f_index_max += n_labels**2 
    
    return dataset

@numba.jit
def log_likelihood_function_numba(log_node_pot, log_edge_pot, dataset_Y_n, object_size, n_labels):
        # log-likelihood for point n, for observed data dataset.Y[n], consists of 
        # "numerator": sum of potentials for  binaries (indexed (y_{t-1}, y_{t})) and unaries (indexed (t, y))
        log_pot = log_edge_pot[dataset_Y_n[:-1], dataset_Y_n[1:]].sum() + log_node_pot[np.arange(object_size), dataset_Y_n].sum()
        # using array indexing in numpy 
        # nodes: for each position in the chain (np.arange(object_size)), select the correct unary factor corresponding to y (dataset_Y_n)
        # edges: select edge factors with tuples (y_{t-1}, y_{t}). Use as the first index, y_{t-1}, obtained by cutting off dataset_Y_n before the last position; and as the second index, y{t}, obtained by shifting dataset_Y_n to the left (ie [1:]).
        
        # "denominator": log_Z obtained from forwards pass
        log_Z = chain_forwards_backwards_logsumexp.forwards_algo_log_Z(log_edge_pot,
                                     log_node_pot,
                                     object_size,
                                     n_labels,
                                     log_likelihood_function_numba.log_alpha, # pre-assigned memory space to speed up likelihood computation
                                     log_likelihood_function_numba.log_kappa,
                                     log_likelihood_function_numba.temp_array_1,
                                     log_likelihood_function_numba.temp_array_2)
        return (log_pot - log_Z)

def marginals_function(log_node_pot, log_edge_pot, object_size, n_labels):
    """
    marginals returned have shape (object_size, n_labels)
    """
    return np.exp(chain_forwards_backwards_logsumexp.forwards_backwards_algo_log_gamma(log_edge_pot, log_node_pot, object_size, n_labels))

def write_marginals(marginals_f, marginals_file):
    #print(marginals_f)
    np.array(np.vstack(marginals_f), dtype=learn_predict.dtype_for_arrays).tofile(marginals_file) # can hstack cos all elements have #labels rows
    
def read_marginals(marginals_file, dataset):
    result = np.fromfile(marginals_file, dtype=learn_predict.dtype_for_arrays)
    result = result.reshape((-1, dataset.n_points * dataset.n_labels))
    result = np.split(result, result.shape[0], axis=0) # make list of rows
    # each element now contains a 1D array containing all the marginals for all data points. need to turn that into a list of n elements.
    result = [np.split(marginals_f.reshape((-1, dataset.n_labels)), dataset.object_size.cumsum()[:-1], axis=0) for marginals_f in result] # undo the hstack followed by writing to disk in row-first C order: first reshape 1D array so that each row is for one label; then split row-wise so that each block corresponds to an object
    # dataset.object_size.cumsum()[:-1] => row indices where to split the marginals array
    #print(result)
    return result
    
def compute_error_nlm(marginals, dataset):
    stats_per_object = np.empty((dataset.N,2)) # first col for error rate, second col for neg log marg
    for n, marginals_n in enumerate(marginals):
        #print("marginals_n.shape: " + str(marginals_n.shape))
        ampm = np.argmax(marginals_n, axis = 1) # argmax posterior marginals
        #print("comparing %s to %s" % (str(ampm.shape), str(dataset.Y[n].shape)))
        stats_per_object[n,0] = (ampm != dataset.Y[n]).sum()
        
        # from marginals, use dataset.Y[n] to select on axis "labels"
        avg_nlpm_object = np.empty_like(dataset.Y[n])
        T = dataset.Y[n].shape[0]
        for t in range(T):
            avg_nlpm_object[t] = -np.log(marginals_n[t, dataset.Y[n][t]])
        stats_per_object[n,1] = avg_nlpm_object.sum()
    return stats_per_object.sum(axis=0) / dataset.n_points # per point averaging
    
if __name__ == "__main__":
    learn_predict_gpstruct_wrapper()
