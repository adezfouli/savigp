import numpy as np
"""
dataset object fields:
- N: number of objects (images, sentences) in dataset
- n_labels: number of possible labels for each atom in an object
- object_size[n]: how many atoms (eg pixels or words) in object n ?
- n_points : object_size.sum(), # atoms in the entire data set
- X: X data (might be sparse matrix) in format (N, # input features), so each row represents X features for an input atom
- Y: list of length N
- Y[n]: array of shape corresponding to the type of data (2D array (grid_size, grid_size) for image, 1D (object_size[n]) for sentences)
- unaries: list of size N
- unaries[n]: array of shape (object_size[n], n_labels) indexed by [t,y]
- binaries: array of shape (n_labels, n_labels) to be indexed with [y_{t-1}, y_{t}], yields an index into f pointing to the binary (edge) MRF parameter corresponding to [y_{t-1}, y_{t}] (remember the edge factor parameters are all tied)
"""

def posterior_marginals(f, dataset, marginals_function):
    """
    def posterior_marginals(f : "learnt latent variables, shape column", 
                        dataset : "dataset object") -> "list of (normalized) posterior marginals, length N":
    """
    pm = []
    for n in range(dataset.N):
        log_node_pot = f[dataset.unaries[n]]
        log_edge_pot = f[dataset.binaries]
        posterior_marginals_single = marginals_function(log_node_pot, log_edge_pot, dataset.object_size[n], dataset.n_labels) # in grid case, object_size will be ignored
        pm.append(posterior_marginals_single)
    return pm

def average_marginals(marginals_list):
    # Python 2 support: in py2, reduce is builtin
    try:
        from functools import reduce
    except ImportError:
        pass
    number_marginals = len(marginals_list)
    if (number_marginals == 1):
        return marginals_list[0]
    else:
        averaged_marginals = []
        N = len(marginals_list[0]) # says how many objects there are
        for n in range(N): # assumes all lists of marginals are the same length, namely N
            marginals_object_n = [marginals_all_objects[n] for marginals_all_objects in marginals_list]
            averaged_marginals.append(reduce(lambda x,y : add_marginals(x,y), marginals_object_n) / number_marginals) # mean of marginals for object n
        return averaged_marginals
        
def add_marginals(x,y):
    """
    x,y : marginals for one object
    adds marginals, and returns their sum
    """
    return x+y
    
import numba
#@numba.jit           
def log_likelihood_dataset(f, dataset, log_likelihood_datapoint, ll_fun_wants_log_domain):
    """
    f : log-domain potentials
    ll_fun_wants_log_domain : whether or not the log-likelihood function needs f to be in log-domain (this is false only for the native chain LL implementation)
    """
    #print("f.dtype : %s" % f.dtype)
    if not ll_fun_wants_log_domain:
        f = np.exp(f) # changing semantics of f instead of inserting if's on edge_pot=... and node_pot=...
    ll = 0
    edge_pot = f[dataset.binaries]
#    print(dataset.binaries)
#    print(log_edge_pot)
    #assert(log_edge_pot.shape == (dataset.n_labels, dataset.n_labels))
    for n in range(dataset.N):
        node_pot = f[dataset.unaries[n]]
#        print(dataset.unaries[n][:5,:5,0])
#        print(log_node_pot[:5,:5,0])
        #assert(log_node_pot.shape == (dataset.object_size, dataset.object_size, dataset.n_labels))
        
        ll_datapoint = log_likelihood_datapoint(node_pot, edge_pot, dataset.Y[n], dataset.object_size[n], dataset.n_labels) 
        # if (ll_datapoint >0):
            # info_string = ""
            # info_string += 'log_likelihood_datapoint as computed: %g\n' % ll_datapoint
            # info_string += 'n: %g\n' % n
            # info_string += 'node_pot.tolist(): %s\n' % node_pot.tolist()
            # info_string += 'edge_pot.tolist(): %s\n' % edge_pot.tolist()
            # info_string += 'dataset.Y[n]: %s\n' % dataset.Y[n].tolist()
            # raise Exception("positive log-likelihood is not allowed. More information:\n" + info_string)
        ll += ll_datapoint
        # in grid case, object_size will be ignored
    return ll # LL should not be scaled !
