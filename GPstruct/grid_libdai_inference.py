import numpy as np
import dai
import numba

# depends on 
# libDAI for exact and TRW BP inference  http://staff.science.uva.nl/~jmooij1/libDAI/libDAI-0.3.1.tar.gz (must be installed on the python path)
# pylibdai Python wrapper for libDAI by Sameh Kamis https://github.com/samehkhamis/pylibdai

# NB interesting libDAI parameters list https://github.com/probml/pmtk3/blob/4582f0eed9acd65691681256f9c895769b439f81/toolbox/GraphicalModels/inference/libdai/libdaiOptions.m

#@numba.jit error Numba 0.13 NotImplementedError: offset=63 opcode=69 opname=BUILD_MAP
def grid_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, method="JTREE"):
    n_rows = grid_size
    n_cols = grid_size
    node_pot = np.exp(log_node_pot)
    edge_pot = np.exp(log_edge_pot)
    
    factors = make_factors(node_pot, edge_pot, n_rows, n_cols)
    if method == 'JTREE':
        props = {'inference': 'SUMPROD', 'verbose':0, 'updates':'SHSH'}
        # JTREE doesn't seem to support logdomain factors (as opposed to methods MR, BP, TRW)
    elif method == 'TRWBP':
        props = {'inference': 'SUMPROD', 'verbose':0, 'updates': 'SEQMAX', 'tol': '1e-9', 'maxiter': '100', 'logdomain': '0'}
    else:
        raise Exception("inference method %s not among supported methods JTREE, TRWBP" % method)
    logz = dai.dai(factors, method, props, with_extra_beliefs=False, with_map_state=False)[0] # ignoring q, maxdiff return values
    
    # to obtain log p(evidence), will iterate over cliques and their potential functions obtained by DAI 
    ll = - logz # potential functions are unnormalized, the normalizer is Z
    #print('LL inited with %g' % ll)
    
    # compute log product of potentials (the numerator of the likelihood)
    # code below is slow because of the tuple array lookup
#    y_vector = y.ravel()
#     for member, prob in factors:        
#         # if member has m variables v_1 ... v_m, then prob has shape (v_1_range, v_2_range, ... v_m_range)
#         # create the index into prob (a tuple of length m) using variable values from the evidence
#         index_into_prob = tuple(y_vector[member])
#         # now index prob with it, obtain the value of potential(v_1 .. v_m)
#         ll += np.log(prob[index_into_prob])
#         #print('adding logprob %g, new value LL = %g' % (np.log(prob[index_into_prob]), ll))

    # this is faster
    for row in range(n_rows):
        for col in range(n_cols):
            ll += log_node_pot[row, col, y[row, col]]
            #print('adding %g' % log_node_pot[row, col, y[row, col]])
            # edge to right neighbour
            if (row + 1 < n_rows):
                ll += log_edge_pot[y[row, col], y[row+1, col]]
            #    print('adding %g' % log_edge_pot[y[row, col], y[row+1, col]] )
            # edge to bottom neighbour
            if (col + 1 < n_cols):
                ll += log_edge_pot[y[row, col], y[row, col+1]]
                
    return ll

def grid_marginals(log_node_pot, log_edge_pot, grid_size, n_labels, method='JTREE'):
    n_rows = grid_size
    n_cols = grid_size
    node_pot = np.exp(log_node_pot)
    edge_pot = np.exp(log_edge_pot)
    factors = make_factors(node_pot, edge_pot, n_rows, n_cols)
    if method == 'JTREE':
        props = {'inference': 'SUMPROD', 'verbose':0, 'updates':'SHSH'}
        # JTREE doesn't seem to support logdomain factors (as opposed to methods MR, BP, TRW)
    elif method == 'TRWBP':
        props = {'inference': 'SUMPROD', 'verbose':0, 'updates': 'SEQMAX', 'tol': '1e-9', 'maxiter': '100', 'logdomain': '0'}
    else:
        raise Exception("inference method %s not among supported methods JTREE, TRWBP" % method)

    qv= dai.dai(factors, method, props, with_extra_beliefs=True, with_map_state=False)[3] # ignoring the rest; full return is logz, q, maxdiff, qv, qf 

    posterior_marginals = np.empty((n_rows, n_cols, n_labels))
    for member, prob in qv:
        variable_index = member[0] # member.shape is guaranteed to be (1,) because they all contain vertices
        # to convert variable_index into a (row, col) representation, assume that variable_index_table was filled in row-first order (default of reshape)
        row = variable_index // n_cols
        col = variable_index % n_cols
        posterior_marginals[row, col, :] = prob
    return posterior_marginals    # shape (row, col, label)
    
@numba.jit('double(double[:,:,:], double[:,:], int8, int8)')
def make_factors(node_pot, edge_pot, n_rows, n_cols):
    factors = []
    variable_index_table = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) # stays the same for all calls of this function - TODO make persistent

    #normz_edge_pot = normalize(edge_pot)
    for row in range(n_rows):
        for col in range(n_cols):
            # unary factor
            factors.append(([variable_index_table[row, col]], node_pot[row, col, :]))
            
            # binary factors
            # edge to right neighbour
            if (row + 1 < n_rows):
                factors.append(([variable_index_table[row, col], variable_index_table[row + 1, col]], edge_pot))
            # edge to bottom neighbour
            if (col + 1 < n_cols):
                factors.append(([variable_index_table[row, col], variable_index_table[row, col + 1]], edge_pot))
    return factors
