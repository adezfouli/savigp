import numba
import numpy as np
import numpy.testing

@numba.jit('double(double[:])', nopython=True) #jit call without type indications will give an error
def lse_numba(a):
    result = 0.0
    largest_in_a = 0.0
    for i in range(a.shape[0]): # numba is slow when using max or np.max, so re-implementing
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(a.shape[0]):
        result += np.exp(a[i] - largest_in_a)
    return np.log(result) + largest_in_a

@numba.njit('double[:](double[:,:], double[:])') 
def lse_numba_axis_1(a, result):
#    result = np.empty((a.shape[0]))
    for r in range(a.shape[0]):
        result[r] = lse_numba(a[r,:])
    return result


@numba.njit('double[:](double[:], double[:], double[:])')
def sum_vec(a,b, result):
    for r in range(a.shape[0]):
        result[r] = a[r]+b[r]
    return result # cannot use nopython cos this means allocating new memory

@numba.njit
def lse_numba_axis_1_tile(a,v, 
    result, # (n_labels)
    temp_array): # (n_labels)
    # pre-assigned for speed
    # result = np.empty((a.shape[0]))
    for r in range(a.shape[0]):
        result[r] = lse_numba(sum_vec(a[r,:], v, temp_array))
    return result


@numba.jit('double[:,::1](double[:,::1], double[:,::1], int32, int32)')
def forwards_algo_log_alpha(log_edge_pot, #'(t-1,t)', 
                  log_node_pot, # '(t, label)', 
                  T, 
                  n_labels): 
    """ to obtain Z 
    - log_edge_pot = bold \psi in MLAPP (17.48)
    - log_node_pot[t,j] = \psi_t(y_t = j)
    
    wrt MLAPP, I also note the evidence X, and I note the labels Y (MLAPP notes Z)
    """

    temp_array_1 = np.empty((n_labels))
    temp_array_2 = np.empty((n_labels))
    log_alpha = np.empty((T, n_labels))
    log_alpha[0,:] = log_node_pot[0,:]
# don't need to preserve normalizing constant, except to produce log-likelihood
#    log_kappa = np.empty((T))
#    log_kappa[0] = lse_numba(log_alpha[0,:])
#    log_alpha[0,:] -= log_kappa[0]
    log_alpha[0,:] -= lse_numba(log_alpha[0,:])
    for t in range(1,T):
        log_alpha[t,:] = log_node_pot[t,:] + lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2)
#        log_kappa[t] = lse_numba(log_alpha[t,:])
#        log_alpha[t,:] -= log_kappa[t]
        log_alpha[t,:] -= lse_numba(log_alpha[t,:])
    return log_alpha

@numba.jit('double[:,::1](double[:,::1], double[:,::1], int32, int32)')
def forwards_backwards_algo_log_gamma(log_edge_pot, log_node_pot, T, n_labels):
    """ to obtain the (smoothed) posterior marginals p(y_t = j | X_1:T) = gamma_t(j) """

    temp_array_1 = np.empty((n_labels))
    temp_array_2 = np.empty((n_labels))
    temp_array_3 = np.empty((T))
    log_alpha = forwards_algo_log_alpha(log_edge_pot, log_node_pot, T, n_labels)
    log_beta = np.empty((T, n_labels))
    
    # set log_beta[T-1,:] to 0
    # need loop cos the following won't work
#    log_beta[T-1,:] = np.zeros((n_labels)) # numba cannot coerce
    for c in range(n_labels):
        log_beta[T-1,c] = 0
# no need to preserve kappa
#    log_kappa = np.empty((T))
#    log_kappa[T-1] = lse_numba(log_beta[-1,:])
    #log_beta[-1,:] -= log_kappa[-1] # not necessary cos already normalized
    for t in range(1,T):
        log_beta[T-1-t,:] = lse_numba_axis_1_tile(log_edge_pot, log_node_pot[T-t,:] + log_beta[T-t,:], temp_array_1, temp_array_2)
#        log_kappa[T-1-t] = lse_numba(log_beta[T-1-t,:])
        log_beta[T-1-t,:] -= lse_numba(log_beta[T-1-t,:])
    
    log_gamma = log_alpha + log_beta
    # perform the following (ie normalize log_gamma with axis=1=, but faster (cos removing np.tile python call):
    #log_gamma -= np.tile(lse_numba_axis_1(log_gamma), (n_labels, 1)).T
    temp = lse_numba_axis_1(log_gamma, temp_array_3)
    for c in range(n_labels):
        log_gamma[:,c] -= temp
    return log_gamma #return gamma MLAPP (17.52), shape (t, label)

@numba.jit('double[:,::1](double[:,::1], double[:,::1], int32, int32, double[:,::1], double[:,::1], double[:,::1], double[:,::1])')
def forwards_algo_log_Z(log_edge_pot, #'(t-1,t)', 
                  log_node_pot, # '(t, label)', 
                  T, 
                  n_labels,
                  log_alpha,
                  log_kappa,
                  temp_array_1, 
                  temp_array_2): 
    # log_alpha and log_kappa are pre-assigned for speed. The T dimension is set to max_T over the entire train and test datasets. Must not access beyond index T in this function !
    
#    log_alpha = np.empty((T, n_labels))
#    log_kappa = np.empty((T))
    log_alpha[0,:] = log_node_pot[0,:]
    log_kappa[0] = lse_numba(log_alpha[0,:])
    log_alpha[0,:] -= log_kappa[0] # normalize alpha
    for t in range(1,T):
        log_alpha[t,:] = log_node_pot[t,:] + lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2)
        #print("pre_mult" + str(np.exp(lse_numba_axis_1_tile(log_edge_pot.T, log_alpha[t-1,:], temp_array_1, temp_array_2))))
        log_kappa[t] = lse_numba(log_alpha[t,:])
        log_alpha[t,:] -= log_kappa[t] # normalize alpha
    log_Z = 0
    for i in range(T):
        log_Z += log_kappa[i]
        #print(log_kappa[i])
    return log_Z

