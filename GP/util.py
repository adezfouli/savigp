import numpy as np

def mdiag_dot(A, B):
    return np.einsum('ij,ji -> i', A, B)