import numpy as np
import numba

@numba.njit('double(double[:])') # this is the workaround for 0.12.2
def lse_numba(a):
    result = 0.0
    largest_in_a = 0.0
    for i in range(a.shape[0]): # numba is slow when using max or np.max, so re-implementing
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(a.shape[0]):
        result += np.exp(a[i] - largest_in_a)
    return np.log(result) + largest_in_a

@numba.jit('double(double, double, double[:,:,:], double[:,:], int8[:,:], int16, int16, double[:])')
# making it njit (which it should be, and it's the reason why ll_local is assigned outside this function) yields
#TypingError: Undeclared getitem(array(int8, 2d, A), (float64 x 2))
# the numba.njit line -- don't know why. Removing the ll_local[l] = assignemnt doesnt avoid the error.
def local_pseudolikelihood(row, col, log_node_pot, log_edge_pot, y, grid_size, n_labels, ll_local):
            #print("current pixel " + str((row, col)) + ", using unaries" + str(log_node_pot[row, col, :]))
            # Numba performance: rewrite the following line
            #ll_local = np.copy(log_node_pot[row, col, :])
            # -- start rewrite
            for l in range(n_labels):
                ll_local[l] = log_node_pot[row, col, l] 
            # -- end rewrite
            # stores unnormalized LL for current pixel, based on direct neighbours, for all values of the current pixel
            # shape (n_labels,)
            
            # now walk through all cliques of which current pixel is a member
    
            # pixel above
            if (row - 1 >= 0):
                # take edges ending in current pixel, and starting in current pixel
                # in my representation, each edge really appears twice, because log_edge_pot is not constrained to be symetric:
                # and so there's an edge (x_i, x_j) with a factor different from (x_j, x_i) !
                #print("neighbour above, using " + str(log_edge_pot[y[row - 1, col], :]))
                ll_local += log_edge_pot[y[row - 1, col],:] # log_edge_pot[y[row - 1, col], :] + log_edge_pot[:, y[row - 1, col]]
                
            # pixel below
            if (row + 1 < grid_size):
                #print("neighbour below, using " + str(log_edge_pot[:, y[row + 1, col]]))
                ll_local += log_edge_pot[:, y[row + 1, col]] # log_edge_pot[y[row + 1, col], :] + log_edge_pot[:, y[row + 1, col]]
            
            # pixel left
            if (col - 1 >= 0):
                #print("neighbour left, using " + str( log_edge_pot[y[row, col - 1], :]))
                ll_local += log_edge_pot[y[row, col - 1], :] # log_edge_pot[y[row, col - 1], :] + log_edge_pot[:, y[row, col - 1]]
            # pixel right
            if (col + 1 < grid_size):
                #print("neighbour right, using " + str(log_edge_pot[:, y[row, col + 1]]))
                ll_local += log_edge_pot[:, y[row, col + 1]] # log_edge_pot[y[row, col + 1], :] + log_edge_pot[:, y[row, col + 1]]
    
            #print(ll_local)
            # now ll_local contains unnormalized LL
            return ll_local[y[row, col]] - lse_numba(ll_local)

@numba.jit('double(double[:,:,:], double[:,:], int8[:,:], int16, int16, int8[:,:])', nopython=False)
# Numba 0.13 doesn't support default arguments, will require argument visible_pixels
def grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, visible_pixels):
    """ 
    log_node_pot.shape = (grid_size, grid_size, n_labels)
    log_edge_pot.shape = (n_labels, n_labels)
    y.shape = (grid_size, grid_size) contains labels in range(n_labels)
    visible_pixels=None if all pixels are visible, otherwise (n,2) array for n visible pixels
    """
    ll_grid = 0
    temp_array = np.empty((n_labels)) #used by local_pseudolikelihood

    if visible_pixels == None:
        for row in range(grid_size):
            for col in range(grid_size):
                ll_grid += local_pseudolikelihood(row, col, log_node_pot, log_edge_pot, y, grid_size, n_labels, temp_array)
    else:
        for i in range(visible_pixels.shape[0]):
            (row, col) = tuple(visible_pixels[i,:])
            ll_grid += local_pseudolikelihood(row, col, log_node_pot, log_edge_pot, y, grid_size, n_labels, temp_array)
    return ll_grid



if __name__ == '__main__':
    import numpy.testing
    
    n_labels = 3
    grid_size = 5
    log_node_pot = np.linspace(start=0,stop=1,num= grid_size**2 * n_labels).reshape((grid_size, grid_size, n_labels), order='F')
    log_edge_pot = np.linspace(start=0,stop=1, num= n_labels**2).reshape((n_labels, n_labels), order='F')
    y = np.asarray([[1, 2, 0, 0, 1],
           [1, 1, 0, 1, 2],
           [2, 1, 0, 2, 0],
           [1, 0, 1, 1, 2],
           [1, 0, 2, 1, 0]], dtype=np.int8) # generated randomly using np.random.randint(0, n_labels, size=(grid_size, grid_size))
    
    numpy.testing.assert_almost_equal(grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, visible_pixels=None), -40.0308354818)
    #%timeit -n 1000 grid_pseudo_likelihood.grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels)
    print("SUCCESS. Python implementation of pseudo likelihood has same result as given by Matlab using J Domke's graphical_model_toolbox/Losses/pseudo.m.")

    # test occlusion: visible_pixels cover all the pixels
    visible_pixels = np.empty((grid_size**2,2))
    for row in range(grid_size):
        for col in range(grid_size):
            visible_pixels[row*grid_size+ col,:]= np.array([row, col])
    numpy.testing.assert_almost_equal(grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, visible_pixels), -40.0308354818)
    print("SUCCESS. Occlusion works (test using all pixels).")

    visible_pixels = visible_pixels[::4,:]
    numpy.testing.assert_almost_equal(grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels, visible_pixels), -13.9461129)
    print("SUCCESS. Occlusion works (test using every 4th pixel).")

""" same test from within IPython
import sys
sys.path.append('/bigscratch/sb358/grid-exact/codes/')
import grid_pseudo_likelihood

n_labels = 3
grid_size = 5
log_node_pot = np.linspace(start=0,stop=1,num= grid_size**2 * n_labels).reshape((grid_size, grid_size, n_labels), order='F')
log_edge_pot = np.linspace(start=0,stop=1, num= n_labels**2).reshape((n_labels, n_labels), order='F')
y = np.asarray([[1, 2, 0, 0, 1],
       [1, 1, 0, 1, 2],
       [2, 1, 0, 2, 0],
       [1, 0, 1, 1, 2],
       [1, 0, 2, 1, 0]], dtype=np.int8) # generated randomly using np.random.randint(0, n_labels, size=(grid_size, grid_size))

numpy.testing.assert_almost_equal(grid_pseudo_likelihood.grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels), -40.0308354818)
%timeit -n 1000 grid_pseudo_likelihood.grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels)
"""


"""
% Matlab comparison test
% run from within GPStruct/codes/segmentation_dags

addpath('../../external/graphicalmodel_toolbox_justin/Models');
addpath('../../external/graphicalmodel_toolbox_justin/Util');
addpath('../../external/graphicalmodel_toolbox_justin/Losses');
n_labels=3;
grid_size=5;
model = gridmodel(grid_size, grid_size, n_labels);
log_edge_pot = linspace(0,1,n_labels^2)';
log_node_pot = reshape(linspace(0,1, grid_size^2 * n_labels), grid_size^2, n_labels)';
all_bin_factors = repmat(log_edge_pot,1,model.ncliques);
y = [[1, 2, 0, 0, 1];
       [1, 1, 0, 1, 2];
       [2, 1, 0, 2, 0];
       [1, 0, 1, 1, 2];
       [1, 0, 2, 1, 0]] + 1; %repmat([0 1 0 1 0] + 1,grid_size, 1);
- pseudo(model, all_bin_factors, log_node_pot, y) % will give same result as python code above

"""


r""" random Matlab/ Py testing, to run after the above

# IPy
log_node_pot = np.random.rand(grid_size, grid_size, n_labels)
log_edge_pot = np.random.rand(n_labels, n_labels)
y = np.random.randint(0,2, size=(grid_size, grid_size))
import scipy.io
scipy.io.savemat('/bigscratch/sb358/grid-exact/codes/PL_test_data.mat', {'y' : y, 'log_edge_pot' : log_edge_pot, 'log_node_pot' : log_node_pot})

import grid_libdai_inference
print("exact: %g" % grid_libdai_inference.grid_exact_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels))
print("pseudo li : %g" % grid_pseudo_likelihood.grid_pseudo_likelihood(log_node_pot, log_edge_pot, y, grid_size, n_labels))

% now retrieve the .mat file from Fermat

% MATLAB
load('C:\Users\bratieress\Documents\PERSONAL\Google Drive\Cambridge\GPstruct\GPstruct_python\src\PL_test_data.mat')
y = y+1;
all_bin_factors = repmat(log_edge_pot(:),1,model.ncliques); 
- pseudo(model, all_bin_factors, reshape(log_node_pot, grid_size^2, n_labels)', y)
% above 3 lines in 1 line:
% y=y+1; all_bin_factors = repmat(log_edge_pot(:),1,model.ncliques); - pseudo(model, all_bin_factors, reshape(log_node_pot, grid_size^2, n_labels)', y)


addpath('/home/sb358/exp-nlpfast/external/graphicalmodel_toolbox_justin/Models');
addpath('/home/sb358/exp-nlpfast/external/graphicalmodel_toolbox_justin/Util');
addpath('/home/sb358/exp-nlpfast/external/graphicalmodel_toolbox_justin/Losses');
addpath('/home/sb358/exp-nlpfast/external/graphicalmodel_toolbox_justin/Inference');
addpath('/home/sb358/exp-nlpfast/external/libDAI-0.3.1/matlab/')
addpath('/bigscratch/sb358/grid-exact/matlab/')
load('PL_matlab_pseudo')

pseudo(model, fxx, fxy, matlab_y)


"""