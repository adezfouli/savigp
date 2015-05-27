# will work with dict
# keeps only the last result in memory at any given time (useful for very big results)
# will hash input using str() function (cos it should work on small dicts)
class memoize_once(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        h = str(args)
        if h in self :
            return self[h]
        else:
            self.clear() # will clear dict before storing new key
            result = self[h] = self.func(*args)
            return result
            
import learn_predict
def read_randoms(n=-1, type=None, should=None, true_random_source=True):
    """
    to intialize this function:
    read_randoms.offset=0 #DEBUG
    read_randoms.file = np.loadtxt('/tmp/sb358/ess_randoms.txt') #DEBUG
    
    This function was written for an easy switch between obtaining pseudo-random numbers from
    - a PRNG
    - a file where a sequence of such PRN is stored (to allow reusing the same random sequence between Matlab and Python, in my case)
    """
    global dtype
    if true_random_source:
        if type != None:
            if type=='u':
                result=read_randoms.prng.rand(n)
            else:
                result=read_randoms.prng.randn(n)
        else:
            return # type==None, but we're generating random numbers so can't check anything
    else:
        if (n == -1 and should != None):
            n = len(should)
        result = read_randoms.file[read_randoms.offset:read_randoms.offset+n]
        if should != None:
            #print("testing start offset : " + str(read_randoms.offset) + ", length : " + str(n))
            try:
                numpy.testing.assert_almost_equal(should, result)
            except AssertionError as e:
                raise e
            
        read_randoms.offset = read_randoms.offset+n
    return learn_predict.dtype_for_arrays(result)


import time
def stop_check(delay = None):
    if (delay != None):
        stop_check.stop_time = delay + time.time()
    else:
        if stop_check.stop_time < time.time():
            return True
        else:
            return False

import numpy as np
import prepare_from_data_chain
import prepare_from_data_grid

def run_job(lbview, params_list, common_params, chain=True, results_folder = '/scratch/sb358/pygpstruct-chain/results'):
    import hashlib
    import time
    import os.path
    
    hash = hashlib.sha1()
    hash.update(str(time.time()))
    job_hash = hash.hexdigest()[:10]
    
    for params in params_list: # each of params is a set of params for a given run
        param_as_filename = ''
        for (k,v) in params.iteritems():
            if isinstance(v, np.ndarray):
                v_string = "range" + str(v[0]) + "-" + str(v[-1] + 1)
            else:
                v_string = str(v)
            param_as_filename += '++' + str(k) + "=" + v_string
        param_as_filename = param_as_filename.replace(' ', '').replace('(','').replace(')','').replace(',','-')
                
        params.update(common_params)
        params['result_prefix'] = os.path.join(results_folder, job_hash, param_as_filename) + "."
        param_string = str(params)
    print("example params:   " + param_string) # will print from last param in for-loop; will not print experimental_conditions_string
    if (chain == True): 
        asr = lbview.map_async(lambda args : prepare_from_data_chain.learn_predict_gpstruct_wrapper(**args), params_list)
    else:
        asr = lbview.map_async(lambda args : prepare_from_data_grid.learn_predict_gpstruct_wrapper(**args), params_list)    
    print("started " + job_hash)
    return asr

# code to plot figures from file results.txt

import numpy as np
import matplotlib.pyplot as plt
import glob

def read_data(file_pattern, data_col, max_display_length):

    data_sets = []
    print("file_pattern: " + file_pattern )
    files = glob.glob( file_pattern )
    if (files == []):
        print("Error: no matching files !")
    else:
        print("matching files: " + str(files))
    for file_name in files:
        if data_col == None:  # matlab
            data_from_file = np.loadtxt(file_name)
        else: # python
            if (file_pattern.endswith(".txt")):
                data_from_file = np.loadtxt(file_name)[:,data_col]
            else:
                with open(file_name, 'rb') as f:
                    data_from_file = np.fromfile(f, dtype=np.float32)
                data_from_file = data_from_file.reshape((-1,5))[:,data_col]
        if (data_from_file.shape[0] < max_display_length):
            max_display_length = data_from_file.shape[0]
        data_sets.append(data_from_file)
    data_sets = [data_from_file[:max_display_length] for data_from_file in data_sets]
    data = np.vstack(data_sets).T
    return data
    
def plot_data(data, color, label, ax, linestyle):
    iterations_per_log_line = 1
    t = np.arange(0,data.shape[0] * iterations_per_log_line, iterations_per_log_line)
    #ax.plot(t, data, linestyle=linestyle, lw=0.5, color=color, label='%s, shuffle' % label)
    ax.plot(t, data.mean(axis=1), lw=1, label='%s' % label, linestyle=linestyle, color='black') 

def make_figure(data_col_list, file_pattern_list, bottom=None, top=None, max_display_length=1e6, save_pdf=False):
    for data_col in data_col_list: # new figure for each data type/ column
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        
        linestyles = ['-', '--', '-.', ':']
        linestyle_index = 0
        for (file_pattern_legend, file_pattern) in file_pattern_list: # new curve for each file group
            data = read_data(file_pattern, data_col, max_display_length)
            plot_data(data, 'black', file_pattern_legend, ax, linestyles[linestyle_index])
            linestyle_index += 1
    
        ax.set_xlabel('MCMC iterations')
        data_col_legend = {None: 'Matlab error rate', 
                           4: 'per-atom average negative log marginal',
                           3: 'error rate', 
                           2: 'scaled LL test set',
                           1: 'current error rate on training set',
                           0: 'current LL train set'}
        ax.set_ylabel(data_col_legend[data_col])
        ax.set_ylim(bottom=bottom, top=top)
        ax.legend() #loc='upper right')
        
        if save_pdf:
            import matplotlib
            matplotlib.rcParams['pdf.fonttype'] = 42 # to avoid PDF Type 3 fonts in resulting plots, cf http://www.phyletica.com/?p=308
            fig.savefig('figure.pdf',bbox_inches='tight');