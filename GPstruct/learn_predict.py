from __future__ import division
from __future__ import print_function
import numpy as np
import logging
import time
import ess_k_sampler
import sys
import os
import numpy.testing
import pickle
import glob
import slice_sample_hyperparameters
import util
import kernels

dtype_for_arrays=np.float32
    
def learn_predict_gpstruct( prepare_from_data,
                            result_prefix=None, 
                            console_log=True,
                            n_samples=0, 
                            prediction_thinning=1,
                            prediction_verbosity=None,
                            n_f_star=0, 
                            hp_sampling_thinning=1, 
                            hp_sampling_mode=None, 
                            prior=1, 
                            lhp_update={}, # defaults are     lhp = {'unary': np.log(1), 'binary': np.log(0.01), 'length_scale': np.log(8), 'noise' : np.log(1e-4)}
                            kernel=kernels.kernel_exponential_unary,
                            random_seed=0,
                            stop_check=None, 
                            hp_debug=False
                            ):
    """
    result_prefix should end with the desired character to allow result_prefix + string constructions:
    end in / for directory: will put files into result_prefix directory
    end in . for file prefix (ie result_prefix = result_dir + '/' + file_prefix: will put files into result_dir, with filenames prefixed with file_prefix
    
    # n_samples: # f samples, ie # of MCMC iterations
    # prediction_thinning: #samples f (MCMC iterations) after which to compute f* and p(y*) - eg 1 or 10 (the f samples which are not used here are thrown away)
    # hp_thinning: #samples of f after which to resample hyperparams
    # n_f_star: # f* samples given f (0 = MAP) - eg 2
    # hp_mode: 0 for no hp sampling, 1 for prior whitening, 2 for surrogate
    %   data/null aux
    % prior: 1 for narrow, 2 for wide uniform prior
    default hyperparameters:     lhp = {'unary': np.log(1), 'binary': np.log(0.01), 'length_scale': np.log(8), 'noise' : np.log(1e-4)}

    """
    global dtype
    function_args = locals() # store just args passed to function, so as to log them later on
    
    ## test for hotstart
    if (glob.glob(result_prefix + 'state.pickle') != []): # hotstart
        hotstart=True # no safety net prevents hotstarting with different parameters. Could do: store parameters dict, check at hotstart that it is identical. Hard to do: just have an "hotstart this" feature which works based on a results folder; implies re-obtain things like dataset, kernel matrices.
    else:
        # make results dir
        # mkdir result_prefix in case it doesnt exist, from http://stackoverflow.com/questions/16029871/how-to-run-os-mkdir-with-p-option-in-python second answer updated !
        hotstart=False
        try:
            os.makedirs(os.path.split(result_prefix)[0])
        except OSError as err:
            if err.errno!=17:
                raise

    ## logging initialization
    # logging tree blog post http://rhodesmill.org/brandon/2012/logging_tree/
    logger = logging.getLogger('pyGPstruct') # reuse logger if existing
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = [] # remove any existing handlers
    # create formatter
    formatter = logging.Formatter('%(asctime)sZ - %(levelname)s - %(message)s')
    formatter.converter=time.gmtime # will use GMT timezone for output, hence the Z
    # console log handler
    if (console_log):
        ch = logging.StreamHandler(stream=sys.stdout)
        #ch.setLevel(logging.DEBUG) # seems not needed by default 
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    fh = logging.FileHandler(result_prefix + "log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # logging all input parameters cf http://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function -- bottom line use locals() which is a dict of all local vars
    logger.info('learn_predict_gpstruct started with arguments: ' + str(function_args))

    
#    if errorThinning < prediction_thinning:
#        logger.error("Value for errorThinning is %g, smaller than prediction_thinning, which is %g. This is illegal. Using %g instead.\n" % errorThinning, prediction_thinning, prediction_thinning)

    if (stop_check == None):
        stop_check = lambda : None # equivalent to pass
    results = []
    util.read_randoms.offset=0 #DEBUG
    #read_randoms.file = np.loadtxt('/tmp/sb358/ess_randoms.txt') #DEBUG

    (ll_train, 
     posterior_marginals_test, 
     compute_error_nlm, 
     ll_test, 
     average_marginals, 
     write_marginals,
     read_marginals,
     n_labels, 
     X_train, 
     X_test) = prepare_from_data(logger=logger) 
    # posterior_marginals_test is a function f -> posterior marginals
    # compute_error_nlm(marginals) returns error and avg post marginals
    # ll_test(current f) returns LL of test data
    
    TT_train = X_train.shape[0]
    TT_test = X_test.shape[0]
    #read_randoms(len(X_train.todense().flatten(order='F').T), should=np.squeeze(np.array(X_train.todense()).flatten(order='F').T), true_random_source=False) # DEBUG
    #read_randoms(len(X_test.todense().flatten(order='F').T), should=np.squeeze(np.array(X_test.todense()).flatten(order='F').T), true_random_source=False) # DEBUG
    # prepare kernel matrix
    logger.debug("prepare kernel matrices")

    # override default hyperparameters with argument lhp_update
    lhp = {'unary': np.log(1), 'binary': np.log(0.01), 'length_scale': np.log(8), 'noise' : np.log(1e-4)}
    lhp.update(lhp_update)
    compute_kernels = lambda lhp : kernels.compute_kernels_from_data(kernel, lhp, X_train, X_test, n_labels)
    (lower_chol_k_compact, k_star_T_k_inv) = compute_kernels(lhp)
    
    logger.debug("start MCMC chain")
    if (prediction_verbosity != None):
        prediction_at = np.round(np.linspace(start=0, stop=n_samples-1, num=prediction_verbosity))

    if hotstart: # restore state from disk
        with open(result_prefix + 'state.pickle', 'rb') as random_state_file:
            saved_state_dict = pickle.load(random_state_file)
        util.read_randoms.prng = saved_state_dict['prng']
        current_f = saved_state_dict['current_f']
        mcmc_step = saved_state_dict['mcmc_step']
        current_ll_train = saved_state_dict['current_ll_train']
        current_error = saved_state_dict['current_error']
        current_ll_test = saved_state_dict['current_ll_test']
        avg_error = saved_state_dict['avg_error']
        avg_nlm = saved_state_dict['avg_nlm']
        logger.info('hotstart from iteration %g, including stored random state' % mcmc_step)
    else: # initialize state
        current_f = np.zeros(n_labels * TT_train + n_labels**2, dtype=dtype_for_arrays)
        mcmc_step=0
        util.read_randoms.prng = np.random.RandomState(random_seed)
        # no need to initialize other variables, since they will be computed during prediction, since we are starting from iteration 0 (for which we are sure prediction will happen)

    if hp_debug:
        history_f = np.ones((n_samples, current_f.shape[0]))*4 # flag
        history_ll = np.ones((n_samples)) * 4
        history_hp = np.ones((n_samples, 10)) * 4
        
    # ---------------- MCMC loop 
    while not stop_check() and (mcmc_step < n_samples or n_samples == 0):
        if hp_debug:
            history_f[mcmc_step, :] = current_f
            history_ll[mcmc_step] = ll_train(current_f)
            history_hp[mcmc_step, :] = lhp['variances']
        if not (hp_sampling_mode == None) and np.mod(mcmc_step, hp_sampling_thinning) == 0 :
            def update_lhp(lhp, variances): 
                lhp['variances'] = variances
                return lhp
            if (hp_sampling_mode == 'slice sample theta'):
                (lhp_target, current_f, current_ll_train) = slice_sample_hyperparameters.update_theta_simple(
                    theta = lhp['variances'], 
                    ff = current_f, 
                    Lfn = ll_train, 
                    Ufn = util.memoize_once(lambda _lhp_target : kernels.gram_compact(np.linalg.cholesky(kernel(X_train, X_train, update_lhp(lhp, _lhp_target), no_noise=False)), 
                                                                                      np.sqrt(np.exp(lhp["binary"])),
                                                                                      n_labels)),
                    theta_Lprior = lambda _lhp_target : np.log((np.all(_lhp_target>np.log(1e-3)) and np.all(_lhp_target<np.log(1e2)))),
                    )
                update_lhp(lhp, lhp_target) # should be already done because particle.pos (mutable) is updated in-place
            elif (hp_sampling_mode == 'prior whitening'):
                (lhp_target, current_f, current_ll_train) = slice_sample_hyperparameters.update_theta_aux_chol(
                    theta = lhp['variances'], 
                    ff = current_f, 
                    Lfn = ll_train, 
                    Ufn = util.memoize_once(lambda _lhp_target : kernels.gram_compact(np.linalg.cholesky(kernel(X_train, X_train, update_lhp(lhp, _lhp_target), no_noise=False)), 
                                                                                      np.sqrt(np.exp(lhp["binary"])),
                                                                                      n_labels)),
                    theta_Lprior = lambda _lhp_target : np.log((np.all(_lhp_target>np.log(1e-3)) and np.all(_lhp_target<np.log(1e2)))),
                    )
                update_lhp(lhp, lhp_target) # should be already done because particle.pos (mutable) is updated in-place
            else:
                raise Exception('hyperparameter sampling mode %s not supported; should be one of "prior whitening" or None.' % hp_sampling_mode)
            # recompute kernels and factors involving kernels
            #logger.debug('lhp update: %s' % str(lhp))
            (lower_chol_k_compact, k_star_T_k_inv) = compute_kernels(lhp)

        current_f, current_ll_train = ess_k_sampler.ESS(current_f, ll_train, lower_chol_k_compact, util.read_randoms) 
        #read_randoms(should=current_f, true_random_source=False)
        #current_ll_train = read_randoms(1, should=ll_train(current_f), true_random_source=False)
        
        # prediction : compute f*|D and p(y*|f)
        # - compute f*, then marginals p(y*|f*)
        # - save marginals to disk
        # - read in all marginals so far
        # - discard burnin, from remaining marginals compute Bayesian averaged error rate
        if (prediction_verbosity == None and np.mod(mcmc_step, prediction_thinning) == 0) or (prediction_verbosity != None and mcmc_step in prediction_at):

            #logger.debug("start prediction it %g" % mcmc_step)
            # compute mean of f*|D - this involves f (expanded) and k_star_T_k_inv_unary (compact), so need to iterate over n_labels
            f_star_mean = k_star_T_k_inv.dot(current_f)
            if n_f_star == 0:
                marginals_f = posterior_marginals_test(f_star_mean)
            # else:
            # % sample f*
            # % want f* sampling to not affect f sampling, so preserve randn state before this
            # f_rng_state = randn('state');
            # marginals = zeros(TT_test, n_labels, n_f_star);
            # for i=1:n_f_star
            #     marginals(:,:,i) = predictiveMarginalsN(f_star_mean + ...
            #         [sampleFFromKernel(n_labels, lowerCholfStarCov, true ) ; ...
            #         zeros(n_labels^2, 1)]);
            #     % must pad output of sample... cos it's applied to unary
            #     % matrices, while f*mean has unary and bin
            #     % NOTE: using the binaries from f, cos they are not
            #     % data-dependent, so don't need to sample them with f*|f.
            # end
            # % reinstate randn state for further f sampling
            # randn('state', f_rng_state);
            #===================================================================
            #read_randoms(should=f_star_mean, true_random_source=False) # DEBUG
            #read_randoms(should=np.vstack(marginals_f).ravel(order='F'), true_random_source=False) # DEBUG
            
            # using marginals_f, compute current error and current LL test data
            current_error = compute_error_nlm(marginals_f)[0] # discard neg log marg for prediction from single f
            #read_randoms(1, should=current_error, true_random_source=False) # DEBUG
            current_ll_test = ll_test(f_star_mean)
            #read_randoms(1, should=scaled_ll_test, true_random_source=False) # DEBUG
            
            with open(result_prefix + "marginals.bin", 'ab') as marginals_file:
                write_marginals(marginals_f, marginals_file)

            # read marginals from disk
            marginals_read = np.array([0]) # init to non-empty list
            with open(result_prefix + "marginals.bin", 'rb') as marginals_file:
                all_marginals = read_marginals(marginals_file)

            #compute        
            #    - avg_error: test set error averaged over all f*|D draws 
            #    - current_ll_test: log-likelihood of very last f*|D sample
            marginals_after_burnin = all_marginals[len(all_marginals)//3:]
            (avg_error, avg_nlm) = compute_error_nlm(average_marginals(marginals_after_burnin))
            if not hp_debug:
                logger.info(("ESS it %g -- " +
                            "LL train | last f = %.5g -- " +
                            "test set error | last f = %.5g -- " + 
                            "LL test | last f = %.5g -- " + 
                            "test set error (marginalized over f's)= %.5g -- " +
                            "average per-atom negative log posterior marginals = %.5g") % 
                            (mcmc_step, 
                             current_ll_train,
                             current_error, 
                             current_ll_test,
                             avg_error,
                             avg_nlm
                             )
                            )
            else:
                logger.info(("ESS it %g -- " +
                            "LL train | last f = %.5g -- " +
                            "test set error | last f = %.5g -- " + 
                            "LL test | last f = %.5g -- " + 
                            "test set error (marginalized over f's)= %.5g -- " +
                            "average per-atom negative log posterior marginals = %.5g -- " +
                            "lhp = %s") % 
                            (mcmc_step, 
                             current_ll_train,
                             current_error, 
                             current_ll_test,
                             avg_error,
                             avg_nlm,
                             str(lhp)
                             )
                            )
            
        mcmc_step += 1 # now ready for next iteration
        
            
        # finally save results for this MCMC step (avg_error, avg_nlp unchanged from previous step in case no prediction occurred)
        with open(result_prefix + 'results.bin', 'ab') as results_file:
            last_results = np.array([current_ll_train, 
                     current_error, 
                     current_ll_test,
                     avg_error,
                     avg_nlm], dtype=dtype_for_arrays)
            last_results.tofile(results_file) # file format = row-wise array, shape #mcmc steps * 5 float32
        
        # save state in case we are interrupted
        with open(result_prefix + 'state.pickle', 'wb') as random_state_file:
            pickle.dump({'prng' : util.read_randoms.prng,
                         'current_f' : current_f,
                         'mcmc_step' : mcmc_step,
                         'current_ll_train' : current_ll_train,
                         'current_error' : current_error, 
                         'current_ll_test' : current_ll_test,
                         'avg_error' : avg_error,
                         'avg_nlm' : avg_nlm}, 
                         random_state_file)
    fh.close()
    if hp_debug:
        return (lower_chol_k_compact, history_f, history_ll, history_hp)
        
# LATER
# - separate learning (write marginals to disk) 
#    from prediction (read, skipping burnin, applying extra thinning, and compute errors)
# - f* vs f* MAP
