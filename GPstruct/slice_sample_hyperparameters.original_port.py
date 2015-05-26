import numpy as np

def hp_sample(theta, ff, Lfn, Kfn, slice_width=10, 
    theta_Lprior = lambda _lhp : np.log(1e-9+(_lhp['length_scale']>np.log(0.1) and (_lhp['length_scale']<np.log(10))))
    ):
#function [theta, ff, U] = update_theta_aux_chol(theta, ff, Lfn, Kfn, cardY, slice_width, theta_Lprior, U)
#UPDATE_THETA_AUX_CHOL MCMC update to GP hyperparam. Fixes nu used to draw f, rather than f itself
# ported to Python from Iain's Matlab code, Sebastien Bratieres August 2014

    Ufn = lambda theta : np.linalg.cholesky(Kfn(theta)) # can optimize that
    U = Ufn(theta)

    nu = np.linalg.solve(U.T,ff)

    # Slice sample theta|nu
    class particle:
        pass
    particle.pos = theta
    particle = eval_particle(particle, np.NINF, nu, Lfn, theta_Lprior, U=U)
    step_out = (slice_width > 0)
    slice_width = abs(slice_width)
    slice_fn = lambda pp, Lpstar_min : eval_particle(pp, Lpstar_min, nu, Lfn, theta_Lprior, Ufn=Ufn)
    particle = slice_sweep(particle, slice_fn, slice_width, step_out)
    
    # return values
    theta = particle.pos
    ff = particle.ff
    U = particle.U
    return (theta, ff, particle.Lfn_ff)

def eval_particle(pp, Lpstar_min, nu, Lfn, theta_Lprior, Ufn = None, U = None):
# U is a precomputed chol(Kfn(pp.pos)) or a function that will compute it

# Prior
    theta = pp.pos
    Ltprior = theta_Lprior(theta)
    if Ltprior == np.NINF:
        pp.on_slice = false
        return
    if U == None:
        U = Ufn(theta)
    ff = np.dot(nu.T,U).T

    pp.Lfn_ff = Lfn(ff)
    pp.Lpstar = Ltprior + pp.Lfn_ff
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    pp.U = U
    pp.ff = ff
    return pp

def slice_sweep(particle, slice_fn, sigma=1, step_out=True):
# %SLICE_SWEEP one set of axis-aligned slice-sampling updates of particle.pos
# %
# %     particle = slice_sweep(particle, slice_fn[, sigma[, step_out]])
# %
# % The particle position is updated with a standard univariate slice-sampler.
# % Stepping out is linear (if step_out is true), but shrinkage is exponential. A
# % sensible strategy is to set sigma conservatively large and turn step_out off.
# % If it's hard to set a good sigma though, you should leave step_out=true.
# %
# % Inputs:
# %     particle   sct   Structure contains:
# %                              .pos - initial position on slice as Dx1 vector
# %                                     (or any array)
# %                           .Lpstar - log probability of .pos (up to a constant)
# %                         .on_slice - needn't be set initially but is set
# %                                     during slice sampling. Particle must enter
# %                                     and leave this routine "on the slice".
# %     slice_fn   @fn   particle = slice_fn(particle, Lpstar_min)
# %                      If particle.on_slice then particle.Lpstar should be
# %                      correct, otherwise its value is arbitrary.
# %        sigma (D|1)x1 step size parameter(s) (default=1)
# %     step_out   1x1   if non-zero, do stepping out procedure (default), else
# %                      only step in (saves on fn evals, but takes smaller steps)
# %
# % Outputs:
# %     particle   sct   particle.pos and .Lpstar are updated.

#% Originally based on pseudo-code in David MacKay's text book p375
#% Iain Murray, May 2004, January 2007, June 2008, January 2009
# Sebastien Bratieres ported to Python August 2014

#    DD = particle.pos.shape[0] # dimensionality of parameter space
#if length(sigma) == 1
#    sigma = repmat(sigma, DD, 1);
#end
# Note: in Iain's code, sigma can be an array of step-sizes, aligned with particle.pos which is the array theta. In my code, theta is a dict. I haven't ported the feature allowing sigma to be an array. So here, the step-size is equal for all hyperparameters.

# A random order (in hyperparameters) is more robust generally and important inside algorithms like nested sampling and AIS
    for (dd, x_cur) in [('length_scale', particle.pos['length_scale'])]: #particle.pos.items():
        #print('working in param %s' % dd)
        Lpstar_min = particle.Lpstar + np.log(np.random.rand())

        # % Create a horizontal interval (x_l, x_r) enclosing x_cur
        rr = np.random.rand()
        x_l = x_cur - rr*sigma
        x_r = x_cur + (1-rr)*sigma
        if step_out:
            particle.pos[dd] = x_l
            while True:
                particle = slice_fn(particle, Lpstar_min)
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] - sigma
            # print('placed x_l')
            x_l = particle.pos[dd]
            particle.pos[dd] = x_r
            while True:
                particle = slice_fn(particle, Lpstar_min)
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] + sigma
            # print('placed x_r')

            x_r = particle.pos[dd]

        #% Make proposals and shrink interval until acceptable point found
        #% One should only get stuck in this loop forever on badly behaved problems,
        #% which should probably be reformulated.
        while True:
            particle.pos[dd] = np.random.rand()*(x_r - x_l) + x_l
            particle = slice_fn(particle, Lpstar_min)
            if particle.on_slice:
                break # Only way to leave the while loop.
            else:
                # Shrink in
                if particle.pos[dd] > x_cur:
                    x_r = particle.pos[dd]
                elif particle.pos[dd] < x_cur:
                    x_l = particle.pos[dd]
                else:
                    raise Exception('BUG DETECTED: Shrunk to current position and still not acceptable.')
    return particle