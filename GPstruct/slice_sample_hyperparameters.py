import numpy as np
import util

def update_theta_simple(theta, ff, Lfn, Ufn, 
    slice_width=10, 
    theta_Lprior = lambda _lhp_target : 0 if np.all(_lhp_target>np.log(0.1)) and np.all(_lhp_target<np.log(10)) else np.NINF
    ):
    """
    update to GP hyperparam. slice sample theta| f, data.
    ported to Python from Iain's Matlab code, Sebastien Bratieres August 2014
    """

    U = Ufn(theta)

    # Slice sample theta|ff
    class particle:
        pass
    particle.pos = theta
    particle.ff = ff
    slice_fn = lambda pp, Lpstar_min : eval_particle_simple(pp, Lfn, theta_Lprior, Lpstar_min, Ufn)
    slice_fn(particle, Lpstar_min = np.NINF)
    slice_sweep(particle, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    return (particle.pos, particle.ff, particle.Lfn_ff) # could return U !? so you don't have to recompute it ?


def eval_particle_simple(pp, Lfn, theta_Lprior, Lpstar_min, Ufn): # should not need to return particle, can modify in-place
    """
    pp modified in place
    U is a precomputed chol(Kfn(pp.pos))
    alternatively, Ufn is a function that will compute U
    """

    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF: # save time in case Ltprior is NINF, don't need to run Ufn
        #print('off slice cos prior limit hit')
        pp.Lpstar = Ltprior
        pp.on_slice = False
        return 
    U = Ufn(pp.pos)
    L_inv_f = U.T_solve(pp.ff) # equivalent of L.solve. NB all my U variables are actually lower triangular and should be renamed!
    Lfprior = -0.5 * L_inv_f.T.dot(L_inv_f) - U.diag_log_sum(); # + const 
    # log(p(f|theta)) = log(N(pp.ff ; 0, U_theta)) = -1/2 f.T (U.T.dot(U))^1 f - log(sqrt(2 * pi * det(U.T.dot(U)))) 

    pp.Lfn_ff = Lfn(pp.ff)
    pp.Lpstar = pp.Lfn_ff + Lfprior + Ltprior # p(x|f) + p(f|theta) + p(theta)
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    pp.U = U
    
def update_theta_aux_chol(theta, ff, Lfn, Ufn, 
    slice_width=10, 
    theta_Lprior = lambda _lhp_target : np.log((np.all(_lhp_target>np.log(0.1)) and np.all(_lhp_target<np.log(10))))
    ):
    """
    update to GP hyperparam. Fixes nu used to draw f, rather than f itself
    ported to Python from Iain's Matlab code, Sebastien Bratieres August 2014
    """

    U = Ufn(theta)
    nu = U.T_solve(ff)

    # Slice sample theta|nu
    class particle:
        pass
    particle.pos = theta
    slice_fn = lambda pp, Lpstar_min : eval_particle_aux_chol(pp, nu, Lfn, theta_Lprior, Lpstar_min, Ufn)
    slice_fn(particle, Lpstar_min = np.NINF)
    slice_sweep(particle, slice_fn, sigma = abs(slice_width), step_out = (slice_width > 0))
    return (particle.pos, particle.ff, particle.Lfn_ff)

def eval_particle_aux_chol(pp, nu, Lfn, theta_Lprior, Lpstar_min, Ufn): # should not need to return particle, can modify in-place
    """
    pp modified in place
    U is a precomputed chol(Kfn(pp.pos))
    alternatively, Ufn is a function that will compute U
    """

    Ltprior = theta_Lprior(pp.pos)
    if Ltprior == np.NINF:
        pp.on_slice = False
        return
    U = Ufn(pp.pos)
    ff = U.T_dot(nu) # ff = np.dot(nu.T,U).T

    pp.Lfn_ff = Lfn(ff)
    pp.Lpstar = Ltprior + pp.Lfn_ff
    pp.on_slice = (pp.Lpstar >= Lpstar_min)
    pp.U = U
    pp.ff = ff

def slice_sweep(particle, slice_fn, sigma=1, step_out=True):# should not need to return particle, can modify in-place
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
    import util
    # A random order (in hyperparameters) is more robust generally and important inside algorithms like nested sampling and AIS
    for (dd, x_cur) in enumerate(particle.pos):
        #print('working in param %s' % dd)
        Lpstar_min = particle.Lpstar + np.log(util.read_randoms(1, 'u'))
        #print('particle.on_slice? %g' % particle.on_slice)
        # % Create a horizontal interval (x_l, x_r) enclosing x_cur
        rr = util.read_randoms(1, 'u')
        x_l = x_cur - rr*sigma
        x_r = x_cur + (1-rr)*sigma
        if step_out:
            #print('stepping out left with Lpstar_min=%g' % Lpstar_min)
            particle.pos[dd] = x_l
            while True:
                slice_fn(particle, Lpstar_min)
            #    print('on-slice %g is (particle.Lpstar = %g >= Lpstar_min = %g)' % (particle.on_slice, particle.Lpstar, Lpstar_min))
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] - sigma
            #print('placed x_l, now stepping out right')
            x_l = particle.pos[dd]
            particle.pos[dd] = x_r
            while True:
                slice_fn(particle, Lpstar_min)
                if not particle.on_slice:
                    break
                particle.pos[dd] = particle.pos[dd] + sigma
            #print('placed x_r, particle.on_slice? %g' % particle.on_slice)

            x_r = particle.pos[dd]

        #% Make proposals and shrink interval until acceptable point found
        #% One should only get stuck in this loop forever on badly behaved problems,
        #% which should probably be reformulated.
        while True:
            particle.pos[dd] = util.read_randoms(1, 'u')*(x_r - x_l) + x_l
            slice_fn(particle, Lpstar_min)
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