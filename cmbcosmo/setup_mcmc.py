import numpy as np

# ----------------------------------------------------------------------
class setup_mcmc(object):
    """"
    Class to handle various functions needed for MCMC.
    Covariances missing right now.
    """
    # ---------------------------------------------
    def __init__(self, datavector, param_priors, theory):
        """
        * datavector: arr: stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE
        * param_priors: arr: arr of priors on the params to constrain
        * theory: theory object, initialized
        """
        self.datavector = datavector
        self.param_priors = param_priors
        self.npar = len(param_priors)
        self.theory = theory

    # ---------------------------------------------
    def get_loglikelihood(self, theory_vec):
        """
        * theory_vec: arr: theory array (stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE)
                           to compare against data
        """
        # diff
        #delta = theory_vec - self.datavector
        # cov
        #covinv = np.linalg.pinv(covmat)
        # calculate chi2
        #chi2 = np.linalg.multi_dot([delta, covinv, delta])

        # temporary measure; without cov
        chi2 = np.nansum((theory_vec - self.datavector)**2 / theory_vec)   
        return -0.5 * chi2
    
    # ---------------------------------------------
    def get_logprior(self, p):
        """
        * p: arr: parameter array to consider
        """
        good_to_go = True
        # loop over all params and check to make sure this sample is within the priors
        i, out = 0, 0
        while good_to_go is True and i < self.npar:
            if self.param_priors[i][0] <= p[i] <= self.param_priors[i][1]:
                out += np.log( 1 / (self.param_priors[i][1] - self.param_priors[i][0]) )
            else:
                good_to_go = False
            i += 1

        # return out if sample is within prior ranges
        if good_to_go:
            return out
        else:
            return -np.inf
    # ---------------------------------------------
    # set up log-posterior
    def get_logposterior(self, p):
        """
        * p: arr: parameter array to consider
        """
        # check to confirm that this sample is good with the priors
        logprior = self.get_logprior(p)

        if not np.isfinite(logprior):
            # i.e. value outside the prior => unlikely
            return -np.inf

        prediction = self.theory.get_prediction(r=p)
        return self.get_loglikelihood(theory_vec=prediction) + logprior
    # ---------------------------------------------
    # set up the sampler
    def setup_sampler(self, nwalkers, npar):
        """
        * nwalkers: int: number of walkers
        * npar: int: number of parameters
        """
        import emcee as emcee
        self.sampler = emcee.EnsembleSampler(nwalkers, npar, self.get_logposterior)
    # ---------------------------------------------
    # run burn in
    def burnin(self, starts, nsteps, progress=True):
        """
        * starts: arr: array of starting positions of the walkers
        * nsteps: int: number of steps for burn-in
        * progress: bool: set to False to not show progress bar.
                          Default: True
        """
        print('## burning in ... ')
        # run burn-in
        self.latest_walker_coords, _, _ = self.sampler.run_mcmc(starts, nsteps, progress=progress)
        # now reset the sampler
        self.sampler.reset()
    # ---------------------------------------------
    # run MCMC post burn-in
    def post_burn(self, nsteps, progress=True):
        """
        * nsteps: int: number of steps for post-burn-in
        * progress: bool: set to False to not show progress bar.
                          Default: True
        """
        print('## running the full chain ... ')
        self.sampler.run_mcmc(self.latest_walker_coords, nsteps, progress=progress)
    # ---------------------------------------------
    # get samples
    def get_samples(self, flat=True):
        """
        * progress: bool: set to False to get the 2D array.
                          Default: True
        """
        return self.sampler.get_chain(flat=flat)
    # ---------------------------------------------------------------------