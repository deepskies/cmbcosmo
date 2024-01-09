import numpy as np
import emcee as emcee
import shutil
# ----------------------------------------------------------------------
class setup_mcmc(object):
    """"
    Class to handle various functions needed for MCMC.
    Covariances missing right now.
    """
    # ---------------------------------------------
    def __init__(self, datavector, cov, param_priors, theory, outdir):
        """
        * datavector: arr: stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE
        * param_priors: arr: arr of priors on the params to constrain
        * theory: theory object, initialized
        * outdir: str: path to the output dir
        """
        self.datavector = datavector
        self.covinv = np.linalg.pinv(cov)
        self.param_priors = param_priors
        self.npar = len(param_priors)
        self.theory = theory
        self.outdir = outdir
    # ---------------------------------------------
    def get_loglikelihood(self, theory_vec):
        """
        * theory_vec: arr: theory array (stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE)
                           to compare against data
        """
        # diff
        delta = theory_vec - self.datavector
        # calculate chi2
        chi2 = np.linalg.multi_dot([delta, self.covinv, delta])

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
    # set up the sampler, burn in, post-burn
    def run_mcmc(self, nwalkers,
                 starts, nsteps_burn, nsteps_post,
                 restart=False, restart_from_burn=False,
                 progress=True):
        """
        * nwalkers: int: number of walkers
        * npar: int: number of parameters
        * starts: arr: array of starting positions of the walkers
        * nsteps_burnin: int: number of steps for burn-in
        * nsteps_post: int: number of steps for post-burn-in
        * restart: bool: set to True to restart using the backend
                         Default: False
        * restart_from_burn: bool: set to True restart from burn in
                                   Default: False
        * progress: bool: set to False to not show progress bar.
                          Default: True
        """
        # setup sampler backend
        backend_burnin_fname = f'{self.outdir}/backend-burnin.h5'
        backend_fname = f'{self.outdir}/backend.h5'
        backend = emcee.backends.HDFBackend(backend_fname)
        # set up the sampler
        sampler = emcee.EnsembleSampler(nwalkers, self.npar,
                                        self.get_logposterior,
                                        backend=backend)
        # figure out where to start from
        if restart:
            if restart_from_burn:
                print('## resuming burn in ... ')
                # run the chain; n-steps modified based on how many were completed before
                pos, _, _ = sampler.run_mcmc(None,
                                             nsteps_burn - backend.iteration,   progress=progress)
                # save the backend for the burn in
                shutil.copy(backend_fname, backend_burnin_fname)
                # now reset the sampler
                sampler.reset()
                # run post-burn
                print('## running the full chain ... ')
                sampler.run_mcmc(None, nsteps_post, progress=progress)
            else:
                print('## resuming the chain ... ')
                # run the chain; n-steps modified based on how many were completed before
                sampler.run_mcmc(None, nsteps_post - backend.iteration, progress=progress)
        else:
            # ------
            print('## burning in ... ')
            # run burn-in
            pos, _, _ = sampler.run_mcmc(starts, nsteps_burn, progress=progress)
            # ------
            # save the backend for the burn in
            shutil.copy(backend_fname, backend_burnin_fname)
            # now reset the sampler
            sampler.reset()
            # run post-burn
            print('## running the full chain ... ')
            sampler.run_mcmc(pos, nsteps_post, progress=progress)

        # for later
        self.sampler = sampler
        self.nsteps_burn = nsteps_burn
        self.nsteps_post = nsteps_post
        self.starts = starts
         # save the burnin sampler
        backend_burnin = emcee.backends.HDFBackend(backend_burnin_fname)
        self.sampler_burnin = emcee.EnsembleSampler(nwalkers, self.npar,
                                                    self.get_logposterior,
                                                    backend=backend_burnin)

    # ---------------------------------------------
    # get samples
    def get_samples(self, flat=True):
        """
        * progress: bool: set to False to get the 2D array.
                          Default: True
        """
        return self.sampler.get_chain(flat=flat)
    # ---------------------------------------------------------------------
    def plot_chainvals(self, truths, param_labels):
        from helpers_plots import plot_chainvals
        # first the chain
        plot_chainvals(chain_unflattened=self.sampler.get_chain(),
                       outdir=self.outdir, npar=self.npar, nsteps=self.nsteps_post,
                       starts=self.starts, truths=truths, param_labels=param_labels, filetag='post-burnin')
        # now the burnin
        plot_chainvals(chain_unflattened=self.sampler_burnin.get_chain(),
                       outdir=self.outdir, npar=self.npar, nsteps=self.nsteps_post,
                       starts=self.starts, truths=truths, param_labels=param_labels, filetag='burnin')