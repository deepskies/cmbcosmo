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
    def loglikelihood(self, theory_vec):
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
    def logprior(self, p):
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
    def logposterior(self, p):
        """
        * p: arr: parameter array to consider
        """
        # check to confirm that this sample is good with the priors
        logprior = self.logprior(p)

        if not np.isfinite(logprior):
            # i.e. value outside the prior => unlikely
            return -np.inf

        prediction = self.theory.get_prediction(r=p)
        return self.loglikelihood(theory_vec=prediction) + logprior
    # ----------------------------------------------------------------------
