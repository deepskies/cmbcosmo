import numpy as np

# ----------------------------------------------------------------------
class setup_sbi(object):
    """"
    Class to handle various steps for SBI.
    """
    # ---------------------------------------------
    def __init__(self, theory):
        """
        * theory: theory object, initialized
        """
        self.theory = theory

    # ---------------------------------------------
    def setup_prior(self, param_priors):
        """
        * param_priors: arr: arr of priors on the params to constrain
        """
        print(f'## setting up prior ..')
        from sbi import utils as utils
        self.npar = len(param_priors)
        low = [param_priors[i][0] for i in range(self.npar)]
        high = [param_priors[i][1] for i in range(self.npar)]
        self.prior = utils.BoxUniform(low=low, high=high)
    # ---------------------------------------------
    def setup_posterior(self, nsims):
        """
        * nsims: int: nsims to run
        * method: str: method to use for infer. Default: 'SNPE'
        """
        print(f'## setting up posterior ..')
        from sbi.inference.base import infer
        self.posterior = infer(simulator=self.theory.get_prediction,
                               prior=self.prior,
                               method='SNPE',
                               num_simulations=nsims, 
                               )
    # ---------------------------------------------
    def get_samples(self, nsamples, datavector):
        """
        * nsamples: int: nsamples to draw from the posterior
        * datavector: arr: stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE
        """
        print(f'## getting samples ..')
        samples = self.posterior.sample(sample_shape=(nsamples,),
                                        x=datavector
                                        )
        return samples.cpu().detach().numpy()