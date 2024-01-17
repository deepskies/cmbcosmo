import numpy as np

# ----------------------------------------------------------------------
class setup_sbi(object):
    """"

    Class to handle various steps for SBI.

    """
    # ---------------------------------------------
    def __init__(self, theory, param_labels_in_order):
        """

        * theory: theory object, initialized

        """
        self.theory = theory
        self.param_labels_in_order = param_labels_in_order

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
    def simulator(self, params):

        param_dict = {}
        for i, key in enumerate(self.param_labels_in_order):
            param_dict[key] = params[i]

        return self.theory.get_prediction(param_dict=param_dict)

    # ---------------------------------------------
    def setup_posterior(self, nsims):
        """

        * nsims: int: nsims to run

        """
        print(f'## setting up posterior ..')
        from sbi.inference.base import infer
        self.posterior = infer(simulator=self.simulator,
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