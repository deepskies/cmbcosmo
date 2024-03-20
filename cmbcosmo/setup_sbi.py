import numpy as np
import os
import pickle
import torch
# ----------------------------------------------------------------------
class setup_sbi(object):
    """"

    Class to handle various steps for SBI.

    """
    # ---------------------------------------------
    def __init__(self, theory, outdir, param_labels_in_order):
        """

        * theory: theory object, initialized

        """
        self.theory = theory
        self.outdir = outdir
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
    def setup_posterior(self, nsims, restart=False):
        """

        * nsims: int: nsims to run

        """
        print(f'## setting up posterior ..')
        from sbi.inference.base import infer
        fname = 'sbi_posterior.pickle'
        if restart:
            if not os.path.exists(f'{self.outdir}/{fname}'):
                raise ValueError(f'cant restart since {fname} not found in {self.outdir}.')
            else:
                # read in
                print(f'## reading in saved posteriors from {self.outdir}/{fname}')
                self.posterior = pickle.load( open(f'{self.outdir}/{fname}', 'rb') )
        else:
            self.posterior = infer(simulator=self.simulator,
                               prior=self.prior,
                               method='SNPE',
                               num_simulations=nsims, 
                               )
            # now save the posterior for later
            pickle.dump( self.posterior, open(f'{self.outdir}/{fname}', 'wb' ) )
            print(f'## saved posterior as {self.outdir}/{fname}')
    # ---------------------------------------------
    def get_samples(self, nsamples, datavector, seed):
        """

        * nsamples: int: nsamples to draw from the posterior
        * datavector: arr: stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE

        """
        print(f'## getting samples ..')
        _ = torch.manual_seed(seed)

        samples = self.posterior.sample(sample_shape=(nsamples,),
                                        x=datavector
                                        )
        return samples.cpu().detach().numpy()