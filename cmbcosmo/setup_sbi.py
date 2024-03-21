import numpy as np
import os
import pickle
import torch
import time
from tqdm import tqdm
import cmbcosmo.settings
import matplotlib.pyplot as plt
from sbi.analysis import pairplot
# ----------------------------------------------------------------------
class setup_sbi(object):
    """"

    Class to handle various steps for SBI.

    """
    # ---------------------------------------------
    def __init__(self, theory, outdir, param_labels_in_order):
        """

        * theory: theory object, initialized
        * outdir: str: path to the output dir
        * param_labels_in_order: list: list of str to specify the order
                                       of params. will be used to set up the
                                       the dictionary for predictions; order
                                       specified assumed as the order used
                                       for sbi.

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
        """

        * params: arr: arr of params to reproduce the "sim" for

        """
        param_dict = {}
        for i, key in enumerate(self.param_labels_in_order):
            param_dict[key] = params[i]

        return self.theory.get_prediction(param_dict=param_dict)

    # ---------------------------------------------
    def setup_posterior(self, nsims, restart=False):
        """

        * nsims: int: nsims to run
        * restart: bool: set to True to read posterior from disk.
                         Default: False

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
        * datavector: arr: stacked spectra
        * seed: int: seed to be used for generating samples

        """
        print(f'## getting samples ..')
        _ = torch.manual_seed(seed)

        samples = self.posterior.sample(sample_shape=(nsamples,),
                                        x=datavector
                                        )
        return samples.cpu().detach().numpy()

    # ---------------------------------------------
    def _pred_check_helper(self, samples, samples_tag,
                           datavector, datavector_param_dict,
                           subset_inds_to_plot, additional_tag=None
                           ):
        """

        helper function to deal with the various plots for the
        predictive checks.

        note: have checked this code only with 1spectrum so will
        throw an error if trying to run it with >1 spectrum since
        that functionality is untested.

        * samples: arr: array of samples from either the posterior
                        or prior.
        * samples_tag: str: tag for the samples: 'prior', 'posterior'
        * datavector: arr: datavector to compare against
        * datavector_param_dict: dict: dictionary used to generate datavector.
        * subset_inds_to_plot: arr: indices to consider when plotting the cls
                                    in the pairplot; more than 10 is likely
                                    not a good idea. Could be None but beware
                                    of runtime associated with plotting an
                                    impossibly large plot.
        * additional_tag: str: any additional tags to be added to the outfiles'
                               name. Default: None

        """
        # check to ensure that we're working with just one spectrum.
        cls_tags = self.theory.keys_of_interest
        if len(cls_tags) > 1:
            err = '## dont have the functionality to run this for more than 1spec:'
            err += f' got: {len(cls_tags)}'
            raise ValueError(err)

        if additional_tag is None: additional_tag = ''
        else: additional_tag = f'_{additional_tag}'
        nsamples = len(samples)
        # pairplot to check what samples were drawn for PPC
        _, axes = pairplot(samples=samples,
                           offdiag=["kde"],
                           diag=["kde"],
                           labels=self.param_labels_in_order,
                           figsize=(self.npar * 2, self.npar * 2),
                           )
        # add lines for true params
        for ind, par in enumerate(self.param_labels_in_order):
            axes[ind, ind].axvline(x=datavector_param_dict[par],
                                   color='k', ls='--', lw=2)
        # title
        plt.suptitle(f'{samples_tag} predictive check - {nsamples} nsamples')
        # save fig
        fname = f'plot_{samples_tag}-pred-check_samples{additional_tag}.png'
        plt.savefig(f'{self.outdir}/{fname}', format='png', bbox_inches='tight')
        print('## saved %s' % fname )
        plt.close()

        # now generate data
        print(f'## starting data generation using the {samples_tag} samples ...')
        x_pp = []
        for pars in tqdm(samples.tolist()):
            dict_ = { f: pars[i] for i, f in enumerate(self.param_labels_in_order) }
            x_pp.append(self.theory.get_prediction(dict_))

        # get ells
        ells, _, _ = self.theory.get_prediction(dict_, return_ell_keys_too=True)

        # lets extract the subset if specified for the pairplot
        print(f'## extracting subset as needed ..')
        if subset_inds_to_plot is not None:
            x_pp_subset = []
            for nth in range(len(x_pp)):
                x_pp_subset.append(x_pp[nth][subset_inds_to_plot])
        else:
            x_pp_subset = x_pp
            subset_inds_to_plot = len(x_pp[0])

        ninds = len(subset_inds_to_plot)
        print(f'## working on the pairplot ...')
        x_pp_subset = np.array(x_pp_subset)
        # plot xpp vs observed data
        _, axes = pairplot(samples=np.log(x_pp_subset),
                           points=np.log(datavector.reshape(1,-1)[0]),
                           points_colors="red",
                           upper="scatter",
                           scatter_offdiag=dict(marker="."), #, s=5),
                           points_offdiag=dict(marker="+"), #markersize=15),
                           labels=[r"log($C_{%s}$)" % ells[d] for d in subset_inds_to_plot],
                           figsize=(ninds * 2, ninds * 2),
                           )
        # title
        plt.suptitle(f'{samples_tag} predictive check - {nsamples} nsamples')
        # save plot
        fname = f'plot_{samples_tag}-pred-check-{ninds}ells{additional_tag}.png'
        plt.savefig(f'{self.outdir}/{fname}', format='png', bbox_inches='tight')
        print('## saved %s' % fname )
        plt.close()

        # lets plot of the spectra - this piece should work for >1 spectra type
        print(f'## working on the spectra plot ...')
        # plot
        plt.clf()
        nrows = len(cls_tags)
        _, axes = plt.subplots(nrows, 1,)
        plt.subplots_adjust(hspace=0.5)
        for j in range(len(cls_tags)):
            if nrows == 1:
                ax = axes
            else:
                ax = axes[j]
            # loop over the drawn samples
            for i in range(len(x_pp)):
                ax.loglog(ells, x_pp[i][self.theory.nells*j:self.theory.nells*(j+1)], '.-', color='C0', alpha=0.5)
            # plot the data vector
            ax.loglog(ells, datavector[self.theory.nells*j:self.theory.nells*(j+1)], 'r.-', lw=0.75)
            # plot the subset for a correspondence with the pairplot
            ax.loglog(ells[subset_inds_to_plot], datavector[subset_inds_to_plot], 'kP', lw=1.5)
            # set title
            ax.set_title(cls_tags[j])
        # plot details
        if nrows == 1:
            axes.set_ylabel(r'$C_\ell$')
            axes.set_xlabel(r'$\ell$')
        else:
            axes[1].set_ylabel(r'$C_\ell$')
            axes[-1].set_xlabel(r'$\ell$')
        # title
        plt.suptitle(f'{samples_tag} predictive check - {nsamples} nsamples')
        # save plot
        fname = f'plot_{samples_tag}-pred-check_datavector-vs-prediction{additional_tag}.png'
        plt.savefig(f'{self.outdir}/{fname}',
                    bbox_inches='tight', format='png')
        print('## saved %s' % fname)
        plt.close()

    # ---------------------------------------------
    def run_pred_checks(self, datavector, nsamples,
                        datavector_param_dict, seed,
                        subset_inds_to_plot
                        ):
        """

        run both prior and posterior predictive checks.

        * datavector: arr: stacked cls
        * nsamples: int: nsamples to draw from prior/posterior for PPC
        * datavector_param_dict: dict: cosmo dict used for datavector
        * seed: int: seed to be used for generating samples
        * subset_inds_to_plot: arr: indices to consider when plotting the cls
                                    in the pairport; more than 10 is likely not a good idea.

        """
        time0 = time.time()
        seed_tag = f'seed{seed}forsampling'
        print(f'\n## running predictive checks with {nsamples} samples to be drawn ..')
        # run things for the prior
        print(f'\n## running prior predictive check ..')
        # set the seed
        _ = torch.manual_seed(seed)
        # draw samples
        samples = self.prior.sample(sample_shape=(nsamples,),)
        # run helper
        self._pred_check_helper(samples=samples, samples_tag='prior',
                                datavector=datavector, datavector_param_dict=datavector_param_dict,
                                subset_inds_to_plot=subset_inds_to_plot, additional_tag=seed_tag
                                )
        print(f'## done with the prior predictive check. time taken: {(time.time() - time0) / 60: .2f} min')

        # now run things for the posterior
        print(f'\n## running posterior predictive check ..')
        _ = torch.manual_seed(seed)
        # draw samples
        samples = self.posterior.sample(sample_shape=(nsamples,),
                                        x=datavector
                                        )
        # run helper
        self._pred_check_helper(samples=samples, samples_tag='posterior',
                                datavector=datavector, datavector_param_dict=datavector_param_dict,
                                subset_inds_to_plot=subset_inds_to_plot, additional_tag=seed_tag
                                )
        print(f'## all done. time taken: {(time.time() - time0) / 60: .2f} min')