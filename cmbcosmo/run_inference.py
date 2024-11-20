# ------------------------------------------------------------------------------
# script to run inference. mcmc and sbi options.
# ------------------------------------------------------------------------------
import time, datetime
import numpy as np
import os
from cmbcosmo.setup_config import setup_config
from cmbcosmo.theory import theory
from cmbcosmo.helpers_misc import get_time_passed
import deepcmbsim as simcmb
from cmbcosmo.settings import *
from multiprocessing import Pool
from tqdm import tqdm
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config-path',
                  dest='config_path',
                  help='path to the (yml) config file.')
parser.add_option('--gen-data',
                  action='store_true', dest='gen_data', default=False,
                  help='generate data vector.')
parser.add_option('--debug',
                  action='store_true', dest='debug', default=False,
                  help='run everything in debug mode.')
# mcmc options
parser.add_option('--mcmc',
                  action='store_true', dest='mcmc', default=False,
                  help='use to run MCMC inference.')
parser.add_option('--restart-mcmc-burn',
                  action='store_true', dest='restart_mcmc_fromburn', default=False,
                  help='use to restart mcmc from burnin (using backend).')
parser.add_option('--restart-mcmc-postburn',
                  action='store_true', dest='restart_mcmc_postburn', default=False,
                  help='use to restart mcmc post-burnin (using backend).')
# sbi options
parser.add_option('--sbi',
                  action='store_true', dest='sbi', default=False,
                  help='use to run SBI.')
parser.add_option('--reanalyze-sbi',
                  action='store_true', dest='reanalyze_sbi', default=False,
                  help='use to reanalyze sbi samples (using saved samples).')
parser.add_option('--no-checks',
                  action='store_true', dest='no_sbi_checks', default=False,
                  help='use to not run any sbi checks.')
# ------------------------------------------------------------------------------
start_time = time.time()
(options, args) = parser.parse_args()
# add date; when running on a cluster, it will be timestamp in the sbatch output
print(datetime.datetime.now())
# print inputs
print('\n## inputs: %s' % options)
# -----------------------------------------------
# read in the inputs
config_path = options.config_path
gen_data = options.gen_data
run_mcmc = options.mcmc
run_sbi = options.sbi
restart_mcmc_postburn = options.restart_mcmc_postburn
restart_mcmc_fromburn = options.restart_mcmc_fromburn
reanalyze_sbi = options.reanalyze_sbi
no_sbi_checks = options.no_sbi_checks
debug = options.debug
# deal with imports
if run_mcmc:
    import emcee as emcee
    import shutil
    from helpers_plots import plot_chainvals
if run_sbi:
    from sbi.analysis import pairplot
    import pickle
    import torch
    from sbi import utils as utils
    from sbi.inference.base import infer
    from sbi.analysis import check_sbc, run_sbc, sbc_rank_plot
# -----------------------------------------------
# set up the config
config_data = setup_config(config_path=config_path)
if debug:
    config_data['inference']['mcmc']['nwalkers'] = 5
    config_data['inference']['mcmc']['nburn'] = 5
    config_data['inference']['mcmc']['nchain'] = 5
    config_data['inference']['sbi']['infer_nsims'] = 10
    config_data['inference']['sbi']['posterior_nsamples'] = 10
    config_data['inference']['sbi']['pc_nsamples'] = 5
    config_data['inference']['sbi']['sbc_nruns'] = 10
    config_data['inference']['sbi']['sbc_nsamples'] = 5
# now pull some things from the config
params_to_fit = config_data['inference']['params_to_fit']
param_labels = config_data['inference']['param_labels']
param_priors = config_data['inference']['param_priors']
npar = len(params_to_fit)
# check to make sure we can handle the requested params to fit
params_allowed = ['r', 'Alens']
for param in params_to_fit:
    if param not in params_allowed:
        raise ValueError(f'dont have functionality to fit {param}')
# set up truths
datavector_param_dict = config_data['datavector']['cosmo']
truths = np.zeros(npar)
for i, param in enumerate(params_to_fit):
    truths[i] = datavector_param_dict[param]
# set up datadir
datadir = config_data['paths']['outdir'] + 'data'
# make sure folder exists
os.makedirs(datadir, exist_ok=True)
print(f'## datadir: {datadir}')
# -----------------------------------------------
# set up the data vector and the theory object
lmin, lmax = config_data['datavector']['lmin_lmax']
cls_to_consider = ['BB']
nells = int(lmax - lmin + 1) * len(cls_to_consider)
theory = theory(lmin=lmin, lmax=lmax,
                fsky=config_data['datavector']['fsky'],
                outdir=datadir,
                camb_params=config_data['datavector']['camb_params'],
                base_params=datavector_param_dict
                )
# set up data vector
# also covariance - used in mcmc and chi2 numbers in the final plots
if gen_data:
    datavector = theory.get_prediction(param_dict=datavector_param_dict,
                                       add_sample_variance=False,
                                       writetodisk=True,
                                       plot_things=True, plot_tag='data')
    cov = theory.get_cov(param_dict=config_data['datavector']['cosmo'],
                         readfromdisk=False,
                         plot_things=True, plot_tag='')
    print('## exiting. rerun the script without gen-data flag.')
    print(f'## overall time taken: {get_time_passed(time0=start_time)}\n\n')
    quit()
else:
    datavector = theory.get_prediction(param_dict=datavector_param_dict,
                                       add_sample_variance=False,
                                       readfromdisk=True, writetodisk=False,
                                       plot_things=False)
    cov = theory.get_cov(param_dict=config_data['datavector']['cosmo'],
                         readfromdisk=True,
                         plot_things=False, plot_tag='')
# set up ells
ells = np.arange(lmin, lmax+1)
# add a tag for the datavector
datatag = f'lmin{lmin}_lmax{lmax}_{len(cls_to_consider)}spectra'
# -----------------------------------------------
starts, nwalkers = None, None
samples, outdirs = {}, {}
if run_mcmc:
    print(f'\n## running mcmc .. \n')
    time0 = time.time()
    # pull mcmc related config details
    mcmc_dict = config_data['inference']['mcmc']
    nwalkers = mcmc_dict['nwalkers']
    nsteps_burn, nsteps_chain = mcmc_dict['nburn'], mcmc_dict['nchain']
    # set up the outdir
    outdir = f'lk_mcmc_{nwalkers}walkers_{nsteps_burn}burn_{nsteps_chain}post_' \
                + config_data['outtag'] + '_' + datatag
    if debug:
        outdir = f'debug_{outdir}'
    outdir = config_data['paths']['outdir'] + outdir
    # make sure folder exists
    os.makedirs(outdir, exist_ok=True)
    print(f'## saving mcmc stuff in {outdir}')

    # starting points for the chains
    # initialize - perturbation from the truth
    for param in params_to_fit:
        if param not in config_data['datavector']['cosmo']:
            raise ValueError(f'{param} not in param-dict for datavector => cant initialize walkers')

    starts = np.zeros(shape=(nwalkers, npar))
    for i, param in enumerate(params_to_fit):
        # set up the seed - different one for each param
        np.random.seed(mcmc_dict['randomseed_starts'] * (i+1))
        # initalize as truth
        starts[:, i] += datavector_param_dict[param]
        # now add a perturbation around the truth; 10% of prior width
        prior_width =  abs(param_priors[i][0] - param_priors[i][1])
        starts[:, i] += 0.1 * prior_width * np.random.rand(nwalkers)
        # now check to make sure things are within the priors
        # lower bound
        ind = np.where( starts[:, i] < param_priors[i][0] )[0]
        if len(ind) > 0:
            print(f'## {len(ind)} walkers starting below prior for {param}')
            # add a random perturbation
            starts[:, ind] += (0.1*prior_width/2) * abs(np.random.rand(len(ind)))
            print(f'## added another perturbation for {len(ind)} walkers to not be < prior\n')
        # upper bound
        ind = np.where( starts[:, i] > param_priors[i][1] )[0]
        if len(ind) > 0:
            print(f'## {len(ind)} walkers starting above prior for {param}')
            # add a random perturbation
            starts[:, ind] += (0.1*prior_width/2) * -abs(np.random.rand(len(ind)))
            print(f'## added another perturbation for {len(ind)} walkers to not be > prior\n')
        # check
        ind = np.where( (starts[:, i] < param_priors[i][0]) | (starts[:, i] > param_priors[i][1]) )[0]
        if len(ind) > 0:
            raise ValueError(f'## somethings wrong - {len(ind)} starting points outside prior: {starts[:, ind]}\n')

    # set up mcmc details - log prior, likelihood, posterior
    covinv = np.linalg.pinv(cov)
    # ---------------------------------------------
    def get_loglikelihood(theory_vec):
        """
        * theory_vec: arr: theory array (stacked cls) to compare against data
        """
        # diff
        delta = theory_vec - datavector
        # calculate chi2
        chi2 = np.linalg.multi_dot([delta, covinv, delta])

        return -0.5 * chi2
    # ---------------------------------------------
    def get_logprior(p):
        """
        * p: arr: parameter array to consider
        """
        good_to_go = True
        # loop over all params and check to make sure this sample is within the priors
        i, out = 0, 0
        while good_to_go is True and i < npar:
            if param_priors[i][0] <= p[i] < param_priors[i][1]:
                out += np.log( 1 / (param_priors[i][1] - param_priors[i][0]) )
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
    def get_logposterior(p):
        """
        * p: arr: parameter array to consider
        """
        # check to confirm that this sample is good with the priors
        logprior = get_logprior(p)

        if not np.isfinite(logprior):
            # i.e. value outside the prior => unlikely
            return -np.inf

        param_dict = {}
        for i, key in enumerate(params_to_fit):
            param_dict[key] = p[i]

        prediction = theory.get_prediction(param_dict=param_dict,
                                           add_sample_variance=False)
        return get_loglikelihood(theory_vec=prediction) + logprior
    # ---------------------------------------------
    # now run mcmc
    # setup sampler backend
    backend_burnin_fname = f'{outdir}/backend-burnin.h5'
    backend_fname = f'{outdir}/backend.h5'
    backend = emcee.backends.HDFBackend(backend_fname)
    # figure out where to start from
    if restart_mcmc_fromburn:
        # restart from burn
        print('## resuming burn in ... ')
        with Pool() as pool:
            # set up the sampler
            sampler = emcee.EnsembleSampler(nwalkers, npar,
                                            get_logposterior,
                                            backend=backend,
                                            pool=pool
                                            )
            # run the chain; n-steps modified based on how many were completed before
            pos, _, _ = sampler.run_mcmc(None,
                                        nsteps_burn - backend.iteration,
                                        progress=True
                                        )
            # save the backend for the burn in
            shutil.copy(backend_fname, backend_burnin_fname)
            # now reset the sampler
            sampler.reset()
            # run post-burn
            print('## running the full chain ... ')
            sampler.run_mcmc(None, nsteps_chain, progress=True)
    elif restart_mcmc_postburn:
        # start from postburn
        print('## resuming the chain postburn ... ')
        with Pool() as pool:
            # set up the sampler
            sampler = emcee.EnsembleSampler(nwalkers, npar,
                                            get_logposterior,
                                            backend=backend,
                                            pool=pool
                                            )
            # run the chain; n-steps modified based on how many were completed before
            sampler.run_mcmc(None, nsteps_chain - backend.iteration, progress=True)
    else:
        # start from scratch
        with Pool() as pool:
            # set up the sampler
            sampler = emcee.EnsembleSampler(nwalkers, npar,
                                            get_logposterior,
                                            backend=backend,
                                            pool=pool
                                            )
            # ------
            print('## burning in ... ')
            # run burn-in
            pos, _, _ = sampler.run_mcmc(starts, nsteps_burn, progress=True)
            # ------
            # save the backend for the burn in
            shutil.copy(backend_fname, backend_burnin_fname)
            # now reset the sampler
            sampler.reset()
            # run post-burn
            print('## running the full chain ... ')
            sampler.run_mcmc(pos, nsteps_chain, progress=True)

    # get samples
    samples['mcmc'] = sampler.get_chain(flat=True)
    print(f'\n## time taken: {get_time_passed(time0=time0)}')
    # save chainvals
    # first the chain
    plot_chainvals(chain_unflattened=sampler.get_chain(),
                    outdir=outdir, npar=npar, nsteps=nsteps_chain,
                    starts=starts, truths=truths, param_labels=param_labels, filetag='post-burnin')
    # now the burnin
    backend_burnin = emcee.backends.HDFBackend(backend_burnin_fname)
    sampler_burnin = emcee.EnsembleSampler(nwalkers, npar,
                                           get_logposterior,
                                           backend=backend_burnin)
    plot_chainvals(chain_unflattened=sampler_burnin.get_chain(),
                    outdir=outdir, npar=npar, nsteps=nsteps_chain,
                    starts=starts, truths=truths, param_labels=param_labels, filetag='burnin')
    backend, sampler, backend_burnin, sampler_burnin = [], [], [], []
    print('# ----------')
    outdirs['mcmc'] = outdir

if run_sbi:
    print(f'\n## running sbi .. \n')
    time0 = time.time()
    # pull sbi related config details
    sbi_dict = config_data['inference']['sbi']
    nsims = sbi_dict['infer_nsims']
    nsamples = sbi_dict['posterior_nsamples']
    # set up the outdir
    outdir = f'lk_sbi_{nsims}nsims_{nsamples}nsamples_' + config_data['outtag'] + '_' + datatag
    if debug:
        outdir = f'debug_{outdir}'
    outdir = config_data['paths']['outdir'] + outdir
    # make sure folder exists
    os.makedirs(outdir, exist_ok=True)
    print(f'## saving sbi stuff in {outdir}')

    # extra the cov diagonal for the sample variance
    sigma_sample_variance = np.sqrt(cov.diagonal())

    # construct prior
    low = [param_priors[i][0] for i in range(npar)]
    high = [param_priors[i][1] for i in range(npar)]
    prior = utils.BoxUniform(low=torch.FloatTensor(low),
                             high=torch.FloatTensor(high)
                             )
    # construct posterior
    # ---------------------------------------------
    # first set up simulator
    def simulator(params):
        """
        * params: arr: arr of params to reproduce the "sim" for
        """
        param_dict = {}
        for i, key in enumerate(params_to_fit):
            param_dict[key] = params[i]

        return theory.get_prediction(param_dict=param_dict,
                                     add_sample_variance=True,
                                     sigma_to_use=sigma_sample_variance)
    # ---------------------------------------------
    # now set up posterior
    print('## ---')
    print(f'## setting up posterior ..')
    time0 = time.time()
    fname = f'sbi_posterior_nsims{nsims}.pickle'
    if reanalyze_sbi:
        if not os.path.exists(f'{outdir}/{fname}'):
            raise ValueError(f'cant restart since {fname} not found in {outdir}.')
        else:
            # read in
            print(f'## reading in saved posteriors from {outdir}/{fname}')
            posterior = pickle.load( open(f'{outdir}/{fname}', 'rb') )
    else:
        posterior = infer(simulator=simulator,
                          prior=prior,
                          method='SNPE',
                          num_simulations=nsims,
                          )
        # now save the posterior for later
        pickle.dump(posterior, open(f'{outdir}/{fname}', 'wb' ) )
        print(f'## saved posterior as {outdir}/{fname}')
    print(f'## time taken done. {get_time_passed(time0=time0)}')
    print('## ---')
    # get samples
    print(f'## getting samples ..')
    _ = torch.manual_seed(sbi_dict['sampling_seed'])
    samples['sbi'] = posterior.sample(sample_shape=(nsamples,),
                                      x=datavector
                                      ).cpu().detach().numpy()

    if not no_sbi_checks:
        def helper_ppc(sample):
                return theory.get_prediction(param_dict={f: sample[i] for i, f in enumerate(params_to_fit)},
                                             add_sample_variance=True, sigma_to_use=sigma_sample_variance
                                            )
        # ---------------------------------------------
        def _pred_check_helper(samples, samples_tag, datavector, datavector_param_dict,
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
            if len(cls_to_consider) > 1:
                err = '## dont have the functionality to run this for more than 1spec:'
                err += f' got: {len(cls_to_consider)}'
                raise ValueError(err)

            if additional_tag is None: additional_tag = ''
            else: additional_tag = f'_{additional_tag}'
            nsamples = len(samples)
            # pairplot to check what samples were drawn for PPC
            _, axes = pairplot(samples=samples,
                               offdiag=["kde"],
                               diag=["kde"],
                               labels=params_to_fit,
                               figsize=(npar * 2, npar * 2),
                               )
            # add lines for true params
            for ind, par in enumerate(params_to_fit):
                axes[ind, ind].axvline(x=datavector_param_dict[par],
                                    color='k', ls='--', lw=2)
            # title
            plt.suptitle(f'{samples_tag} predictive check - {nsamples} nsamples')
            # save fig
            fname = f'plot_{samples_tag}-pred-check_samples{additional_tag}.png'
            plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
            print('## saved %s' % fname )
            plt.close()

            # now generate data
            print(f'## starting data generation using the {samples_tag} samples ...')
            # lets parallelize
            samples_ = samples.tolist()
            x_pp = list(tqdm(Pool().imap(helper_ppc, samples_),
                             total=len(samples_)
                            )
                        )

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
            # lets set up the limits to ensure we see everything
            # first min, max from the sampples
            min_, max_ = np.min(np.log(x_pp_subset)), np.max(np.log(x_pp_subset))
            # now loop in datavector
            min_ = min([min_, np.min(np.log(datavector))])
            max_ = min([max_, np.max(np.log(datavector))])
            # now implement
            for nrow in range(len(axes)):
                for ncol in range(len(axes)):
                    # diagonal is a count histogram => update xlims
                    if nrow == ncol:
                        axes[nrow, ncol].set_xlim([min_, max_])
                    # upper diagonal subplots need both lims updated
                    if nrow < ncol:
                        axes[nrow, ncol].set_ylim([min_, max_])
                        axes[nrow, ncol].set_xlim([min_, max_])
            # title
            plt.suptitle(f'{samples_tag} predictive check - {nsamples} nsamples')
            # save plot
            fname = f'plot_{samples_tag}-pred-check-{ninds}ells{additional_tag}.png'
            plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
            print('## saved %s' % fname )
            plt.close()

            # lets plot of the spectra - this piece should work for >1 spectra type
            print(f'## working on the spectra plot ...')
            # plot
            plt.clf()
            nrows = len(cls_to_consider)
            _, axes = plt.subplots(nrows, 1,)
            plt.subplots_adjust(hspace=0.5)
            for j in range(len(cls_to_consider)):
                if nrows == 1:
                    ax = axes
                else:
                    ax = axes[j]
                # loop over the drawn samples
                for i in range(len(x_pp)):
                    ax.loglog(ells, x_pp[i][nells*j:nells*(j+1)], '.-', color='C0', alpha=0.5)
                # plot the data vector
                ax.loglog(ells, datavector[nells*j:nells*(j+1)], 'r.-', lw=0.75)
                # plot the subset for a correspondence with the pairplot
                ax.loglog(ells[subset_inds_to_plot], datavector[subset_inds_to_plot], 'kP', lw=1.5)
                # set title
                ax.set_title(cls_to_consider[j])
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
            plt.savefig(f'{outdir}/{fname}',
                        bbox_inches='tight', format='png')
            print('## saved %s' % fname)
            plt.close()

        # ---------------------------------------------
        def run_pred_checks(datavector, nsamples,
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
            print('## ---')
            print(f'## running predictive checks with {nsamples} samples to be drawn ..')
            time0 = time.time()
            seed_tag = f'seed{seed}forsampling'
            # run things for the prior
            print(f'\n## running prior predictive check ..')
            # set the seed
            _ = torch.manual_seed(seed)
            # draw samples
            samples = prior.sample(sample_shape=(nsamples,),)
            # run helper
            _pred_check_helper(samples=samples, samples_tag='prior',
                                    datavector=datavector, datavector_param_dict=datavector_param_dict,
                                    subset_inds_to_plot=subset_inds_to_plot, additional_tag=seed_tag
                                    )
            print(f'## done with the prior predictive check. time taken: {(time.time() - time0) / 60: .2f} min')

            # now run things for the posterior
            print(f'\n## running posterior predictive check ..')
            _ = torch.manual_seed(seed)
            # draw samples
            samples = posterior.sample(sample_shape=(nsamples,),
                                            x=datavector
                                            )
            # run helper
            _pred_check_helper(samples=samples, samples_tag='posterior',
                               datavector=datavector, datavector_param_dict=datavector_param_dict,
                               subset_inds_to_plot=subset_inds_to_plot, additional_tag=seed_tag
                               )
            # time passed
            print(f'## all done. {get_time_passed(time0=time0)}')
            print('## ---')
        # ---------------------------------------------
        # helper for parallelizing sbc sample set up
        def helper_sbc(theta):
            return theory.get_prediction(param_dict={params_to_fit[i]: val for i,val in enumerate(np.array(theta))},
                                         add_sample_variance=True,
                                         sigma_to_use=sigma_sample_variance
                                        )
        # ---------------------------------------------
        def run_sim_based_check(nsbc_runs, nsamples, seed):
            """
            run simulation based check

            * nsbc_runs: int: number of runs for SBC
                            from documentation: should be ~100s or ideally 1000
            * nsamples: int: nsamples to draw from the posterior
            * seed: int: seed to be used for generating samples
            """
            print('## ---')
            print(f'## running simulation based check ..')
            time0 = time.time()

            # generate ground truth parameters and corresponding simulated observations
            # set seed
            _ = torch.manual_seed(seed)
            # sample from prior params for SBC
            thetas = prior.sample((nsbc_runs,))
            # now simulate "obervations"
            print(f'## simulating observations ..')
            xs = torch.FloatTensor(list(tqdm(Pool().map(helper_sbc, thetas.numpy()),
                                             total=len( thetas.numpy())
                                            )
                                        )
                                    )
            # run sbc now
            print(f'## running run_sbc ..')
            ranks, dap_samples = run_sbc(thetas, xs, posterior, num_posterior_samples=nsamples)
            print(f'## running check_sbc ..')
            check_stats = check_sbc(ranks, thetas, dap_samples, num_posterior_samples=nsamples)

            # ------------------------------
            # set up plots
            print(f'## working on plots ..')
            # title
            title = f"ks_pvals = {check_stats['ks_pvals'].numpy()} ;\n"
            title += f"c2st_ranks = {check_stats['c2st_ranks'].numpy()} ; "
            title += f"c2st_dap = {check_stats['c2st_dap'].numpy()}"
            # sbc params tag
            tag = f'{nsbc_runs}sbcruns_{nsamples}postsamples_{seed}seed'

            # figure out nbins
            if nsbc_runs/20 < 1:
                num_bins = 10
            else:
                num_bins = None
            print(f'num_bins = {num_bins}')
            # rank plot
            f, _ = sbc_rank_plot(ranks=ranks,
                                num_posterior_samples=nsamples,
                                plot_type="hist",
                                num_bins=num_bins,
                                figsize=((npar*5, 5))
                                )
            # add title
            f.suptitle(title)
            # save fig
            fname = f'plot_sbc_rank-plot_{tag}.png'
            plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
            print('## saved %s' % fname )
            plt.close()

            # cdf plot
            f, _ = sbc_rank_plot(ranks, nsamples, plot_type="cdf",
                                num_bins=num_bins, figsize=((8, 5))
                                )
            # add title
            f.suptitle(title)
            # save fig
            fname = f'plot_sbc_cdf_{tag}.png'
            plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
            print('## saved %s' % fname )
            plt.close()

            # time passed:
            print(f'## all done. time taken: {get_time_passed(time0=time0)}')
            print('## ---')

        # run the predictive checks
        run_pred_checks(datavector=datavector,
                        nsamples=sbi_dict['pc_nsamples'],
                        datavector_param_dict=datavector_param_dict,
                        seed=sbi_dict['pc_seed'],
                        subset_inds_to_plot=sbi_dict['pc_inds_for_pairplot']
                        )
        # sbc
        run_sim_based_check(nsbc_runs=sbi_dict['sbc_nruns'],
                            nsamples=sbi_dict['sbc_nsamples'],
                            seed=sbi_dict['sbc_seed']
                            )
    # store outdir to outdirs dictionary
    outdirs['sbi'] = outdir
    print(f'\n## time taken: {get_time_passed(time0=time0)}')
    print('# ----------')

if not run_mcmc and not run_sbi:
    print('\n## not sure what were doing here since run_mcmc and run_sbi are set to False .. \n')
    quit()

print(f'\n## processing results (if applicable) .. \n')
# now plot things
from helpers_plots import plot_chainconsumer
for tech_tag in samples:
    outdir = outdirs[tech_tag]
    # --
    # not saving this plot just yet
    fname = f'plot_{tech_tag}_chainconsumer.png'
    out = plot_chainconsumer(samples=samples[tech_tag],
                             truths=truths,
                             param_labels=param_labels,
                             color_posterior=None, color_truth=None,
                             starts=starts, nwalkers=nwalkers,
                             color_starts='r',
                             showplot=False, savefig=False, fname=fname, outdir=outdir,
                             get_bestfits=True, check_convergence=not debug
                            )
    bestfit, bestfit_low, bestfit_upp = out
    # --
    # set up the chi2
    # first need to get the cls (stacked)
    datavector = theory.get_prediction(param_dict=datavector_param_dict,
                                       add_sample_variance=False
                                       )
    bestfit_dict = {key: bestfit[i] for i,key in enumerate(params_to_fit)}
    bestfit_lower_dict = {key: bestfit[i]-bestfit_low[i] for i,key in enumerate(params_to_fit)}
    bestfit_upper_dict = {key: bestfit[i]+bestfit_upp[i] for i,key in enumerate(params_to_fit)}
    # adding any missing params
    # need to make sure that everything else is the same as for the
    # datavector except the params to fit
    for key in datavector_param_dict:
        if key not in bestfit_dict:
            bestfit_dict[key] = datavector_param_dict[key]
            bestfit_lower_dict[key] = datavector_param_dict[key]
            bestfit_upper_dict[key] = datavector_param_dict[key]
    bestfitvector = theory.get_prediction(param_dict=bestfit_dict,
                                          add_sample_variance=False
                                          )
    # diff
    delta = datavector - bestfitvector
    # calculate chi2
    chi2 = np.linalg.multi_dot([delta, np.linalg.pinv(cov), delta])
    ndof = len(datavector) - len(params_to_fit)
    # set up title
    title = r'$\chi^2_{data}$ = ' + f'{chi2:.2f}' + f'; / ndof ({ndof}) = {chi2/ndof:.2f}'
    # --
    # now plot above, with the title - and save
    out = plot_chainconsumer(samples=samples[tech_tag],
                             truths=truths,
                             param_labels=param_labels,
                             color_posterior=None, color_truth=None,
                             starts=starts, nwalkers=nwalkers,
                             color_starts='r',
                             showplot=False, savefig=True, fname=fname, outdir=outdir,
                             get_bestfits=False, check_convergence=not debug,
                             title=title
                            )
    # now replot with prior limits
    fname = f'plot_{tech_tag}_chainconsumer_prior-limited-ranges.png'
    plot_chainconsumer(samples=samples[tech_tag],
                       truths=truths,
                       param_labels=param_labels,
                       color_posterior=None, color_truth=None,
                       starts=starts, nwalkers=nwalkers,
                       color_starts='r',
                       showplot=False, savefig=True, fname=fname, outdir=outdir,
                       get_bestfits=False, check_convergence=not debug,
                       param_ranges=param_priors,
                       title=title
                    )
    # replot with truth-centric limits
    fname = f'plot_{tech_tag}_chainconsumer_truth-limited-ranges.png'
    param_ranges = list(np.zeros_like(param_priors))
    for i in range(npar):
        delta = abs( param_priors[i][0] - param_priors[i][1] )
        param_ranges[i][0] = truths[i] - delta/2
        param_ranges[i][1] = truths[i] + delta/2
        # make sure we aren't going past the priors
        # lower bound
        if param_ranges[i][0] < param_priors[i][0]:
            param_ranges[i][0] = param_priors[i][0]
        # upper bound
        if param_ranges[i][1] > param_priors[i][1]:
            param_ranges[i][1] = param_priors[i][1]
    # plot
    plot_chainconsumer(samples=samples[tech_tag],
                       truths=truths,
                       param_labels=param_labels,
                       color_posterior=None, color_truth=None,
                       starts=starts, nwalkers=nwalkers,
                       color_starts='r',
                       showplot=False, savefig=True, fname=fname, outdir=outdir,
                       get_bestfits=False, check_convergence=not debug,
                       param_ranges=param_ranges,
                       title=title
                    )
    print(f'\n## {tech_tag}')
    print('## bestfits vs truth')
    for i in range(npar):
        print(f'{param_labels[i]}: {bestfit[i]:.2f}^{bestfit_upp[i]:.2f}_{bestfit_low[i]:.2f} vs {truths[i]:.2f}')

    # bestfit cls - and relative residuals
    datavector = theory.get_prediction(param_dict=datavector_param_dict,
                                       add_sample_variance=False
                                       )
    bestfitvector = theory.get_prediction(param_dict=bestfit_dict,
                                          add_sample_variance=False
                                          )
    bestfit_lower_vector = theory.get_prediction(param_dict=bestfit_lower_dict,
                                                 add_sample_variance=False
                                                 )
    bestfit_upper_vector = theory.get_prediction(param_dict=bestfit_upper_dict,
                                                 add_sample_variance=False
                                                 )
    # set up the labels
    # truth label
    truth_label = ''
    for key in datavector_param_dict:
        truth_label += key + f': {datavector_param_dict[key]:.2f}, '
    truth_label = '\{' + truth_label[:-2] + '\}'
    # now the bestfit label
    bestfit_label = ''
    for key in datavector_param_dict:
        if key in params_to_fit:
            # i.e. we have a fit
            key_ = r'$\textbf{%s}$' % key
            bestfit_label += key_ + r': $%.2f^{+%.2f}_{-%.2f}$, ' % (bestfit_dict[key],
                                                                     bestfit_upper_dict[key] - bestfit_dict[key],
                                                                     bestfit_dict[key] - bestfit_lower_dict[key],
                                                                    )
        else:
            # i.e. we have the truth value
            bestfit_label += key + f': {bestfit_dict[key]:.2f}, '
    # finalize
    bestfit_label = r'\{%s\}' % bestfit_label[:-2]

    plt.clf()
    fig, axes = plt.subplots(2,1, sharex=True, height_ratios=[2,1])
    plt.subplots_adjust(hspace=0)
    # add datavector
    axes[0].errorbar(x=ells, y=datavector,
                     yerr=np.sqrt(cov.diagonal()),
                     fmt='.-', capsize=2, zorder=-1,
                     label=f'datavector: from {truth_label}; error bars from sample covariance object'
                    )
    label = r'bestfit: %s; %s' % (bestfit_label, title)
    axes[0].plot(ells, bestfitvector, 'k.-', label=label)
    axes[0].fill_between(ells,  bestfit_lower_vector, bestfit_upper_vector, color='k', alpha=0.1)
    # add relative residuals
    axes[1].plot(ells, 100 * (bestfitvector - datavector) / datavector, 'k.-')
    axes[1].fill_between(ells,  100 * (bestfit_lower_vector - datavector) / datavector,
                        100 *  (bestfit_upper_vector - datavector) / datavector,
                        color='k', alpha=0.1)
    # plot details
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc='upper left')
    axes[0].set_ylabel(r'$C_{\ell,BB}$')
    axes[1].set_ylabel(r'[$C_{\ell,BB}^{bestfit}/C_{\ell,BB}^{data}-1$] (\%)', fontsize=12)
    axes[-1].set_xlabel(r'$\ell$')
    # save plot
    fname = f'plot_{tech_tag}_cls_comparison.png'
    plt.suptitle(r'$%s$' % config_data['outtag'].replace('_', '; ').replace('<=', '\leq '), y=0.99)
    plt.savefig(f'{outdir}/{fname}',
                bbox_inches='tight', format='png')
    print('\n## saved %s' % fname)
    plt.close()

    # save config data in the outdir - for later reference
    with open(f'{outdir}/config_data.txt', 'w') as f:
        print(datetime.datetime.now(), file=f)
        print(f'\n## inputs: {options}', file=f)
        print(f'\n## config_data: {config_data}', file=f)

print(f'\n## overall time taken: {get_time_passed(time0=start_time)}')