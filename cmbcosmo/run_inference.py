# ------------------------------------------------------------------------------
# script to run inference. mcmc and sbi options.
# ------------------------------------------------------------------------------
import time, datetime
import numpy as np
import os
from cmbcosmo.setup_config import setup_config
from cmbcosmo.theory import theory
from cmbcosmo.helpers_misc import get_time_passed
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
parser.add_option('--restart-mcmc',
                  action='store_true', dest='restart_mcmc', default=False,
                  help='use to restart mcmc (using backend).')
# sbi options
parser.add_option('--sbi',
                  action='store_true', dest='sbi', default=False,
                  help='use to run SBI.')
parser.add_option('--reanalyze-sbi',
                  action='store_true', dest='reanalyze_sbi', default=False,
                  help='use to reanalyze sbi samples (using saved samples).')
parser.add_option('--reanalyze-sbi-checks',
                  action='store_true', dest='reanalyze_sbi_checks', default=False,
                  help='use to reanalyze sbi checks (using saved samples).')
parser.add_option('--no-sbi-checks',
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
restart_mcmc = options.restart_mcmc
reanalyze_sbi = options.reanalyze_sbi
reanalyze_sbi_checks = options.reanalyze_sbi_checks
no_sbi_checks = options.no_sbi_checks
debug = options.debug
# deal with imports
if run_mcmc:
    import emcee as emcee
    from helpers_plots import plot_chainvals
if run_sbi:
    from sbi.analysis import pairplot
    from sbi.inference import NPE, simulate_for_sbi
    from sbi.utils import BoxUniform
    from sbi.utils.user_input_checks import (
            check_sbi_inputs, process_prior, process_simulator,
            )
    from sbi.analysis.plot import sbc_rank_plot, plot_tarp
    from sbi.diagnostics import check_sbc, run_sbc, check_tarp, run_tarp
    import pickle
    import torch
# -----------------------------------------------
# set up the config
config_data = setup_config(config_path=config_path)
if debug:
    config_data['inference']['mcmc']['nwalkers'] = 5
    config_data['inference']['mcmc']['max_nsteps'] = 10
    config_data['inference']['mcmc']['check_every_nsteps'] = 2
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
    print('## exiting. rerun the script without gen-data flag to run inference.')
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
datatag = f'lmin{lmin}_lmax{lmax}_BB-only'
# -----------------------------------------------
starts, nwalkers = None, None
samples, outdirs = {}, {}
if run_mcmc:
    print(f'\n## running mcmc .. \n')
    time0 = time.time()
    # pull mcmc related config details
    mcmc_dict = config_data['inference']['mcmc']
    nwalkers = mcmc_dict['nwalkers']
    nsteps_max = mcmc_dict['max_nsteps']
    burnin_tau_factor = mcmc_dict.get('burnin_tau_factor', 3)
    check_every_nsteps = mcmc_dict['check_every_nsteps']
    convergence_ntau = mcmc_dict.get('convergence_ntau', 100)
    convergence_deltau = mcmc_dict.get('convergence_deltau', 0.01)
    # set up the outdir
    outdir = f'lk_mcmc_{nwalkers}walkers_{nsteps_max}max-steps_' \
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
    backend_fname = f'{outdir}/backend.h5'
    backend = emcee.backends.HDFBackend(backend_fname)
    # ---
    # set things up for checking covergence along the chain
    old_tau = np.inf
    # array to hold auto corr times
    stepnum, autocorr = [], []
    # --
    # figure out where to start from
    if restart_mcmc:
        print('## resuming mcmc run ... ')
        nsteps_backend = backend.iteration
        with Pool() as pool:
            # set up the sampler
            sampler = emcee.EnsembleSampler(nwalkers, npar,
                                            get_logposterior,
                                            backend=backend,
                                            pool=pool
                                            )
            # run the chain
            for sample in sampler.sample(backend.get_last_sample(),
                                         iterations=nsteps_max-nsteps_backend,
                                         progress=True):
                # check convergence every N (user specified) steps
                if sampler.iteration % check_every_nsteps:
                    continue

                # get the autocorrelation time so far
                # using tol=0 will give something even if its not perfect
                tau = sampler.get_autocorr_time(tol=0)
                autocorr.append(np.mean(tau))
                stepnum.append(sampler.iteration)

                # now look at the convergence
                # first check if chain is N (user specified or 100) times estimated tau
                converged = np.all(tau * convergence_ntau < sampler.iteration)
                # also check if the estimated tau changes by the threshold
                # (user specifif or 1%) or not
                converged &= np.all(np.abs(old_tau - tau) / tau < convergence_deltau)
                if converged:
                    break
                old_tau = tau
    else:
        print('## starting mcmc run ... ')
        backend.reset(nwalkers, npar)
        nsteps_backend = 0
        with Pool() as pool:
            # set up the sampler
            sampler = emcee.EnsembleSampler(nwalkers, npar,
                                            get_logposterior,
                                            backend=backend,
                                            pool=pool
                                            )
            # run the chain
            for sample in sampler.sample(starts, iterations=nsteps_max, progress=True):
                # check convergence every N (user specified) steps
                if sampler.iteration % check_every_nsteps:
                    continue

                # get the autocorrelation time so far
                # using tol=0 will give something even if its not perfect
                tau = sampler.get_autocorr_time(tol=0)
                autocorr.append(np.mean(tau))
                stepnum.append(sampler.iteration)

                # now look at the convergence
                # first check if chain is N (user specified or 100) times estimated tau
                converged = np.all(tau * convergence_ntau < sampler.iteration)
                # also check if the estimated tau changes by the threshold
                # (user specifif or 1%) or not
                converged &= np.all(np.abs(old_tau - tau) / tau < convergence_deltau)
                if converged:
                    break
                old_tau = tau

    nsteps = backend.iteration
    # get autocorr time
    tau = sampler.get_autocorr_time(quiet=True)
    nsteps_to_forget = np.ceil(max(tau))
    print(f'## autocorr time: {tau}\n## nsteps_to_forget: {nsteps_to_forget}')
    # we should be throwing away a few times tau steps - lets say 3x (or user-specified)
    # lets only really implement this if not in debug mode
    burn_steps = int(burnin_tau_factor * nsteps_to_forget)
    print(f'## will be discarding {burn_steps} out of {nsteps} as burn in.')
    if burn_steps > nsteps:
        # need to run longer chain
        # raise error when not in debug mode
        if debug:
            print(f'## tau is {nsteps_to_forget} so cant discard {burnin_tau_factor}x = {burn_steps}; ' +
                    'setting burn_steps to 0 here.')
            burn_steps = 0
        else:
            raise ValueError(f'## need to run longer chain - tau is {nsteps_to_forget};' +
                             f'so cant discard 3x = {burn_steps} if chain is {nsteps} steps.')
    # get samples
    samples['mcmc'] = sampler.get_chain(discard=burn_steps, flat=True)
    print(f'\n## time taken: {get_time_passed(time0=time0)}')

    # lets save autocorr and plot it, if applicable
    if len(autocorr) > 0:
        stepnum, autocorr = np.array(stepnum), np.array(autocorr)
        # plot
        plt.clf()
        plt.plot(stepnum, autocorr, '.-')
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")
        # save fig
        fname = f'plot_mcmc_autocorr_steps{nsteps_backend}-{nsteps}.png'
        plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
        print('## saved %s' % fname )
        plt.close()

        # lets also save autocorr array
        fname = f'{outdir}/autocorr_steps{nsteps_backend}-{nsteps}.npz'
        np.savez_compressed(fname, autocorr=autocorr, stepnum=stepnum)
        print(f'## saved autcorr data in {fname}')

    # save chainvals
    # full chain
    plot_chainvals(chain_unflattened=sampler.get_chain(),
                    outdir=outdir, npar=npar, nsteps=nsteps,
                    starts=starts, truths=truths, param_labels=param_labels,
                    filetag='full-chain')
    # burnin discarded
    plot_chainvals(chain_unflattened=sampler.get_chain(discard=burn_steps),
                    outdir=outdir, npar=npar, nsteps=nsteps-burn_steps,
                    starts=None, truths=truths, param_labels=param_labels,
                    filetag='burn-discarded')
    backend, sampler = [], []
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

    # extract the cov diagonal for the sample variance
    sigma_sample_variance = np.sqrt(cov.diagonal())

    # construct prior
    low = [param_priors[i][0] for i in range(npar)]
    high = [param_priors[i][1] for i in range(npar)]
    prior = BoxUniform(low=torch.FloatTensor(low),
                       high=torch.FloatTensor(high)
                       )
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

    ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f'## working with ncpus = {ncpus}')

    fname = f'sbi_posterior_nsims{nsims}.pickle'
    if reanalyze_sbi:
        if not os.path.exists(f'{outdir}/{fname}'):
            raise ValueError(f'cant restart since {fname} not found in {outdir}.')
        else:
            # read in
            print(f'## reading in saved posteriors from {outdir}/{fname}')
            posterior = pickle.load( open(f'{outdir}/{fname}', 'rb') )
    else:
        # check prior
        prior, num_parameters, prior_returns_numpy = process_prior(prior=prior)
        # check simulator
        simulator = process_simulator(user_simulator=simulator,
                                      prior=prior,
                                      is_numpy_simulator=prior_returns_numpy
                                      )
        # check prior, simulator
        check_sbi_inputs(simulator=simulator, prior=prior)
        # create inference object
        inference = NPE(prior=prior)
        # generate simulations
        theta, x = simulate_for_sbi(simulator=simulator,
                                    proposal=prior, num_simulations=nsims,
                                    seed=sbi_dict['infer_seed'],
                                    show_progress_bar=True,
                                    num_workers=ncpus
                                    )
        # pass sims to inference object
        inference = inference.append_simulations(theta=theta, x=x)
        # now train the netwrok
        density_estimator = inference.train()
        # build posterior
        posterior = inference.build_posterior(density_estimator=density_estimator)
        # now save the posterior for later
        pickle.dump(posterior, open(f'{outdir}/{fname}', 'wb' ) )
        print(f'\n## saved posterior as {outdir}/{fname}')
    print(f'## time taken done. {get_time_passed(time0=time0)}')
    print('## ---')
    # get samples
    print(f'## getting samples ..')
    _ = torch.manual_seed(sbi_dict['sampling_seed'])
    samples['sbi'] = posterior.sample(sample_shape=(nsamples,),
                                      x=datavector
                                      ).cpu().detach().numpy()

    if not no_sbi_checks:
        def helper_sbc_ppc(sample):
                return theory.get_prediction(param_dict={f: sample[i] for i, f in enumerate(params_to_fit)},
                                             add_sample_variance=True, sigma_to_use=sigma_sample_variance
                                            )
        # ---------------------------------------------
        def _pred_check_helper(samples, samples_tag, datavector, datavector_param_dict,
                               subset_inds_to_plot, reanalyze=False, additional_tag=None
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
                               upper='scatter',
                               labels=params_to_fit,
                               figsize=(npar * 2, npar * 2),
                               )
            # add lines for true params
            for ind, par in enumerate(params_to_fit):
                if npar == 1:
                    ax = axes
                else:
                    ax = axes[ind, ind]
                ax.axvline(x=datavector_param_dict[par], color='k', ls='--', lw=2)
            # title
            plt.suptitle(f'{samples_tag} predictive check - {nsamples} nsamples')
            # save fig
            fname = f'plot_{samples_tag}-pred-check_samples{additional_tag}.png'
            plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
            print('## saved %s' % fname )
            plt.close()

            # now generate data
            print(f'## starting data generation using the {samples_tag} samples ...')
            if subset_inds_to_plot is None:
                fname = f'sbi_ppc-samples_{samples_tag}-pred-check-all-ells{additional_tag}.pickle'
            else:
                fname = f'sbi_ppc-samples_{samples_tag}-pred-check-{len(subset_inds_to_plot)}ells{additional_tag}.pickle'
            if reanalyze:
                if not os.path.exists(f'{outdir}/{fname}'):
                    raise ValueError(f'cant reanalyze ppc since {fname} not found in {outdir}.')
                else:
                    # read in
                    print(f'## reading in saved ppc samples from {fname}')
                    x_pp = pickle.load( open(f'{outdir}/{fname}', 'rb') )['x_pp']
            else:
                # lets parallelize
                samples_ = samples.tolist()
                x_pp = list(
                            tqdm(Pool().imap(
                                            helper_sbc_ppc, samples_,
                                            chunksize=int(len(samples_)/ncpus)
                                            ),
                                total=len(samples_)
                                )
                            )
                # now save the data for later
                pickle.dump({'samples': samples_,
                             'x_pp': x_pp
                             }, open(f'{outdir}/{fname}', 'wb' ) )
                print(f'\n## saved ppc samples as {fname}')

            # lets plot of the spectra - this piece should work for >1 spectra type
            print(f'## working on the spectra plot ...')
            # plot
            plt.clf()
            _, ax = plt.subplots(1, 1,)
            plt.subplots_adjust(hspace=0.5)
            # loop over the drawn samples
            for i in range(len(x_pp)):
                ax.loglog(ells, x_pp[i], '.-', color='C0', alpha=0.5)
            # plot the data vector
            ax.loglog(ells, datavector, 'r.-', lw=0.75)
            # set title
            ax.set_title(cls_to_consider[0])
            # plot details
            ax.set_ylabel(r'$C_\ell$')
            ax.set_xlabel(r'$\ell$')
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
                            subset_inds_to_plot, reanalyze_checks
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
            _pred_check_helper(samples=samples, samples_tag='prior', reanalyze=reanalyze_checks,
                               datavector=datavector, datavector_param_dict=datavector_param_dict,
                               subset_inds_to_plot=subset_inds_to_plot, additional_tag=seed_tag
                               )
            print(f'## done with the prior predictive check. time taken: {get_time_passed(time0=time0)}')

            # now run things for the posterior
            print(f'\n## running posterior predictive check ..')
            _ = torch.manual_seed(seed)
            # draw samples
            samples = posterior.sample(sample_shape=(nsamples,),
                                       x=datavector
                                       )
            # run helper
            _pred_check_helper(samples=samples, samples_tag='posterior', reanalyze=reanalyze_checks,
                               datavector=datavector, datavector_param_dict=datavector_param_dict,
                               subset_inds_to_plot=subset_inds_to_plot, additional_tag=seed_tag
                               )
            # time passed
            print(f'## all done. {get_time_passed(time0=time0)}')
            print('## ---')
        # ---------------------------------------------
        def run_sim_based_check(nsbc_runs, nsamples, seed, reanalyze):
            """
            run simulation based check - and also tarp using the same
            nsamples/samples.

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
            # sbc params tag
            tag = f'{nsbc_runs}sbcruns_{nsamples}postsamples_{seed}seed'
            fname = f'sbi_sbc-samples_{tag}.pickle'

            # see if we need to read things from disk
            if reanalyze:
                if not os.path.exists(f'{outdir}/{fname}'):
                    raise ValueError(f'cant reanalyze sbc since {fname} not found in {outdir}.')
                else:
                    # read in
                    print(f'## reading in saved sbc samples from {fname}')
                    xs = pickle.load( open(f'{outdir}/{fname}', 'rb') )['xs']
            else:
                samples_ = thetas.tolist()
                xs = torch.FloatTensor(
                                    list(
                                        tqdm(Pool().imap(
                                                        helper_sbc_ppc, samples_,
                                                        chunksize=int(len(samples_)/ncpus)
                                                        ),
                                            total=len(samples_)
                                            )
                                        )
                                    )
                # now save the data for later
                pickle.dump({'samples': thetas,
                             'xs': xs
                             }, open(f'{outdir}/{fname}', 'wb' ) )
                print(f'\n## saved sbc samples as {fname}')

            # run sbc now
            fname = f'sbi_sbc-ranks+_{tag}.pickle'
            if reanalyze and os.path.exists(f'{outdir}/{fname}'):
                # read in
                print(f'## reading in saved run_sbc output from {fname}\n')
                out = pickle.load( open(f'{outdir}/{fname}', 'rb') )
                ranks, dap_samples = out['ranks'], out['dap_samples']
                out = []
            else:
                print(f'## running run_sbc ..')
                ranks, dap_samples = run_sbc(thetas=thetas, xs=xs,
                                            posterior=posterior,
                                            num_posterior_samples=nsamples,
                                            show_progress_bar=True,
                                            num_workers=ncpus
                                            )
                # now save the data for later
                pickle.dump({'ranks': ranks,
                             'dap_samples': dap_samples
                             }, open(f'{outdir}/{fname}', 'wb' ) )
                print(f'\n## saved run_sbc output as {fname}\n')

            print(f'## running check_sbc ..')
            check_stats = check_sbc(ranks=ranks, prior_samples=thetas,
                                    dap_samples=dap_samples,
                                    num_posterior_samples=nsamples
                                    )

            # ------------------------------
            # lets also include the tarp test
            fname = f'sbi_tarp-output_{tag}.pickle'
            if reanalyze and os.path.exists(f'{outdir}/{fname}'):
                # read in
                print(f'## reading in saved run_tarp output from {fname}\n')
                out = pickle.load( open(f'{outdir}/{fname}', 'rb') )
                ecp, alpha = out['ecp'], out['alpha']
                out = []
            else:
                print(f'## running run_tarp ..')
                ecp, alpha = run_tarp(thetas=thetas, xs=xs, posterior=posterior,
                                    references=None,  # will be calculated automatically.
                                    num_posterior_samples=nsamples)
                # now save the data for later
                pickle.dump({'ecp': ecp,
                             'alpha': alpha
                             }, open(f'{outdir}/{fname}', 'wb' ) )
                print(f'\n## saved run_tarp output as {fname}\n')

            print(f'## running check_tarp ..')
            atc, ks_pval = check_tarp(ecp, alpha)
            # ------------------------------
            # set up plots
            print(f'## working on plots ..')
            # title
            title = f"ks_pvals = {check_stats['ks_pvals'].numpy()} ;\n"
            title += f"c2st_ranks = {check_stats['c2st_ranks'].numpy()} ; "
            title += f"c2st_dap = {check_stats['c2st_dap'].numpy()}"

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

            # ----------------
            # now the tarp plot
            # title
            title = f"atc = {atc} ;\n"
            title += f"ks_pval = {ks_pval}"

            # tarp plot
            f, ax = plot_tarp(ecp=ecp, alpha=alpha)
            # update color .. and redo the legend
            ax.lines[0].set_color('C0')
            ax.legend()
            # figsize
            f.set_size_inches((npar*5, 5))
            # add title
            f.suptitle(title)
            # save fig
            fname = f'plot_tarp_{tag}.png'
            plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
            print('## saved %s' % fname )
            plt.close()
            # ----------------
            # time passed:
            print(f'## all done. time taken: {get_time_passed(time0=time0)}')
            print('## ---')

        # run the predictive checks
        run_pred_checks(datavector=datavector,
                        nsamples=sbi_dict['pc_nsamples'],
                        datavector_param_dict=datavector_param_dict,
                        seed=sbi_dict['pc_seed'],
                        subset_inds_to_plot=None,
                        reanalyze_checks=reanalyze_sbi_checks
                        )
        # sbc
        run_sim_based_check(nsbc_runs=sbi_dict['sbc_nruns'],
                            nsamples=sbi_dict['sbc_nsamples'],
                            seed=sbi_dict['sbc_seed'],
                            reanalyze=reanalyze_sbi_checks
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
                             get_bestfits=False, check_convergence=False,
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
                       get_bestfits=False, check_convergence=False,
                       param_ranges=param_priors,
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