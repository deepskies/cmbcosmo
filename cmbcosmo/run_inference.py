# ------------------------------------------------------------------------------
# script to run inference. mcmc and sbi options.
# ------------------------------------------------------------------------------
import time, datetime
import numpy as np
import os
from cmbcosmo.setup_config import setup_config
from cmbcosmo.theory import theory
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config-path',
                  dest='config_path',
                  help='path to the (yml) config file.')
parser.add_option('--mcmc',
                  action='store_true', dest='mcmc', default=False,
                  help='use to run MCMC inference.')
parser.add_option('--sbi',
                  action='store_true', dest='sbi', default=False,
                  help='use to run SBI.')
parser.add_option('--restart-mcmc',
                  action='store_true', dest='restart_mcmc', default=False,
                  help='use to restart mcmc.')
parser.add_option('--restart-mcmc-burn',
                  action='store_true', dest='restart_mcmc_burn', default=False,
                  help='use to restart mcmc burn too.')
parser.add_option('--debug',
                  action='store_true', dest='debug', default=False,
                  help='run everything in debug mode.')
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
run_mcmc = options.mcmc
run_sbi = options.sbi
restart_mcmc = options.restart_mcmc
restart_mcmc_burn = options.restart_mcmc_burn
debug = options.debug
# -----------------------------------------------
# set up the config
config_data = setup_config(config_path=config_path)
if debug:
    config_data['inference']['mcmc']['nwalkers'] = 2
    config_data['inference']['mcmc']['nburn'] = 5
    config_data['inference']['mcmc']['nchain'] = 5
    config_data['inference']['sbi']['infer_nsims'] = 10
    config_data['inference']['sbi']['posterior_nsamples'] = 10
# now pull some things from the config
params_to_fit = config_data['inference']['params_to_fit']
param_labels = config_data['inference']['param_labels']
param_priors = config_data['inference']['param_priors']
npar = len(params_to_fit)
# check to make sure we can handle the requested params to fit
params_allowed = ['r']
for param in params_to_fit:
    if param not in params_allowed:
        raise ValueError('dont have functionality to fit {param}')
# set up truths
truths = list(config_data['datavector']['cosmo'].values())
# set up datadir
datadir = config_data['paths']['outdir'] + 'data'
# make sure folder exists
os.makedirs(datadir, exist_ok=True)
print(f'## saving data in {datadir}')
# -----------------------------------------------
# set up the data vector and the theory object
theory = theory(lmax=config_data['datavector']['lmax'],
                verbose=False, outdir=datadir,
                detector_noise=config_data['datavector']['detector_white_noise']
                )
datavector = theory.get_prediction(r=config_data['datavector']['cosmo']['r'],
                                   plot_things=True, plot_tag='data')
# -----------------------------------------------
starts, nwalkers = None, None
samples, outdirs = {}, {}
if run_mcmc:
    print(f'\n## running mcmc .. \n')
    time0 = time.time()
    from setup_mcmc import setup_mcmc 
    # pull mcmc related config details
    mcmc_dict = config_data['inference']['mcmc']
    nwalkers = mcmc_dict['nwalkers']
    nsteps_burn, nsteps_chain = mcmc_dict['nburn'], mcmc_dict['nchain']
    # set up the outdir
    outdir = f'lk_mcmc_{nwalkers}walkers_{nsteps_burn}burn_{nsteps_chain}post_' \
                + config_data['outtag']
    if debug:
        outdir = f'debug_{outdir}'
    outdir = config_data['paths']['outdir'] + outdir
    # make sure folder exists
    os.makedirs(outdir, exist_ok=True)
    print(f'## saving mcmc stuff in {outdir}')

    # setup the covariance
    cov = theory.get_cov(r=config_data['datavector']['cosmo']['r'],
                         plot_things=True, plot_tag='')

    # starting points for the chains
    # initialize - perturbation from the truth
    np.random.seed(mcmc_dict['randomseed_starts'])
    starts = list(config_data['datavector']['cosmo'].values())
    starts += 0.1 * np.random.randn(nwalkers, npar)
    # set up mcmc details - log prior, likelihood, posterior
    mcmc_setup = setup_mcmc(datavector=datavector,
                            cov=cov,
                            param_priors=param_priors,
                            theory=theory,
                            outdir=outdir
                            )
    # set up the sampler, bunrin, post
    mcmc_setup.run_mcmc(nwalkers=nwalkers,
                        starts=starts, nsteps_burn=nsteps_burn, nsteps_post=nsteps_chain,
                        restart=restart_mcmc, restart_from_burn=restart_mcmc_burn,
                        progress=True
                        )
    # get samples
    samples['mcmc'] = mcmc_setup.get_samples(flat=True)
    print(f'\n## time taken: {(time.time() - time0)/60: .2f} min')
    # save chainvals
    mcmc_setup.plot_chainvals(truths=truths, param_labels=param_labels)
    print('# ----------')
    outdirs['mcmc'] = outdir

if run_sbi:
    print(f'\n## running sbi .. \n')
    time0 = time.time()
    from setup_sbi import setup_sbi
    # pull sbi related config details
    sbi_dict = config_data['inference']['sbi']
    nsims = sbi_dict['infer_nsims']
    nsamples = sbi_dict['posterior_nsamples']
    # set up sbi
    sbi_setup = setup_sbi(theory=theory)
    # construct prior
    sbi_setup.setup_prior(param_priors=param_priors)
    # construct posterior
    sbi_setup.setup_posterior(nsims=nsims)
    # get samples
    samples['sbi'] = sbi_setup.get_samples(nsamples=nsamples,
                                           datavector=datavector
                                           )
    # set up the outdir
    outdir = f'lk_sbi_{nsims}nsims_{nsamples}nsamples_' + config_data['outtag']
    if debug:
        outdir = f'debug_{outdir}'
    outdir = config_data['paths']['outdir'] + outdir
    # make sure folder exists
    os.makedirs(outdir, exist_ok=True)
    print(f'## saving sbi stuff in {outdir}')

    outdirs['sbi'] = outdir
    print(f'\n## time taken: {(time.time() - time0)/60: .2f} min')
    print('# ----------')

if not run_mcmc and not run_sbi:
    print('\n## not sure what were doing here since run_mcmc and run_sbi are set to False .. \n')
    quit()

print(f'\n## processing results (if applicable) .. \n')
# now plot things
from helpers_plots import plot_chainconsumer
for key in samples:
    outdir = outdirs[key]
    fname = f'plot_{key}_chainconsumer.png'
    out = plot_chainconsumer(samples=samples[key],
                             truths=truths,
                             param_labels=param_labels,
                             color_posterior=None, color_truth=None,
                             starts=starts, nwalkers=nwalkers,
                             color_starts='r',
                             showplot=False, savefig=True, fname=fname, outdir=outdir,
                             get_bestfits=True, check_convergence=not debug
                            )
    bestfit, bestfit_low, bestfit_upp = out
    print(f'\n## {key}')
    print('## bestfits vs truth')
    for i in range(npar):
        print(f'{param_labels[i]}: {bestfit[i]:.2f}^{bestfit_upp[i]:.2f}_{bestfit_low[i]:.2f} vs {truths[i]:.2f}')

    # bestfit cls
    datavector = theory.get_prediction(r=truths[0], plot_things=False,
                                       return_unflat=True)
    bestfitvector = theory.get_prediction(r=bestfit[0], plot_things=False,
                                          return_unflat=True)
    import matplotlib.pyplot as plt
    import cmbcosmo.settings
    plt.clf()
    for dkey in datavector:
        if dkey != 'l':
            plt.loglog(datavector['l'], datavector[dkey], '.-', label=dkey)
            plt.loglog(bestfitvector['l'], bestfitvector[dkey], 'k-')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell$')
    fname = f'plot_{key}_cls_comparison.png'
    plt.title(key)
    plt.savefig(f'{outdir}/{fname}',
                bbox_inches='tight', format='png')
    print('\n## saved %s' % fname)
    plt.close()
print(f'\n## overall time taken: {(time.time() - start_time)/60: .2f} min')