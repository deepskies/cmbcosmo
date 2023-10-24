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
# -----------------------------------------------
# set up the config
config_data = setup_config(config_path=config_path)
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
# set up outdir
datadir = config_data['paths']['outdir'] + 'data'
outdir = config_data['paths']['outdir'] + config_data['outtag']
# make sure folders exist
os.makedirs(datadir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
print(f'## saving things in {outdir}')
# -----------------------------------------------
# set up the data vector and the theory object
theory = theory(randomseed=config_data['datavector']['randomseed'],
                lmax=config_data['datavector']['lmax'],
                verbose=False, outdir=datadir)
datavector = theory.get_prediction(r=config_data['datavector']['cosmo']['r'],
                                   plot_things=True, plot_tag='data')
# -----------------------------------------------
starts = None
samples = {}
if run_mcmc:
    print(f'\n## running mcmc .. \n')
    time0 = time.time()
    from setup_mcmc import setup_mcmc 
    # pull mcmc related config details
    mcmc_dict = config_data['inference']['mcmc']
    nwalkers = config_data['inference']['mcmc']['nwalkers']
    nsteps_burn, nsteps_chain = mcmc_dict['nburn'], mcmc_dict['nchain']
    # starting points for the chains
    # initialize - perturbation from the truth
    np.random.seed(mcmc_dict['randomseed_starts'])
    starts = list(config_data['datavector']['cosmo'].values())
    starts += 0.1 * np.random.randn(nwalkers, npar)
    # set up mcmc details - log prior, likelihood, posterior
    mcmc_setup = setup_mcmc(datavector=datavector,
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
    print('# ----------')
    # save chainvals
    mcmc_setup.plot_chainvals(truths=truths, param_labels=param_labels)

if run_sbi:
    print(f'\n## running sbi .. \n')
    time0 = time.time()
    from setup_sbi import setup_sbi
    # pull sbi related config details
    sbi_dict = config_data['inference']['sbi']
    # set up sbi
    sbi_setup = setup_sbi(theory=theory)
    # construct prior
    sbi_setup.setup_prior(param_priors=param_priors)
    # construct posterior
    sbi_setup.setup_posterior(nsims=sbi_dict['infer_nsims'], method=sbi_dict['infer_method'])
    # get samples
    samples['sbi'] = sbi_setup.get_samples(nsamples=sbi_dict['posterior_nsamples'],
                                           datavector=datavector
                                           )
    print(f'\n## time taken: {(time.time() - time0)/60: .2f} min')
    print('# ----------')

if not run_mcmc and not run_sbi:
    print('\n## not sure what were doing here since run_mcmc and run_sbi are set to False .. \n')
    quit()

# now plot things
from helpers_plots import plot_chainconsumer
for key in samples:
    fname = f'plot_chainconsumer_{key}.png'
    plot_chainconsumer(samples=samples[key],
                       truths=truths,
                       param_labels=param_labels,
                       color_posterior=None, color_truth=None,
                       starts=starts, color_starts='r',
                       showplot=False,
                       savefig=True, fname=fname, outdir=outdir
                       )
print(f'\n## overall time taken: {(time.time() - start_time)/60: .2f} min')