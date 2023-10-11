# ------------------------------------------------------------------------------
# script to run inference. mcmc only as a start.
# ------------------------------------------------------------------------------
import time, datetime
import numpy as np
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
# -----------------------------------------------
# set up the data vector and the theory object
theory = theory(randomseed=config_data['datavector']['randomseed'], verbose=False)
ells, datavector = theory.get_prediction(cosmo_dict=config_data['datavector']['cosmo'])
# -----------------------------------------------
starts = None
if run_mcmc:
    import emcee
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
    runmcmc = setup_mcmc(datavector=datavector,
                         param_priors=param_priors,
                         theory=theory
                         )
    # set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, npar, runmcmc.logposterior)
    # burn-in
    print('## burning in ... ')
    pos, prob, stat = sampler.run_mcmc(starts, nsteps_burn, progress=True)
    # now reset the sampler
    sampler.reset()
    # run the full chain now
    print('## running the full chain ... ')
    sampler.run_mcmc(pos, nsteps_chain, progress=True)
    # extract samples
    samples = sampler.get_chain(flat=True)
else:
    print('\n## not sure what were doing here since run_mcmc is set to False .. \n')
    quit()

# now plot things
outdir = config_data['paths']['outdir']
# make sure the path is accessible
import os
if '~' in outdir:
    outdir = os.path.expanduser(outdir)

from helpers_plots import plot_chainconsumer
fname = 'plot_chainconsumer.png'
plot_chainconsumer(samples=samples,
                   truths=truths,
                   param_labels=param_labels,
                   color_posterior=None, color_truth=None,
                   starts=starts, color_starts='r',
                   showplot=False,
                   savefig=True, fname=fname, outdir=outdir
                   )
print(f'\n##time taken: {(time.time() - start_time)/60: .2f} min')