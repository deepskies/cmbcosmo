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
parser.add_option('--restart-mcmc-burn',
                  action='store_true', dest='restart_mcmc_fromburn', default=False,
                  help='use to restart mcmc from burnin (using backend).')
parser.add_option('--restart-mcmc-postburn',
                  action='store_true', dest='restart_mcmc_postburn', default=False,
                  help='use to restart mcmc post-burnin (using backend).')
parser.add_option('--reanalyze-sbi',
                  action='store_true', dest='reanalyze_sbi', default=False,
                  help='use to reanalyze sbi samples (using saved samples).')
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
restart_mcmc_postburn = options.restart_mcmc_postburn
restart_mcmc_fromburn = options.restart_mcmc_fromburn
reanalyze_sbi = options.reanalyze_sbi
debug = options.debug
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
print(f'## saving data in {datadir}')
# -----------------------------------------------
# set up the data vector and the theory object
lmin, lmax = config_data['datavector']['lmin_lmax']
theory = theory(lmin=lmin, lmax=lmax,
                cls_to_consider=config_data['datavector']['cls_to_consider'],
                verbose=False, outdir=datadir,
                detector_noise=config_data['datavector']['detector_white_noise']
                )
datavector = theory.get_prediction(param_dict=datavector_param_dict,
                                   plot_things=True, plot_tag='data')
# add a tag for the datavector
datatag = f'lmin{lmin}_lmax{lmax}_{len(config_data["datavector"]["cls_to_consider"])}spectra'
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
                + config_data['outtag'] + '_' + datatag
    if debug:
        outdir = f'debug_{outdir}'
    outdir = config_data['paths']['outdir'] + outdir
    # make sure folder exists
    os.makedirs(outdir, exist_ok=True)
    print(f'## saving mcmc stuff in {outdir}')

    # setup the covariance
    cov = theory.get_cov(param_dict=config_data['datavector']['cosmo'],
                         fsky=config_data['datavector']['fsky'],
                         plot_things=True, plot_tag='')

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
        starts[:, i] += config_data['datavector']['cosmo'][param]
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
    mcmc_setup = setup_mcmc(datavector=datavector,
                            cov=cov,
                            param_priors=param_priors,
                            theory=theory,
                            outdir=outdir,
                            param_labels_in_order=params_to_fit
                            )
    # set up the sampler, bunrin, post
    mcmc_setup.run_mcmc(nwalkers=nwalkers,
                        starts=starts, nsteps_burn=nsteps_burn, nsteps_post=nsteps_chain,
                        restart_from_burn=restart_mcmc_fromburn,
                        restart_from_postburn=restart_mcmc_postburn,
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

    from setup_sbi import setup_sbi
    # set up sbi
    sbi_setup = setup_sbi(theory=theory,
                          param_labels_in_order=params_to_fit,
                          outdir=outdir
                          )
    # construct prior
    sbi_setup.setup_prior(param_priors=param_priors)
    # construct posterior
    sbi_setup.setup_posterior(nsims=nsims, restart=reanalyze_sbi)
    # get samples
    samples['sbi'] = sbi_setup.get_samples(nsamples=nsamples,
                                           datavector=datavector,
                                           seed=sbi_dict['sampling_seed']
                                           )
    # predictive checks
    sbi_setup.run_pred_checks(datavector=datavector,
                              nsamples=sbi_dict['pc_nsamples'],
                              datavector_param_dict=datavector_param_dict,
                              seed=sbi_dict['pc_seed'],
                              subset_inds_to_plot=sbi_dict['pc_inds_for_pairplot']
                            )
    # sbc
    sbi_setup.run_sim_based_check(nsbc_runs=sbi_dict['sbc_nruns'],
                                  nsamples=sbi_dict['sbc_nsamples'],
                                  seed=sbi_dict['sbc_seed']
                                  )
    # store outdir to outdirs dictionary
    outdirs['sbi'] = outdir
    print(f'\n## time taken: {(time.time() - time0)/60: .2f} min')
    print('# ----------')

if not run_mcmc and not run_sbi:
    print('\n## not sure what were doing here since run_mcmc and run_sbi are set to False .. \n')
    quit()

print(f'\n## processing results (if applicable) .. \n')
# now plot things
from helpers_plots import plot_chainconsumer
for tech_tag in samples:
    outdir = outdirs[tech_tag]
    fname = f'plot_{tech_tag}_chainconsumer.png'
    out = plot_chainconsumer(samples=samples[tech_tag],
                             truths=truths,
                             param_labels=param_labels,
                             color_posterior=None, color_truth=None,
                             starts=starts, nwalkers=nwalkers,
                             color_starts='r',
                             showplot=False, savefig=True, fname=fname, outdir=outdir,
                             get_bestfits=True, check_convergence=not debug
                            )
    bestfit, bestfit_low, bestfit_upp = out
    # replot with prior limits
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
                    )
    print(f'\n## {tech_tag}')
    print('## bestfits vs truth')
    for i in range(npar):
        print(f'{param_labels[i]}: {bestfit[i]:.2f}^{bestfit_upp[i]:.2f}_{bestfit_low[i]:.2f} vs {truths[i]:.2f}')

    # bestfit cls - and relative residuals
    datavector = theory.get_prediction(param_dict={key: truths[i] for i,key in enumerate(params_to_fit)},
                                       plot_things=False,
                                       return_unflat=True)
    bestfitvector = theory.get_prediction(param_dict={key: bestfit[i] for i,key in enumerate(params_to_fit)},
                                          plot_things=False,
                                          return_unflat=True)
    import matplotlib.pyplot as plt
    from cmbcosmo.settings import *
    plt.clf()
    fig, axes = plt.subplots(2,1, sharex=True, height_ratios=[2,1])
    plt.subplots_adjust(hspace=0)
    for dind, dkey in enumerate(datavector):
        if dkey != 'l':
            # add datavector
            axes[0].loglog(datavector['l'], datavector[dkey], '.-', label=f'datavector: {dkey}')
            # add bestfit
            if dind == len(datavector)-1:
                label = 'bestfit'
            else:
                label = None
            axes[0].loglog(bestfitvector['l'], bestfitvector[dkey], 'k-', label=label)
            # add relative residuals
            axes[1].plot(datavector['l'], 100 * (bestfitvector[dkey] - datavector[dkey]) / datavector[dkey], '.-', label=dkey)
    # plot details
    axes[0].legend(loc='upper left')
    axes[0].set_ylabel(r'$C_\ell$')
    axes[1].set_ylabel(r'[$C_\ell^{bestfit}/C_\ell^{data}-1$] (%)')
    axes[-1].set_xlabel(r'$\ell$')
    # save plot
    fname = f'plot_{tech_tag}_cls_comparison.png'
    plt.suptitle(config_data['outtag'], y=0.99)
    plt.savefig(f'{outdir}/{fname}',
                bbox_inches='tight', format='png')
    print('\n## saved %s' % fname)
    plt.close()
print(f'\n## overall time taken: {(time.time() - start_time)/60: .2f} min')