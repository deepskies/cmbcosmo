import datetime, time
import yaml
import os
import numpy as np
import pickle
import torch
from cmbcosmo.settings import *
from cmbcosmo.helper_compare import plot_tarps, plot_sbc_plots, plot_posteriors_bestfits
from cmbcosmo.theory import theory
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config-path',
                  dest='config_path',
                  help='path to the (yml) config file.')
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

# -----------------------------------------------
# read in the config file
with open(config_path, 'r') as stream:
    try:
        config_data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# get some things from the config
sbc_plots_num_bins = config_data['general-params']['sbc_plots_nbins']
posterior_nsamples = config_data['general-params']['sbi_posterior_nsample']
posterior_sampleseed = config_data['general-params']['sbi_posterior_sampleseed']
datadir = config_data['paths']['datadir']
# set up outdir
outdir = f'{datadir}/comparisons'
if not os.path.exists(f'{outdir}'):
    raise ValueError(f'{outdir} doesnt exist. create this folder.')
print(f'## saving things in {outdir}')

# options
tarp = True
sbc = True
posteriors = True
truths, param_labels, color_posterior, color_truth = None, None, None, None

# now loop over things
# run things for the sbi cases
sbi_cases = config_data['sbi_cases']

# lets pull the various data first
for category in sbi_cases.keys():
    print('## ----------------------------------------')
    print(f'## working with category: {category}')
    print('## ----------------------------------------')
    cases = [f for f in list(sbi_cases[category].keys()) if f != 'param_to_vary']
    param_to_vary = sbi_cases[category]['param_to_vary']
    # now loop over the cases
    for case in cases:
        print('## -----------------')
        print(f'## working with {category}: case={case}')
        print('## -----------------')
        dict_  = sbi_cases[category][case]
        dict_['ells'] = f'lmin{dict_["lmin_lmax"][0]}_lmax{dict_["lmin_lmax"][1]}'
        lst = [dict_['nsims'], dict_['nsamples'], dict_['noise'], dict_['params'], dict_['ells'], dict_['embedding']]
        combs = np.array(np.meshgrid(*lst)).T.reshape(-1,len(lst))

        # initiate data holders
        plt.clf()
        colors = []
        if tarp:
            tarp_data = {}
        if sbc:
            sbc_data = {}
            nsbcsamples, nsbcruns = [], []
        if posteriors:
            posterior_samples = {}
            theory_dict = {}

        comb_ind = 0
        for comb in combs:
            nsims, nsamples, noise, params, ells, embedding = comb
            folder = f'lk_sbi_{nsims}nsims_{nsamples}nsamples_{noise}_{params}_{ells}_BB-only_{embedding}'

            if os.path.exists(f'{datadir}/{folder}'):
                # folder exists
                # proceed with loading things in
                fnames =  os.listdir(f'{datadir}/{folder}')
                variation_name = f'{param_to_vary}: {eval(param_to_vary)}'
                params_to_fit = dict_['params_to_fit']
                npar = len(params_to_fit)
                param_labels = []
                for par in params_to_fit:
                    if 'r' in par:
                        param_labels.append('$r$')
                    elif 'Alens' in par:
                        param_labels.append('$A_{lens}$')
                    else:
                        raise ValueError('## not sure what to do with par = {par}')

                colors.append(f'C{comb_ind}')
                # --------------------------
                # lets start with posteriors
                # --------------------------
                if posteriors:
                    fname = [fname for fname in fnames if fname.__contains__('sbi_posterior_')]
                    if len(fname) != 1:
                        raise ValueError(f'## not sure why we have {len(fname)} files when expected 1: {fname}')
                    posterior = pickle.load( open(f'{datadir}/{folder}/{fname[0]}', 'rb') )
                    _ = torch.manual_seed(posterior_sampleseed)
                    samples = posterior.sample(sample_shape=(posterior_nsamples,),
                                                        #x=datavector
                                                        ).cpu().detach().numpy()
                    posterior_samples[variation_name] = samples
                # --------------------------
                # lets work with the sbc data
                if sbc:
                    fname = [fname for fname in fnames if fname.__contains__('sbc-ranks')]
                    if len(fname) != 1:
                        raise ValueError(f'## not sure why we have {len(fname)} files when expected 1: {fname}')
                    out = pickle.load( open(f'{datadir}/{folder}/{fname[0]}', 'rb') )
                    # store
                    sbc_data[variation_name] = out['ranks']
                    nsbcsamples.append(int(fname[0].split('_')[-1].split('postsamples')[0]))
                    nsbcruns.append(int(fname[0].split('_')[-2].split('sbcruns')[0]))
                # --------------------------
                # lets work with tarp data
                # --------------------------
                if tarp:
                    # lets work with the tarp values
                    fname = [fname for fname in fnames if fname.__contains__('tarp-output')]
                    if len(fname) != 1:
                        raise ValueError(f'## not sure why we have {len(fname)} files when expected 1: {fname}')
                    #print(f'## reading in {fname[0]}')
                    out = pickle.load( open(f'{datadir}/{folder}/{fname[0]}', 'rb') )
                    # plot
                    tarp_data[variation_name] = out
                # --------------------------
                comb_ind += 1
            else:
                print(f'## folder not found: {folder}\n==> ignoring this combination of params:\n{comb}')

        # title
        title = ''
        for entry in dict_:
            if entry not in [param_to_vary, 'params', 'params_to_fit', 'lmin_lmax']:
                if entry in ['nsims', 'nsamples']:
                    title += f'{entry}{dict_[entry]}_'
                else:
                    title += f'{dict_[entry]}_'
        title = title[:-1]
        # plot tag
        plot_tag = f'{param_to_vary}-comparisons_{case}'
        # outdir for this case
        subdir = f'{outdir}/{param_to_vary}-comparisons'
        if param_to_vary != 'nsims':
            subdir += f'_{nsims}nsims'
        if param_to_vary != 'noise':
            subdir += f'_{noise}'
        if param_to_vary != 'lmin_lmax':
            subdir += f'_{dict_["ells"]}'
        if param_to_vary != 'embedding':
            if embedding.__contains__('optimized') or embedding.__contains__('noembedding'):
                subdir += f'_{embedding}'
            else:
                subdir += f'_fixed-embedding'
        os.makedirs(subdir, exist_ok=True)

        # now plot
        if posteriors:
            # set up the data vector and cov
            datavector_param_dict = config_data['datavector']['cosmo']
            ell_split = ells.split('_')
            theory_to_pass = theory(lmin=int(ell_split[0].split('lmin')[-1]),
                                    lmax=int(ell_split[1].split('lmax')[-1]),
                                    fsky=config_data['datavector']['fsky'],
                                    outdir=datadir+'data/',
                                    camb_params=config_data['datavector']['camb_params'],
                                    base_params=datavector_param_dict
                                    )
            # plot
            plot_posteriors_bestfits(data_dict=posterior_samples,
                                     params_to_fit=dict_['params_to_fit'],
                                     datavector_params=datavector_param_dict,
                                     theory=theory_to_pass,
                                     colors=colors, title=title,
                                     param_labels=param_labels,
                                     plot_tag=plot_tag, outdir=subdir,
                                     color_truth='k', param_ranges=None
                                     )
        if sbc:
            plot_sbc_plots(data_dict=sbc_data,
                           nsamples=nsbcsamples, nruns=nsbcruns,
                           colors=colors, title=title,
                           nbins=sbc_plots_num_bins, param_labels=param_labels,
                           plot_tag=plot_tag, outdir=subdir)
        if tarp:
            plot_tarps(data_dict=tarp_data, colors=colors, title=title,
                       plot_tag=plot_tag, outdir=subdir)
        plt.close('all')