from cmbcosmo.settings import *
from sbi.analysis.plot import sbc_rank_plot
from cmbcosmo.helpers_plots import plot_posteriors
import time
import numpy as np

__all__ = ['plot_tarps', 'plot_sbc_plots', 'plot_posteriors_bestfits']
# -----------------------------------------------------------------------------
def plot_tarps(data_dict, colors, title, plot_tag, outdir):
    """

    data_dict: 2-level nested dict: [key1][key2] where key1 is the case
                                    and key2 is 'alpha', 'ecp'.
    colors: list: list of colors. will pick in order of key1 in data_dict.
    title: str: plot title.
    plot_tag: str: tag for the output plot.
    outdir: str: output directory where to save the plot.

    """
    fig = plt.figure()
    ax = fig.subplots()
    # plot
    for i, case in enumerate(data_dict):
        ax.plot(data_dict[case]['alpha'], data_dict[case]['ecp'],
                color=colors[i], label=case)
    # plot details
    ax.plot([0,1], [0,1], color="black", linestyle="--", label="ideal")
    ax.set_xlabel(r"Credibility Level $\alpha$")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(bbox_to_anchor=(1,0.8))
    fig.set_size_inches(5, 5)
    fig.suptitle(title, fontsize=10, y=1.05)
    # save
    if plot_tag is not None and plot_tag != '':
        plot_tag = f'_{plot_tag}'
    fname = f'plot_tarp{plot_tag}.png'
    fig.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
    print(f'## saved {fname}')

# -----------------------------------------------------------------------------
def plot_sbc_plots(data_dict, nsamples, nruns, colors, title,
                   nbins, param_labels, plot_tag, outdir):
    """

    data_dict: dict for ranks: [key1] marks the case
    npar: int: number of params we're wokring with
    nsamples: int: number of samples drawn. will pick in order of key1 in data_dict.
    nruns: list of ints: number of sbc runs. will pick in order of key1 in data_dict.
    colors: list of colors. will pick in order of key1 in data_dict.
    title: str: plot title.
    nbins: list of ints: number of bins for the plots.
    param_labels: list of labels for the parameters.
    plot_tag: str: tag for the output plot.
    outdir: str: output directory where to save the plot.

    """
    npar = len(param_labels)
    fig_sbc_cdfs = plt.figure()
    fig_sbc_ranks = plt.figure()
    # loop over cases and add plot
    for i, case in enumerate(data_dict):
        if i == 0:
            ax_sbc_ranks = fig_sbc_ranks.subplots(1, npar)
            ax_sbc_cdfs = fig_sbc_cdfs.subplots(1, npar)
            if npar == 1:
                ax_sbc_ranks = np.array([ax_sbc_ranks])
            # show_uniform_region = True
        #else:
        #    show_uniform_region = False

        fig_, ax_ = fig_sbc_ranks, ax_sbc_ranks
        _ = sbc_rank_plot(ranks=data_dict[case],
                          num_posterior_samples=nsamples[i],
                          plot_type="hist",
                          num_bins=nbins,
                          ranks_labels=[case],
                          colors=[colors[i]],
                          uniform_region_alpha=0.2,
                          line_alpha=0.8,
                          parameter_labels=param_labels,
                          fig=fig_, ax=ax_
                          )
        for n in range(npar):
            if npar == 1:
                fig_, ax_ = fig_sbc_cdfs, ax_sbc_cdfs
            else:
                fig_, ax_ = fig_sbc_cdfs, ax_sbc_cdfs[n]
            _ = sbc_rank_plot(ranks=data_dict[case][:,n].reshape(nruns[i], 1),
                              num_posterior_samples=nsamples[i],
                              plot_type="cdf",
                              num_bins=nbins,
                              colors=[colors[i]],
                              uniform_region_alpha=0.2,
                              line_alpha=1,
                              parameter_labels=[case],
                              fig=fig_, ax=ax_
                              )
    # plot tag
    if plot_tag is not None and plot_tag != '':
            plot_tag = f'_{plot_tag}'
    # -----------
    # lets drop the "expected under uniformity" label
    handles, labels = ax_sbc_ranks[0].get_legend_handles_labels()
    # lets decide on the legend placement
    if len(labels) > 2:
        loc = (1.01*npar,0.5)
    else:
        loc = 'upper left'
    ax_sbc_ranks[0].legend(handles[::2], labels[::2], loc=loc)
    # subplot spacing
    fig_sbc_ranks.subplots_adjust(wspace=0, hspace=0)
    # title
    fig_sbc_ranks.suptitle(title, fontsize=10)
    # size
    fig_sbc_ranks.set_size_inches(5*npar, 5)
    # save fig
    plot_fname = f'plot_sbc-ranks{plot_tag}.png'
    fig_sbc_ranks.savefig(f'{outdir}/{plot_fname}', format='png', bbox_inches='tight')
    print(f'## saved {plot_fname}')
    # -----------
    # cdf plot details
    if npar > 1:
        ax_ = ax_sbc_cdfs[0]
        for n in range(1,npar):
            ax_sbc_cdfs[n].legend_.remove()
            ax_sbc_cdfs[n].set_ylabel('')
            ax_sbc_cdfs[n].set_yticklabels([])
    else:
        ax_ = ax_sbc_cdfs
    handles, labels = ax_.get_legend_handles_labels()
    # lets decide on the legend placement
    if len(labels) > 2:
        loc = (1.01*npar,0.5)
    else:
        loc = 'upper left'
    ax_.legend(handles[::2], labels[::2], loc=loc)
    # lets also update the labels
    for n in range(npar):
        if npar == 1:
            ax_sbc_cdfs.set_xlabel(f'{ax_sbc_cdfs.get_xlabel()} {param_labels[n]}')
        else:
            ax_sbc_cdfs[n].set_xlabel(f'{ax_sbc_cdfs[n].get_xlabel()} {param_labels[n]}')
    # subplot spacing
    fig_sbc_cdfs.subplots_adjust(wspace=0, hspace=0)
    # title
    fig_sbc_cdfs.suptitle(title, fontsize=10)
    # size
    fig_sbc_cdfs.set_size_inches(5*npar, 5)
    # save fig
    plot_fname = f'plot_sbc-cdfs{plot_tag}.png'
    fig_sbc_cdfs.savefig(f'{outdir}/{plot_fname}', format='png', bbox_inches='tight')
    print(f'## saved {plot_fname}')

# -----------------------------------------------------------------------------
def plot_posteriors_bestfits(data_dict, params_to_fit,
                             datavector_params, theory,
                             colors, title,
                             param_labels, plot_tag, outdir,
                             color_truth='k', param_ranges=None):
    """

    data_dict: dict for ranks: [key1] marks the case
    npar: int: number of params we're wokring with
    nsamples: int: number of samples drawn. will pick in order of key1 in data_dict.
    nruns: list of ints: number of sbc runs. will pick in order of key1 in data_dict.
    colors: list of colors. will pick in order of key1 in data_dict.
    title: str: plot title.
    nbins: list of ints: number of bins for the plots.
    param_labels: list of labels for the parameters.
    plot_tag: str: tag for the output plot.
    outdir: str: output directory where to save the plot.

    """
    npar = len(param_labels)
    # x, y, width, height
    if npar == 1:
        # works for nsims 2: legend_loc = (3, 0.5, 1, 0.15*len(colors))
        legend_loc = (3-0.1*len(colors), 0.5, 1, 0.1*len(colors))
    else:
        legend_loc = (1.05, 0.65, 1, 0.1*len(colors))

    print('legend_loc', legend_loc)
    # plot tag
    if plot_tag is not None and plot_tag != '':
        plot_tag = f'_{plot_tag}'

    # set up the truths
    truths = np.zeros(npar)
    for i, param in enumerate(params_to_fit):
        truths[i] = datavector_params[param]

    # first the posterior plots
    plot_fname = f'plot_posteriors{plot_tag}.png'
    bestfits, bestfit_sigmas = plot_posteriors(data_dict, loglikes=None,
                                               truths=truths, param_labels=param_labels,
                                               color_posterior=colors, color_truth=color_truth,
                                               starts=None, nwalkers=None, color_starts='r',
                                               get_bestfits=True, check_convergence=False,
                                               param_ranges=param_ranges, title=title,
                                               savefig=True, showplot=False,
                                               fname=plot_fname, outdir=outdir,
                                               legend_loc=legend_loc
                                               )
    # now the bestfit spectra
    # first lets get the datavector and cov
    # get the datavector
    datavector = theory.get_prediction(param_dict=datavector_params,
                                       add_sample_variance=False
                                       )
    cov = theory.get_cov(param_dict=datavector_params,
                         readfromdisk=True,
                         plot_things=False, plot_tag='')
    # ells
    ells = np.arange(theory.lmin, theory.lmax+1)
    ndof = len(datavector) - len(params_to_fit)

    # set up the plot
    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[2,1])
    plt.subplots_adjust(hspace=0)

    # plot datavector
    # truth label
    truth_label = ''
    for key in datavector_params:
        if 'r' in key:
            key_label = '$r$'
        elif 'Alens' in key:
            key_label = '$A_{lens}$'
        else:
             raise ValueError('## not sure what to do with par = {key}')
        truth_label += key_label + f': {datavector_params[key]:.2f}, '
    truth_label = '\{' + truth_label[:-2] + '\}'
    # add datavector
    err_data = np.sqrt(cov.diagonal())
    axes[0].errorbar(x=ells, y=datavector,
                    yerr=err_data,
                    color='k', fmt='.-', capsize=2, zorder=-1,
                    label=truth_label + f'; ndof {ndof}'
                    )
    axes[1].fill_between(ells,  100 * - err_data / datavector,
                        100 * err_data / datavector,
                        color='k', alpha=0.05, hatch='x')
    # now loop over cases and plot along the way
    for case_i, case in enumerate(data_dict):
        # set up the bestfit dicts
        bestfit_dict = {key: bestfits[case][i] for i,key in enumerate(params_to_fit)}
        bestfit_lower_dict = {key: bestfits[case][i]-bestfit_sigmas[case][i] for i,key in enumerate(params_to_fit)}
        bestfit_upper_dict = {key: bestfits[case][i]+bestfit_sigmas[case][i] for i,key in enumerate(params_to_fit)}
        # adding any missing params
        # need to make sure that everything else is the same as for the
        # datavector except the params to fit
        for key in datavector_params:
            if key not in bestfit_dict:
                bestfit_dict[key] = datavector_params[key]
                bestfit_lower_dict[key] = datavector_params[key]
                bestfit_upper_dict[key] = datavector_params[key]
        # get the predictions using the bestfit params
        bestfitvector = theory.get_prediction(param_dict=bestfit_dict,
                                              add_sample_variance=False
                                              )
        bestfit_lower_vector = theory.get_prediction(param_dict=bestfit_lower_dict,
                                                     add_sample_variance=False
                                                     )
        bestfit_upper_vector = theory.get_prediction(param_dict=bestfit_upper_dict,
                                                     add_sample_variance=False
                                                     )
        # diff
        delta = datavector - bestfitvector
        # calculate chi2
        chi2 = np.linalg.multi_dot([delta, np.linalg.pinv(cov), delta])

        # set up the bestfit label
        bestfit_label = ''
        for key in datavector_params:
            # i.e. we have a fit
            if 'r' in key:
                key_label = '$r$'
            elif 'Alens' in key:
                key_label = '$A_{lens}$'
            else:
                raise ValueError('## not sure what to do with par = {key}')

            if key in params_to_fit:
                key_ = r'$\mathbf{%s}$' % key_label.replace('$', '')
                bestfit_label += key_ + r': $%.2f\pm{%.2f}$, ' % (bestfit_dict[key],
                                                                  bestfit_upper_dict[key] - bestfit_dict[key]
                                                                  )
            else:
                # i.e. we have the truth value
                bestfit_label += key_label + f': {bestfit_dict[key]:.2f}, '
        # finalize
        bestfit_label = r'\{%s\}' % bestfit_label[:-2]
        bestfit_label += '; ' + r'$\chi^2_{data}$: ' + f'{chi2:.2f}' + f'; /dof: {chi2/ndof:.2f}'
        # now plot
        axes[0].plot(ells, bestfitvector, '.-', color=colors[case_i], label=f'{case}: {bestfit_label}')
        axes[0].fill_between(ells,  bestfit_lower_vector, bestfit_upper_vector, color=colors[case_i], alpha=0.2)
        # add relative residuals
        axes[1].plot(ells, 100 * (bestfitvector - datavector) / datavector, '.-', color=colors[case_i])
        axes[1].fill_between(ells,  100 * (bestfit_lower_vector - datavector) / datavector,
                            100 *  (bestfit_upper_vector - datavector) / datavector,
                            color=colors[case_i], alpha=0.2)
    # plot details
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc='upper left', fontsize=12)
    axes[0].set_ylabel(r'$C_{\ell,BB}$')
    axes[1].set_ylabel(r'[$C_{\ell,BB}^{bestfit}/C_{\ell,BB}^{data}-1$] (\%)', fontsize=12)
    axes[-1].set_xlabel(r'$\ell$')
    # save plot
    fname = f'plot_cls-comparison{plot_tag}.png'
    fig.suptitle(title, y=0.99)
    fig.savefig(f'{outdir}/{fname}',
                bbox_inches='tight', format='png')
    print('\n## saved %s' % fname)
    plt.close()