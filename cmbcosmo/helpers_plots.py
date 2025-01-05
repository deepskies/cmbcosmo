from cmbcosmo.settings import *
from getdist import plots as gdplots
from getdist import MCSamples as gdsamples
import numpy as np

__all__ = ['plot_posteriors', 'plot_chainvals']
# ------------------------------------------------------------------------------
def plot_posteriors(samples, loglikes, truths, param_labels,
                       color_posterior, color_truth,
                       starts=None, nwalkers=None, color_starts='r',
                       showplot=False, savefig=False,
                       fname=None, outdir=None,
                       get_bestfits=False, check_convergence=False,
                       param_ranges=None, title=None
                       ):
    """
    
    Function to plot posteriors.

    Required inputs
    ---------------
    * samples: arr: samples to plot; could be flattened or not; unflattened
                    needed to check convergence.
    * loglikes: arr: loglikehood values; used for checking convergence.
    * truths: arr: truth values
    * param_labels: arr: labels for the parameters constrained
    * color_posterior: str: color for posterior; could be None
    * color_true: str: color for the truth; could be None

    Optional inputs
    ---------------
    * starts: arr: starting points for the walkers (applicable for MCMC).
    * nwalkers: int: number of walkers (needed for checking chain convergence).
                     Default: None
    * color_starts: str: color for the starts.
                         Default: 'r' (for red)
    * showplot: bool: set to True to show the plot.
                      Default: False
    * savefig: bool: set to True to save the plot. Will require passing
                     fname and outdir.
                     Default: False.
    * fname: str: filename for the plot to be saved.
                  Default: None
    * outdir: str: output directory path.
                   Default: None
    * get_bestfits: bool: set to True to return bestfits.
                          Defaut: False
    * check_convergence: boool: set to True to check convergence using
                                chainconsumer. Default: False
    * param_ranges: list: param ranges to impose on the subplots.
                          Default: None
    * title: str: title to add to the plot. Default: None

    """
    # ---------------------------------------------
    # basic check before we start plotting
    if savefig:
        if fname is None or outdir is None:
            raise ValueError('must specify fname and outdir when savefig=True.')
    # get nparameters
    npar = len(param_labels)
    # set up the getdist samples
    gdsample = gdsamples(samples=samples, names=param_labels, loglikes=loglikes)

    # set up the plot
    plt.clf()
    g = gdplots.get_subplot_plotter()
    g.settings.fontsize = 20
    g.settings.linewidth = 2
    g.settings.axes_fontsize = 16
    g.settings.axes_labelsize = 20
    g.settings.alpha_filled_add = 0.75
    g.settings.alpha_factor_contour_lines = 1
    g.settings.lw_contour = 1
    g.settings.norm_1d_density = True
    g.triangle_plot(gdsample,
                    filled=True,
                    params=param_labels,
                    contour_colors=color_posterior
                    )
    # access the figure
    fig = plt.gcf()
    # add plot details
    for nrow in range(npar):
        for ncol in range(npar):
            ax = g.subplots[nrow, ncol]
            if nrow == ncol:
                # get confidence interval under the curve
                dens = gdsample.get1DDensity(param_labels[nrow])
                lb = gdsample.confidence(nrow, 0.16)
                ub = gdsample.confidence(nrow, 0.16, upper=True)
                ax.fill_between(dens.x, dens.P, where=(dens.x > lb) & (dens.x < ub), alpha=0.25)

            # deal with the grid
            if ax is not None:
                ax.grid(False)

    # deal with the truths
    if truths is not None:
        ls = '--'
        for nrow in range(npar):
            for ncol in range(npar):
                ax = g.subplots[nrow, ncol]
                if ax is not None:
                    # deal with the truth lines
                    if nrow == ncol:
                        ax.axvline(x=truths[ncol], color=color_truth, ls=ls)
                    else:
                        ax.axvline(x=truths[ncol], color=color_truth, ls=ls)
                        ax.axhline(y=truths[nrow], color=color_truth, ls=ls)

    # deal with the starts
    if starts is not None:
        for nrow in range(npar):
            for ncol in range(npar):
                ax = g.subplots[nrow, ncol]
                if ax is not None:
                    if nrow != ncol:
                        ax.plot(starts[:, ncol], starts[:, nrow], '+', color=color_starts)

    # now deal with the lims
    if param_ranges is not None:
        for nrow in range(npar):
            for ncol in range(npar):
                ax = g.subplots[nrow, ncol]
                if ax is not None:
                    if nrow == ncol:
                        ax.set_xlim(param_ranges[ncol])
                    else:
                        ax.set_xlim(param_ranges[ncol])
                        ax.set_ylim(param_ranges[nrow])

    if title is not None:
        if npar == 1:
            y = 1.1
        else:
            y = 1.05
        plt.suptitle(title, y=y) # need a better way to determine the y value
    fig.set_size_inches((2*npar, 2*npar))
    # save fig if applicable
    if savefig:
        plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
        print('## saved %s' % fname )
    # show plot if applicable
    if showplot:
        plt.show()
    # close fig
    plt.close('all')

    if check_convergence and nwalkers is not None:
        # the following seems to throw an error in debug mode so lets not run then
        # print out convergence diagnostics
        print(f'\ngelman rubin:\n{gdsample.getGelmanRubin()}; should be << 1 for good convergence\n')

    if get_bestfits:
        bestfit, bestfit_sigma = gdsample.getMeans(), np.sqrt(gdsample.getVars())

        return bestfit, bestfit_sigma
    # ---------------------------------------------
# ------------------------------------------------------------------------------
def plot_chainvals(chain_unflattened, outdir, npar,
                   starts, truths, param_labels, filetag=None):
    """
    Function to plot param values along the chains.

    * chain_unflattened: arr: unflattended array
    * outdir: str: output directory
    * npar: int: number of params
    * starts: arr: starting positions
    * truths: arr: truth values
    * param_labels: arr: parameter labels
    * filetag: str: tag to add to the output file

    """
    plt.clf()
    fig, axes = plt.subplots(npar, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    nsteps = len(chain_unflattened[:, 0, 0])
    # lets start the nsteps at 1
    steps_arr = np.arange(1, nsteps+1)
    # also lets extend the truth line(s) just a little bit
    delta = 0.01 * nsteps       # 10% buffer
    # loop over params
    for i in range(npar):
        if npar == 1: ax = axes
        else: ax = axes[i]
        # plot the chain
        ax.plot(steps_arr, chain_unflattened[:, :, i])
        # add a line for the truth
        ax.plot([1-delta, nsteps+delta], [ truths[i], truths[i] ], 'k-.', lw=2, label='truth' )
        # add a line for the starts
        if starts is not None:
            ax.plot([1], [ starts[:, i] ], 'x', color='#d62728' )
        # set up the ylabel
        ax.set_ylabel(r'%s' % param_labels[i])

    if npar == 1:
        axes.legend(bbox_to_anchor=(1, 1))
        axes.set_xlabel('\# of steps')
    else:
        axes[0].legend(bbox_to_anchor=(1, 1))
        axes[-1].set_xlabel('\# of steps')

    fig.set_size_inches(10, 6*npar/3)

    plt.suptitle(filetag)

    if filetag is not None:
        filetag = '_' + filetag
    else:
        filetag = ''
    fname = f'plot_mcmc_chain-param-values{filetag}.png'
    plt.savefig(f'{outdir}/{fname}',
                bbox_inches='tight', format='png')
    print('# saved %s' % fname)
    plt.close()
