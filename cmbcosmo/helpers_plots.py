from cmbcosmo.settings import *
from chainconsumer import ChainConsumer

__all__ = ['plot_chainconsumer', 'plot_chainvals']
# ------------------------------------------------------------------------------
def plot_chainconsumer(samples, truths, param_labels,
                       color_posterior, color_truth,
                       starts=None, color_starts='r',
                       showplot=False, savefig=False,
                       fname=None, outdir=None,
                       get_bestfits=False, check_convergence=False
                       ):
    """
    
    Function to plot posteriors.

    Required inputs
    ---------------
    * samples: arr: samples to plot
    * truths: arr: truth values
    * param_labels: arr: labels for the parameters constrained
    * color_posterior: str: color for posterior; could be None
    * color_true: str: color for the truth; could be None

    Optional inputs
    ---------------
    * starts: arr: starting points for the walkers (applicable for MCMC).
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

    """
    # ---------------------------------------------
    # basic check before we start plotting
    if savefig:
        if fname is None or outdir is None:
            raise ValueError('must specify fname and outdir when savefig=True.')
    # get nparameters
    npar = len(param_labels)
    # set up the plot
    plt.clf()
    # set up the chainconsumer object
    c = ChainConsumer()
    # add chain
    c.add_chain(samples, parameters=param_labels, color=color_posterior)
    c.configure(statistics='mean', summary=False,
                label_font_size=20, tick_font_size=16,
                usetex=False, serif=False,
                )
    # add truth
    c.configure_truth(color=color_truth)
    # plot
    fig = c.plotter.plot(truth=truths, parameters=param_labels,
                         figsize=(2*npar, 2*npar)
                         )
    # get the axes to turn off the grid
    ax_list = fig.axes
    for ax in ax_list: ax.grid(False)

    # plot starts if specified
    if starts is not None:
        for nrow in range(npar):
            for ncol in range(nrow):
                ax_list[npar * nrow + ncol].plot(starts[:, ncol], starts[:, nrow], '+', color=color_starts)
    # save fig if applicable
    if savefig:
        plt.savefig(f'{outdir}/{fname}', format='png', bbox_inches='tight')
        print('## saved %s' % fname )
    # show plot if applicable
    if showplot:
        plt.show()
    # close fig
    plt.close('all')

    if check_convergence:
        # the following seems to throw an error in debug mode so lets not run then
        # print out convergence diagnostics
        gelman_rubin_converged = c.diagnostic.gelman_rubin()
        geweke_converged = c.diagnostic.geweke()

        print(f'\ngelman_rubin_converged: {gelman_rubin_converged}\n')
        print(f'geweke_converged: {geweke_converged}\n')

    if get_bestfits:
        import numpy as np
        bestfit, bestfit_low, bestfit_upp = np.zeros(npar), np.zeros(npar), np.zeros(npar)

        # get mean bestfit from ChainConsumer
        out = c.analysis.get_summary(chains=c.get_mcmc_chains())
        for i, key in enumerate(out):
            bestfit[i] = out[key][1]
            if out[key][0] is not None:
                bestfit_low[i] = out[key][1] - out[key][0]
                bestfit_upp[i] = out[key][2] - out[key][1]
            else:
                bestfit_low[i] = np.nan
                bestfit_upp[i] = np.nan

        return bestfit, bestfit_low, bestfit_upp
    # ---------------------------------------------
# ------------------------------------------------------------------------------
def plot_chainvals(chain_unflattened, outdir, npar, nsteps,
                   starts, truths, param_labels, filetag=None):
    """
    Function to plot param values along the chains.

    * chain_unflattened: arr: unflattended array
    * outdir: str: output directory
    * npar: int: number of params
    * nsteps: int: int: number of steps
    * starts: arr: starting positions
    * truths: arr: truth values
    * param_labels: arr: parameter labels
    * filetag: str: tag to add to the output file

    """
    plt.clf()
    fig, axes = plt.subplots(npar, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i in range(npar):
        if npar == 1: ax = axes
        else: ax = axes[i]
        # plot the chain
        ax.plot(chain_unflattened[:, :, i])
        xmax = nsteps
        # add a line for the truth
        ax.plot([0-100, xmax+100], [ truths[i], truths[i] ], 'k-.', lw=2, label='truth' )
        # add a line for the starts
        ax.plot([0], [ starts[:, i] ], 'x', color='#d62728' )
        # set up the ylabel
        ax.set_ylabel(r'%s' % param_labels[i])

    if npar == 1:
        axes.legend(bbox_to_anchor=(1, 1))
        axes.set_xlabel('# of steps')
    else:
        axes[0].legend(bbox_to_anchor=(1, 1))
        axes[-1].set_xlabel('# of steps')

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
