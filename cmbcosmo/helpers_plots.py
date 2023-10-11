from cmbcosmo.settings import *
from chainconsumer import ChainConsumer

__all__ = ['plot_chainconsumer']
# ------------------------------------------------------------------------------
def plot_chainconsumer(samples, truths, param_labels,
                       color_posterior, color_truth,
                       starts=None, color_starts='r',
                       showplot=False,
                       savefig=False, fname=None, outdir=None):
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
    # ---------------------------------------------