import camb
from cmbcosmo.helpers_misc import flatten_data
from cmbcosmo.settings import *
import numpy as np
from scipy.stats import truncnorm
import os
from scipy.stats import binned_statistic

# get theory predictions
class theory(object):
    """
    
    Class to deal with theoretical predictions.

    """
    # ---------------------------------------------
    def __init__(self, lmin, lmax, camb_params, base_params,
                 binned=True, nells=10,
                 fsky=1.0, outdir=None,
                 ):
        """
        Required inputs
        ----------------
        * lmin: int: min ell
        * lmax: int: max ell
        * camb_params: dict: dictionary for camb
        * base_params: dict: base params for r, alens

        Optional inputs
        ----------------
        * binned: bool: set to False to not bin the spectra.
                        Default: True.
        * fsky: float: fraction of sky to consider.
                       Default: 1.0
        * outdir: str or None
        
        """
        # store things
        self.lmin = lmin
        self.lmax = lmax
        self.binned = binned
        self.fsky = fsky
        self.camb_params = camb_params
        self.base_params = base_params
        self.outdir = outdir
        # set up the datatag (to be appended to output fileames)
        self.data_tag = f'lmin{lmin}_lmax{lmax}_1spectra'
        # set up ells based on lmin, lmax
        self.ells = np.arange(self.lmin, self.lmax+1)
        # deal with binning
        if self.binned:
            self.ells_unbinned = self.ells
            #self.edges = np.arange(self.lmin, self.lmax+delta_ell, delta_ell)
            self.edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), nells+1)
            self.ells, _, _ = binned_statistic(self.ells, self.ells,
                                               statistic='mean', bins=self.edges)
            self.data_tag += f'_binned-nells{nells}'

    # ---------------------------------------------
    def get_prediction(self, param_dict, add_sample_variance,
                       sigma_to_use=None,
                       writetodisk=False, readfromdisk=False,
                       plot_things=False, plot_tag=''):
        """
        Required inputs
        ----------------
        * param_dict: dict: param values to use for prediction.
                            options: 'r', 'Alens'. having other keys
                            won't throw an error but the values
                            won't be used.
        * add_sample_variance: bool: set to True to add sample variance
                                     to the prediction.

        Optional inputs
        ---------------
        * sigma_to_use: list: sigma to add to add sample variance to cls.
                              if None, will use cls generated from param_dict.
                              Default: None
        * plot_things: bool: set to True to plot the spectra.
                             Default: False
        * plot_tag: str: tag to add to the saved plot fname.
                         Default: ''

        Returns
        -------
        * cls: array: stacked spectra unless return_unflat is True
                      or return_ell_keys_too is True.
        """
        # figure out the fname to reading/writing to disk
        if readfromdisk or writetodisk:
            param_tag = str(param_dict)[1:][:-1].replace(':', '').replace("'", "").replace(" ", "").replace(',', '_')
            fname = f'{self.outdir}/data_{self.data_tag}_{param_tag}.npz'
        # if need to read from disk
        if readfromdisk:
            if os.path.exists(fname):
                print(f'## reading cls from {fname} .. ')
                # data file found - read it in
                cls = np.load(fname)['cls']
            else:
                # no file
                raise ValueError(f'## cls not saved to disk. rerun with writetodisk.')
        else:
            # set up camb
            pars = camb.read_ini(self.camb_params['ini'])
            pars.WantTransfer = self.camb_params.get('WantTransfer', False)
            pars.WantTensors = self.camb_params.get('WantTensors', True)
            pars.InitPower.At = self.camb_params.get('InitPower.At', 1)
            pars.set_for_lmax(self.lmax*2, lens_potential_accuracy=1)
            # loop in base params
            pars.Alens = self.base_params['Alens']
            pars.InitPower.r = self.base_params['r']

            # now loop in input params
            if 'Alens' in param_dict:
                pars.Alens = param_dict['Alens']
            if 'r' in param_dict:
                pars.InitPower.r = param_dict['r']
            # now get the results
            results = camb.get_results(pars)
            # extract BB
            cls = results.get_total_cls(self.lmax, CMB_unit='muK')[self.lmin:self.lmax+1,2]
            # bin if needed
            if self.binned:
                cls, _, _ = binned_statistic(self.ells_unbinned, cls,
                                             statistic='mean', bins=self.edges)
            # now add sample variance, if applicable
            if add_sample_variance:
                # first check if the sigma from the sample variance is available
                if sigma_to_use is None:
                    # calculate
                    sigma_sample_var = np.sqrt( cls**2 * (2 / (self.fsky * (2 * self.ells + 1))) )
                else:
                    sigma_sample_var = sigma_to_use
                # now add random pick from a normal distribution with sigma being the sigma from sample variance
                loc, scale = 0, sigma_sample_var
                cls += truncnorm.rvs(loc=loc,
                                     scale=scale,
                                     a=(-cls-loc)/scale,
                                     b=30,
                                     )
                if len(np.where( cls < 0)[0]) > 0:
                    raise ValueError('## somethings wrong: have negative cls:\n' +
                                     f'add_sample_variance = {add_sample_variance}\n' +
                                     f'sigma_to_use = {sigma_to_use}\n' +
                                     f'param_dict = {param_dict}\n'
                                    )

            if writetodisk:
                np.savez_compressed(fname, cls=cls, ells=self.ells)
                print(f'## saved cls in {fname}')
                # also save the camb params object
                fname = f'{self.outdir}/cambparams_{self.data_tag}_{param_tag}.npz'
                np.savez_compressed(fname, pars=repr(pars))
                print(f'## saved CAMBparams object in {fname}')

            if plot_things:
                plt.clf()
                plt.loglog(self.ells, cls, '.-')
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$C_{\ell,BB}$')
                if add_sample_variance:
                    plt.title('WITH sample variance added to cls')
                if plot_tag != '':
                    plot_tag = '_' + plot_tag
                fname = f'plot_cls{plot_tag}_{self.data_tag}_{param_tag}.png'
                plt.savefig(f'{self.outdir}/{fname}',
                            bbox_inches='tight', format='png')
                print('# saved %s' % fname)
                plt.close()

        return cls

    # ---------------------------------------------
    def get_cov(self, param_dict, writetodisk=True, readfromdisk=True,
                plot_things=False, plot_tag=''):
        """

        Required inputs
        ----------------
        * param_dict: dict: param values to use for prediction.
                            options: 'r', 'Alens'. having other keys
                            won't throw an error but the values
                            won't be used.

        Optional inputs
        ---------------
        * plot_things: bool: set to True to plot the spectra.
                             Default: False
        * plot_tag: str: tag to add to the saved plot fname.
                         Default: ''

        Returns
        -------
        * cov: 2D array: covariance matrix (based on sample variance)

        """
        # set up the filename
        param_tag = str(param_dict)[1:][:-1].replace(':', '').replace("'", "").replace(" ", "").replace(',', '_')
        fname = f'{self.outdir}/cov_{self.data_tag}_{param_tag}.npz'
        # see if the cov is already calculated
        if readfromdisk:
            if os.path.exists(fname):
                # cov file already exists - read it in
                print(f'## reading cov from {fname} .. ')
                cov = np.load(fname)['cov']
            else:
                raise ValueError(f'## cov not saved to disk. rerun with readfromdisk=False.')
        else:
            # set up the cov
            cls = self.get_prediction(param_dict=param_dict, add_sample_variance=False, sigma_to_use=None)
            # now set up the (diagonal) covariance with sample variance
            # now set up: (\Delta C_ell / C_ell)^2 =  2 /  ( fsky * (2ell + 1) ). assume fsky=1 for now.
            cov = np.diag( cls**2 * (2 / (self.fsky * (2 * self.ells + 1))) )
            if writetodisk:
                # save data
                np.savez_compressed(fname, cov=cov, ells=self.ells)
                print(f'## saved cov in {fname}')
            # plot if specified
            if plot_things:
                from matplotlib.ticker import FormatStrFormatter
                # set up the delta to deal with lmin
                delta_l = self.lmax - self.lmin + 1
                min_, max_ = self.lmin, self.lmin+delta_l
                # now plot
                plt.clf()
                plt.imshow(cov, vmin=-1e-10, vmax=1e-10,
                           extent=[min_, max_, max_, min_]
                           )
                plt.colorbar()
                # plot details
                ax = plt.gca()
                # deal with grid lines etc
                if self.binned:
                    ax.grid(False)
                # major ticks
                ticks_major = np.arange(min_, max_+1, delta_l)
                # set major ticks
                ax.set_xticks(ticks_major, minor=False)
                ax.set_yticks(ticks_major, minor=False)
                # formatting
                ax.tick_params(axis='both', labelsize=12,
                                which='major', labelcolor='grey', pad=2)
                # axis labels
                ax.set_xlabel(r'$C_{\ell,BB}$')
                ax.set_ylabel(r'$C_{\ell,BB}$')
                # format tick labels
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))
                # save plot
                fname = f'plot_cov{plot_tag}_{self.data_tag}_{param_tag}.png'
                plt.savefig(f'{self.outdir}/{fname}',
                            bbox_inches='tight', format='png')
                print('# saved %s' % fname)
                plt.close()
        return cov