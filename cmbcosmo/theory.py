import deepcmbsim as simcmb
from cmbcosmo.helpers_misc import flatten_data

# get theory predictions
class theory(object):
    """
    
    Class to deal with theoretical predictions.

    """
    # ---------------------------------------------
    def __init__(self, lmax,
                 verbose=False, outdir=None,
                 detector_noise=True
                 ):
        """
        Required inputs
        ----------------
        * lmax: int: max ell

        Optional inputs
        ----------------
        * verbose: bool: set to True to enable print statements
                         from deepcmbsim. Default: False
        * outdir: str or None
        * detector_noise: bool: set to False if dont want to have
                                detector white noise added to the
                                signal. Default: True.
        
        """
        # set up the keys we want
        self.keys_of_interest = ['clTT', 'clEE', 'clBB', 'clEB']
        # load the default config in deepcmbsim and udpate some things
        self.config_obj = simcmb.config_obj()
        print(f'simcmb initial config: {self.config_obj.UserParams}\n')
        # address lmax
        self.config_obj.update_val('max_l_use', lmax)
        self.lmax = lmax
        # address verbose
        self.verbose = verbose
        self.config_obj.update_val('verbose', int(self.verbose))
        # specify which cls to get
        self.config_obj.update_val('cls_wanted', self.keys_of_interest)
        # now add detector noise
        if detector_noise:
            self.config_obj.update_val('noise_type', 'detector-white')
        # set up the outdir
        self.outdir = outdir
        # set up the datatag (to be appended to output fileames)
        self.data_tag = f'lmax{lmax}_{len(self.keys_of_interest)}spectra'

    # ---------------------------------------------
    def get_prediction(self, r, plot_things=False, plot_tag='',
                       return_unflat=False, return_ell_keys_too=False):
        """
        Required inputs
        ----------------
        * r: int: value for r

        Optional inputs
        ---------------
        * plot_things: bool: set to True to plot the spectra.
                             Default: False
        * plot_tag: str: tag to add to the saved plot fname.
                         Default: ''
        * return_unflat: bool: set to True to get the dictionary, not
                               the flattened array.
                               Default: False
        * return_ell_keys_too: bool: set to True to get ells, stacked spectra,
                               and keys, and not just the stacked spectra.
                               Default: False

        Returns
        -------
        * cls: array: stacked spectra unless return_unflat is True
                      or return_ell_keys_too is True.

        """
        self.config_obj.update_val('InitPower.r', r, verbose=self.verbose)
        data = simcmb.CAMBPowerSpectrum(self.config_obj).get_cls()
        # update data tag if not all keys of interest are a
        if self.keys_of_interest + ['l'] != list(data.keys()):
            self.data_tag = f'lmax{self.lmax}_{len(data.keys())-1}spectra'
        # see if need to plot things
        if plot_things:
            if self.outdir is None:
                raise ValueError('outdir much be set for plotting things.')
            import matplotlib.pyplot as plt
            import cmbcosmo.settings
            plt.clf()
            for key in data:
                if key != 'l':
                    plt.loglog(data['l'], data[key], '.-', label=key)
            plt.legend()
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$C_\ell$')
            if plot_tag != '':
                plot_tag = '_' + plot_tag
            fname = f'plot_cls{plot_tag}_{self.data_tag}.png'
            plt.savefig(f'{self.outdir}/{fname}',
                        bbox_inches='tight', format='png')
            print('# saved %s' % fname)
            plt.close()

        if return_unflat:
            return data
        else:
            if return_ell_keys_too:
                # return: ells, flattened data without ells, keys flattened
                return  data['l'], flatten_data(data_dict=data, ignore_keys=['l']), [f for f in data.keys() if f != 'l']
            else:
                return flatten_data(data_dict=data, ignore_keys=['l'])

    # ---------------------------------------------
    def get_cov(self, r, plot_things=False, plot_tag=''):
        """

        Required inputs
        ----------------
        * r: int: value for r

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
        import os
        import numpy as np
        # set up the filename
        fname = f'{self.outdir}/cov_{self.data_tag}.npz'
        # see if the cov is already calculated
        if os.path.exists(fname):
            # cov file already exists - read it in
            print(f'## reading cov from {fname} .. ')
            cov = np.load(fname)['cov']
        else:
            # set up the cov
            ells, data, keys = self.get_prediction(r=r, return_ell_keys_too=True)
            # now set up the (diagonal) covariance with sample variance
            # first need the ell-array for all the spectra
            larr = np.hstack([ells] * len(keys))
            # now set up: (\Delta C_ell / C_ell)^2 =  2 /  ( fsky * (2ell + 1) ). assume fsky=1 for now.
            cov = np.diag( data**2 * (2 / (2 * larr + 1)))
            # save data
            np.savez_compressed(fname, cov=cov, keys=keys, ells=ells)
            print(f'## saved cov in {fname}')
            # plot if specified
            if plot_things:
                from matplotlib.ticker import FormatStrFormatter
                import matplotlib.pyplot as plt
                import cmbcosmo.settings
                plt.clf()
                plt.imshow(cov, vmin=-1e-10, vmax=1e-10)
                plt.colorbar()
                # plot details
                ax = plt.gca()
                # minor ticks
                ticks_minor = np.arange(self.lmax/2, len(cov), self.lmax)
                ax.set_xticks(ticks_minor, minor=True)
                ax.set_yticks(ticks_minor, minor=True)
                ax.tick_params(axis='both', labelsize=18, which='minor')
                ax.tick_params(axis='both', pad=2, which='minor')
                # tick labels
                if keys is not None:
                    ax.set_xticklabels(keys, minor=True) #rotation=90)
                    ax.set_yticklabels(keys, minor=True) #rotation=90)
                # major ticks
                ticks_major = np.arange(0, len(cov)+1, self.lmax)
                print(ticks_major)
                ax.set_xticks(ticks_major, minor=False)
                ax.set_yticks(ticks_major, minor=False)
                # format tick labels
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))
                ax.tick_params(axis='both', labelsize=12,
                            which='major', labelcolor='grey', pad=2)
                # save plot
                fname = f'plot_cov{plot_tag}_{self.data_tag}.png'
                plt.savefig(f'{self.outdir}/{fname}',
                            bbox_inches='tight', format='png')
                print('# saved %s' % fname)
                plt.close()
        return cov