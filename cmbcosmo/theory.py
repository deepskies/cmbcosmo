import deepcmbsim as simcmb
from cmbcosmo.helpers_misc import flatten_data

# get theory predictions
class theory(object):
    """
    
    Class to deal with theoretical predictions.
    No covariances for now.

    """
    # ---------------------------------------------
    def __init__(self, lmax,
                 verbose=False, outdir=None,
                 detector_noise=True
                 ):
        """
        Required inputs
        ----------------

        Optional inputs
        ----------------
        * verbose: bool: set to True to enable print statements
                         from deepcmbsim. Default: False
        * outdir: str or None
        
        """
        # set up the keys we want
        self.keys_of_interest = ['clTT', 'clEE', 'clBB', 'clEB']
        # load the default config in deepcmbsim and udpate some things
        self.config_obj = simcmb.config_obj()
        print(f'initial config: {self.config_obj.UserParams}\n')
        self.config_obj.update_val('max_l_use', lmax)
        self.lmax = lmax
        self.verbose = verbose
        self.config_obj.update_val('verbose', int(self.verbose))
        self.config_obj.update_val('cls_wanted', self.keys_of_interest)
        # now add detector noise
        if detector_noise:
            self.config_obj.update_val('noise_type', 'detector-white')
        self.outdir = outdir
        self.data_tag = f'lmax{lmax}_{len(self.keys_of_interest)}spectra'

    # ---------------------------------------------
    def get_prediction(self, r, plot_things=False, plot_tag='',
                       return_unflat=False):
        """
        Required inputs
        ----------------
        * r: int: value for r

        Returns
        -------
        * cls: array: stacked clTT, clEE, clBB, clTE, clPP, clPT, clPE

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
            return flatten_data(data_dict=data, ignore_keys=['l'])