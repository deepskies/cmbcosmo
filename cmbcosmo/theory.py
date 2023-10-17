import deepcmbsim as simcmb
from cmbcosmo.helpers_misc import flatten_data

# get theory predictions
class theory(object):
    """
    
    Class to deal with theoretical predictions.
    No covariances for now.

    """
    # ---------------------------------------------
    def __init__(self, randomseed, verbose=False):
        """
        Required inputs
        ----------------
        * randomseed: int: random seed for CAMB

        Optional inputs
        ----------------
        * verbose: bool: set to True to enable print statements
                         from deepcmbsim. Default: False
        
        """
        # load the default config in deepcmbsim and udpate some things
        self.config_obj = simcmb.config_obj()
        print(f'initial config: {self.config_obj.UserParams}\n')
        self.verbose = verbose
        self.config_obj.update_val('verbose', int(self.verbose))
        self.config_obj.update_val('seed', randomseed, verbose=self.verbose)

    # ---------------------------------------------
    def get_prediction(self, r):
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
        return flatten_data(data_dict=data, ignore_keys=['l'])