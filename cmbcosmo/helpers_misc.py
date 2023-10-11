import numpy as np

__all__ = ['flatten_data']

# ------------------------------------------------------------------------------
def flatten_data(data_dict, ignore_keys=None):
    """"
    
    Function to flatten data inputted in the form of a dictionary.
    Can specify if there are any keys to ignore.

    Required inputs
    ---------------
    * data: dict: dictionary containing data to be flattended.

    Optional inputs
    ---------------
    * ignore_keys: list: list of keys to be ignored in the 
                         data dictionary.
                         Default: None

    Returns
    -------
    * data array
    
    """
    # ---------------------------------------------
    data = None
    for key in data_dict:
        if key not in ignore_keys:
            if data is None:
                data = data_dict[key]
            else:
                data = np.hstack([data, data_dict[key]])
    return data