import yaml
import os
__all__ = ['setup_config']

def setup_config(config_path):
    """
    
    Function to read in the config file. Adds some keys.

    Required inputs
    ---------------
    * config_path: str: path to the yml config file.

    Returns
    -------
    * dictionary with all the config data
    
    """
    with open(config_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # for later
    initial_keys = get_nested_dict_keys(config_data)
    # ------------------------------------------------------------------------------
    # set up the paths for the machine
    # first check to make sure the paths are given
    machine = config_data['paths']['current_machine']
    if f'{machine}_paths' not in config_data['paths']:
        raise ValueError(f'must have a key entry for {machine}_paths under paths')
    # add the paths as the paths
    for key in config_data['paths'][f'{machine}_paths']:
        config_data['paths'][key] = os.path.expanduser(config_data['paths'][f'{machine}_paths'][key])

    tag = ''
    for i, key in enumerate(config_data['inference']['params_to_fit']):
        prior = config_data['inference']['param_priors'][i]
        tag += f'{prior[0]:.2f}<={key}<{prior[1]:.2f}_'
    config_data['outtag'] = tag[0:-1]
    # print stuff out
    final_keys = get_nested_dict_keys(config_data)
    added_keys = list(set(final_keys) - set(initial_keys))
    print(f'\n# setup_config: added {len(added_keys)}:\n{added_keys}\n')
    # -----------------------------------------------------
    return config_data

# --
# helper function to get the keys in the nested dictionary
def get_nested_dict_keys(nested_dict):
    keys = []
    for key_l1 in nested_dict:
        if isinstance(nested_dict[key_l1], dict):
            for key_l2 in nested_dict[key_l1]:
                keys.append(f'{key_l1}.{key_l2}')
        else:
            keys.append(key_l1)
    return keys
