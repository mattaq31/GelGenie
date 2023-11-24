import toml
from collections import defaultdict
import os
import copy
from rich import print as rprint

training_folder = os.path.abspath(os.path.join(__file__, os.path.pardir))
defaults_folder = os.path.join(training_folder, 'config_files', 'defaults')
global_default_config = toml.load(os.path.join(defaults_folder,
                                               'global_defaults.toml'))


def get_user_config(user):
    """
    Loads default user config for specified user.
    :param user: Currently only kiros or matthew
    :return: Dictionary of training parameters
    """
    if user == 'kiros':
        kwargs = toml.load(os.path.join(defaults_folder,
                                        'KK_default.toml'))
    elif user == 'matthew':
        kwargs = toml.load(os.path.join(defaults_folder,
                                        'MA_default.toml'))
    else:
        raise RuntimeError(f'User {user} not recognized')

    return kwargs


def cli_sort_and_classify(params):
    """
    Sorts all CLI inputs into their respective parameter sub-dictionaries.
    :param params: CLI params (no inner dictionaries)
    :return: Sorted parameters dictionary with sub-dictionaries
    """
    new_param_dict = defaultdict(dict)
    for key, val in params.items():
        for root_key, root_content in global_default_config.items():
            if isinstance(root_content, dict):
                if key in root_content:
                    new_param_dict[root_key][key] = val
            else:
                new_param_dict[key] = val
    return new_param_dict


def apply_defaults(params):
    """
    Applies default parameter values to any that are missing in the input dictionary.
    :param params: Full parameters dictionary
    :return: Full parameters dictionary with defaults applied if any were missing
    """
    new_params = copy.copy(global_default_config)

    for root_key, root_content in new_params.items():
        if isinstance(root_content, dict):
            new_params[root_key].update(params[root_key])
        else:
            if root_key in params:
                new_params[root_key] = params[root_key]

    for root_key, root_content in new_params.items():
        if isinstance(root_content, dict):
            for inner_key, inner_content in root_content.items():
                if isinstance(inner_content, str) and 'UNDEFINED' in inner_content:
                    raise RuntimeError('%s in %s needs to be defined.' % (inner_key, root_key))
        else:
            if isinstance(root_content, str) and 'UNDEFINED' in root_content:
                raise RuntimeError('%s needs to be defined.' % root_key)
    return new_params


def environment_checks(params):
    """
    Performs a series of checks to ensure that the parameters are valid and that the environment is set up correctly.
    Parameters are modified in-place.
    :param params: Input parameters dictionary
    :return: None
    """
    import torch

    if 'dir_train_mask' not in params['data'] or 'dir_train_img' not in params['data']:
        raise RuntimeError('Need to specify training input and mask file paths')

    if ('dir_val_mask' not in params['data'] or 'dir_val_img' not in params['data']) and params['data'][
        'split_training_dataset'] is False:
        raise RuntimeError('Need to specify validation input/mask folders or enable training dataset split')

    if ('dir_val_mask' in params['data'] or 'dir_val_img' in params['data']) and params['data'][
        'split_training_dataset'] is True:
        raise RuntimeError('Cannot both request a training dataset split and provide validation image/mask folders')

    # Checks if number of workers exceed available threads when using EDDIE, and if so fixes the issue
    if params['processing']['base_hardware'] == "EDDIE" and params['processing']['device'] == "GPU":
        if params['data']['num_workers'] > params['processing']['pe']:
            params['data']['num_workers'] = params['processing']['pe']
            rprint(f"[bold magenta]Number of workers ({params['data']['num_workers']}) specified exceeds "
                   f"number of CPU cores ({params['processing']['pe']}),",
                   "It has been lowered to match the requested core count.[/bold magenta]")

    # Alerts user if GPU is selected but is unavailable, and automatically switches to CPU
    if params['processing']['device'].lower() == "gpu":
        if not torch.cuda.is_available():
            params['processing']['device'] = 'cpu'
            rprint("[bold magenta]GPU specified but cuda is unavailable, CPU will be used instead[/bold magenta]")

    if params['data']['padding'] is False and int(params['data']['batch_size']) != 1:
        rprint(
            f'[bold magenta]Image padding normalisation switched off but batch_size set to {params["batch_size"]}, now set to 1[/bold magenta]')
        params['data']['batch_size'] = 1

    if (params['model']['model_name'] == 'UNetPlusPlus' or params['model']['model_name'] == 'UNet') and \
            params['data']['padding'] is False:
        rprint(f'[bold magenta]The SMP Model {params["model"]["model_name"]} is being used but padding is turned off,\n'
               f'consider turning it on if image size error occurs[/bold magenta]')
