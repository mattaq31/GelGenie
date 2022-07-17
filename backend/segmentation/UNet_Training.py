import os
import logging
import sys
from pathlib import Path
from time import strftime
import click
import toml

import torch

from segmentation.unet import UNet
from segmentation.training.basic_training import train_net


def experiment_setup(parameters, **kwargs):
    """
    This function resolves conflicts between parameters defined in a config file and/or in the command-line options.
    :param parameters: Config filepath
    :param kwargs: All other configuration options extracted from command-line
    :return: Dictionary of all resolved parameters
    """

    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}  # filters out none values

    # The default configuration if none are specified
    kwargs_default = {'parameter_config': ("C:/2022_Summer_Intern/Automatic-Gel-Analysis/backend/segmentation/"
                                           "configs/PC_default.toml"),
                      'base_hardware': "EDDIE",
                      'core': "GPU",
                      'pe': 1,
                      'memory': 64,
                      'epochs': 10,
                      'num_workers': 1,
                      'batch_size': 4,
                      'lr': 1e-5,
                      'validation': 10,
                      'save_checkpoint': True,
                      'img_scale': 0.5,
                      'amp': False,
                      'load': False,
                      'classes': 2,
                      'bilinear': False}

    # Loading the toml config file
    if parameters is not None:
        config_path = parameters
    else:
        config_path = kwargs_default['parameters']
    params = toml.load(config_path)

    params.update(kwargs)  # prioritize command-line configuration over config file
    kwargs_default.update(params)  # replaces defaults with any user-defined parameters
    params = kwargs_default  # rename kwargs_default TODO: this is unnecessary - code should be streamlined instead

    # Checks if number of workers exceed available threads when using EDDIE, and if so fixes the issue
    if params['base_hardware'] == "EDDIE" and params['core'] == "GPU":
        if params['num_workers'] > params['pe']:
            params['num_workers'] = params['pe']
            print(f"Number of workers ({params['num_workers']}) specified exceeds selected CPU cores ({params['pe']}),",
                  "It has been lowered to match the requested core count.")

    # Alerts user if GPU is selected but is unavailable, and automatically switches to CPU
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params['core'] == "GPU":
        if params['device'] == 'cpu':
            print("GPU specified but cuda is unavailable, cpu will be used instead")

    if params['base_hardware'] == "PC":  # Paths for working on Kiros's PC
        base_dir = "C:/2022_Summer_Intern/UNet_Training_With_Images"
        params['dir_img'] = Path('C:/2022_Summer_Intern/UNet_Training_With_Images/Carvana/Input')
        params['dir_mask'] = Path('C:/2022_Summer_Intern/UNet_Training_With_Images/Carvana/Target/')

    elif params['base_hardware'] == "EDDIE":  # Paths for working on EDDIE server
        base_dir = "/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/UNet_Training_With_Images/Model"
        params['dir_img'] = Path('/exports/csce/eddie/eng/groups/DunnGroup/kiros/'
                                 '2022_summer_intern/UNet_Training_With_Images/Carvana/Input/')
        params['dir_mask'] = Path('/exports/csce/eddie/eng/groups/DunnGroup/kiros/'
                                  '2022_summer_intern/UNet_Training_With_Images/Carvana/Target/')

    elif params['base_hardware'] == "MA_mac":  # Paths for working on Matthew's mac
        base_dir = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models"
        params['dir_img'] = Path('/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/Carvana/Input/')
        params['dir_mask'] = Path('/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/Carvana/Target/')

    # TODO: what's the default base directory if none specified?

    # Make base directory for storing everything
    base_dir = os.path.join(base_dir, strftime("%Y_%m_%d_%H;%M;%S"))
    # TODO: I would add the experiment name to the folder, not just the date
    os.mkdir(base_dir)  # TODO: instead of overwriting, warn user if folder already exists and has data inside

    params['base_dir'] = base_dir
    params['dir_checkpoint'] = Path(base_dir + '/checkpoints/')
    os.mkdir(params['dir_checkpoint'])

    # Copies the config file
    config_file_name = config_path.split('/')[-1]  # TODO: just make this a standard name e.g. config.toml
    with open(base_dir + '/' + config_file_name, "w") as f:
        toml.dump(params, f)
        f.close()

    # Path for saving segmentation images
    params['segmentation_path'] = base_dir + '/segmentation_images'
    os.mkdir(params['segmentation_path'])

    return params


@click.command()
@click.option('--parameter_config', default=None, help='[Path] location of TOML parameters file, '
                                                       'containing configs for this experiment')
@click.option('--base_hardware', default=None, help='[String] Where the program is run [EDDIE/PC]')
@click.option('--core', default=None, help='[String] Which processor is used [GPU/CPU]')
@click.option('--pe', default=None, help='[int] How many parallel environments (cores) needed')
@click.option('--memory', default=None, help='[int] Required memory per core in GBytes')
@click.option('--epochs', default=None, help='[int] Number of epochs desired')
@click.option('--num_workers', default=None, help='[int] How many workers for dataloader simultaneously ,'
                                                  '(parallel dataloader threads, speed up data processing)')
@click.option('--batch_size', default=None, help='[int] Batch size for dataloader')
@click.option('--lr', default=None, help='[float] Learning Rate')
@click.option('--validation', default=None, help='[int] % of the data that is used as validation (0-100)')
@click.option('--save_checkpoint', default=None, help='[Bool] Whether checkpoints are saved')
@click.option('--img_scale', default=None, help='[Float] Downscaling factor of the images')
@click.option('--amp', default=None, help='[Bool] Use mixed precision')
@click.option('--load', default=None, help='[Bool/Path] Load model from a .pth file')
@click.option('--classes', default=None, help='[int] Number of classes/probabilities per pixel')
@click.option('--bilinear', default=None, help='[Bool] Use bilinear upsampling')
def unet_train(parameter_config, **kwargs):
    params = experiment_setup(parameter_config, **kwargs)

    # Pre-trainined weights:
    # load = "/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/UNet_Training_With_Images/Pre-trained/unet_carvana_scale0.5_epoch2.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: you've already specified the device earlier - remove this line.
    print(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=params['classes'], bilinear=params['bilinear'])  # initializing random weights

    print(f'Network:\n'
          f'\t{net.n_channels} input channels\n'
          f'\t{net.n_classes} output channels (classes)\n'
          f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    load = params['load']
    if load:
        net.load_state_dict(torch.load(load, map_location=device))
        logging.info(f'Model loaded from {load}')

    net.to(device=device)

    train_net(net=net,
              epochs=int(params['epochs']),
              batch_size=int(params['batch_size']),
              learning_rate=float(params['lr']),
              device=device,
              img_scale=float(params['img_scale']),
              val_percent=int(params['validation']) / 100,
              amp=params['amp'],
              dir_img=params['dir_img'],
              dir_mask=params['dir_mask'],
              dir_checkpoint=params['dir_checkpoint'],
              num_workers=int(params['num_workers'], ),
              segmentation_path=params['segmentation_path'],
              base_dir=params['base_dir'])


if __name__ == '__main__':
    unet_train(sys.argv[1:])  # for use when debugging with pycharm
