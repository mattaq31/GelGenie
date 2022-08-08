import os
import logging
import sys
from pathlib import Path
from time import strftime
import click
import toml
from torchinfo import summary

import torch
import segmentation_models_pytorch as smp

from segmentation.unet import UNet
from segmentation.training.core_training import train_net
from segmentation.helper_functions.general_functions import create_dir_if_empty


def experiment_setup(parameter_config, **kwargs):
    """
    This function resolves conflicts between parameters defined in a config file and/or in the command-line options.
    :param parameter_config: Config filepath
    :param kwargs: All other configuration options extracted from command-line
    :return: Dictionary of all resolved parameters
    """

    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}  # filters out none values

    # The default configuration options if none are specified
    kwargs_default = {'parameter_config': ("C:/2022_Summer_Intern/Automatic-Gel-Analysis/backend/segmentation/"
                                           "configs/PC_default.toml"),
                      'base_hardware': "EDDIE",
                      'core': "GPU",
                      'model_name': 'milesial-UNet',
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
                      'bilinear': False,
                      'n_channels': 1,
                      'base_dir': './',
                      'optimizer_type': 'adam',
                      'scheduler': False,
                      'loss': 'both',
                      'apply_augmentations': False,
                      'padding': False}

    # Loading the toml config file
    if parameter_config is not None:
        config_path = parameter_config
    else:
        config_path = kwargs_default['parameter_config']
    params = toml.load(config_path)

    params.update(kwargs)  # prioritize command-line configuration over config file
    kwargs_default.update(params)  # replaces defaults with any user-defined parameters
    params = kwargs_default  # TODO: streamline code

    if params['load'] == 'false':
        params['load'] = False

    if 'dir_train_mask' not in params or 'dir_train_img' not in params or \
            'dir_val_mask' not in params or 'dir_val_img' not in params:
        raise RuntimeError('Need to specify input and mask file paths')

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

    if params['padding'] is False and int(params['batch_size']) != 1:
        print(f'padding switched off but batch_size set to {params["batch_size"]}, now set to 1')
        params['batch_size'] = 1

    # Make base directory for storing everything
    base_dir = os.path.join(params['base_dir'], params['experiment_name'] + '_' + strftime("%Y_%m_%d_%H;%M;%S"))
    create_dir_if_empty(base_dir)  # TODO: instead of overwriting, warn user if folder already exists and has data inside
    # os.mkdir raises FileExistsError if directory already exists

    params['base_dir'] = base_dir
    params['dir_checkpoint'] = Path(base_dir + '/checkpoints/')
    create_dir_if_empty(params['dir_checkpoint'])

    # Copies the config file to the experiment folder
    config_file_name = 'config.toml'
    with open(base_dir + '/' + config_file_name, "w") as f:
        toml.dump(params, f)
        f.close()

    # Path for saving segmentation images
    params['segmentation_path'] = base_dir + '/segmentation_images'
    create_dir_if_empty(params['segmentation_path'])
    create_dir_if_empty(params['dir_checkpoint'])

    return params


@click.command()
@click.option('--parameter_config', default=None, help='[Path] location of TOML parameters file, '
                                                       'containing configs for this experiment')
@click.option('--base_hardware', default=None, help='[String] Where the program is run [EDDIE/PC]')
@click.option('--core', default=None, help='[String] Which processor is used [GPU/CPU]')
@click.option('--model_name', default=None, help='[String] Which model is used [milesial-UNet/smp]')
@click.option('--pe', type=click.INT, default=None, help='[int] How many parallel environments (cores) needed')
@click.option('--memory', type=click.INT, default=None, help='[int] Required memory per core in GBytes')
@click.option('--epochs', type=click.INT, default=None, help='[int] Number of epochs desired')
@click.option('--num_workers', type=click.INT, default=None,
              help='[int] How many workers for dataloader simultaneously ,'
                   '(parallel dataloader threads, speed up data processing)')
@click.option('--batch_size', type=click.INT, default=None, help='[int] Batch size for dataloader')
@click.option('--lr', type=click.FLOAT, default=None, help='[float] Learning Rate')
@click.option('--validation', type=click.INT, default=None,
              help='[int] % of the data that is used as validation (0-100)')
@click.option('--save_checkpoint', type=click.BOOL, default=None, help='[Bool] Whether checkpoints are saved')
@click.option('--img_scale', type=click.FLOAT, default=None, help='[Float] Downscaling factor of the images')
@click.option('--amp', type=click.BOOL, default=None, help='[Bool] Use mixed precision')
@click.option('--load', default=None, help='[Bool/Path] Load model from a .pth file')
@click.option('--classes', type=click.INT, default=None, help='[int] Number of classes/probabilities per pixel')
@click.option('--bilinear', type=click.BOOL, default=None, help='[Bool] Use bilinear upsampling')
@click.option('--n_channels', type=click.INT, default=None, help='[int] Input image number of colour channels')
@click.option('--base_dir', default=None, help='[Path] Directory for output exports')
@click.option('--dir_train_img', default=None, help='[Path] Directory of training images')
@click.option('--dir_train_mask', default=None, help='[Path] Directory of training masks')
@click.option('--dir_val_img', default=None, help='[Path] Directory of validation images')
@click.option('--dir_val_mask', default=None, help='[Path] Directory of validation masks')
@click.option('--optimizer_type', default=None, help='[String] Type of optimizer to be used [adam/rmsprop]')
@click.option('--scheduler', type=click.BOOL, default=None, help='[Bool] Whether a scheduler is used during training')
@click.option('--loss', default=None, help='[String] Components of the Loss function [CrossEntropy/Dice/Both]')
@click.option('--apply_augmentations', type=click.BOOL, default=None,
              help='[Bool] Whether augmentations are applied to training images')
@click.option('--padding', type=click.BOOL, default=None, help='[Bool] Whether padding is applied to training images')
def unet_train(parameter_config, **kwargs):
    params = experiment_setup(parameter_config, **kwargs)

    # Pre-trainined weights:
    # load = "/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/UNet_Training_With_Images/Pre-trained/unet_carvana_scale0.5_epoch2.pth"

    device = params['device']
    print(f'Using device {device}')

    # n_classes is the number of probabilities you want to get per pixel
    if params['model_name'] == 'milesial-UNet':
        net = UNet(n_channels=int(params['n_channels']), n_classes=params['classes'], bilinear=params['bilinear'])  # initializing random weights
    elif params['model_name'] == 'UnetPlusPlus':
        net = smp.UnetPlusPlus(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
    else:
        raise RuntimeError(f'Model {params["model_name"]} unidentified, must be milesial-UNet or UnetPlusPlus')

    # prints out model summary to output directory
    model_structure = summary(net, mode='train', depth=5, device=device, verbose=0)
    with open(os.path.join(params['base_dir'], 'model_structure.txt'), 'w', encoding='utf-8') as f:
        f.write(str(model_structure))

    if params['model_name'] == 'milesial-UNet':  # TODO: combine these into one - not all print statements are necessary.
        print(f'Model:\n'
              f'\t{params["model_name"]}'
              f'Network:\n'
              f'\t{net.n_channels} input channels\n'
              f'\t{net.n_classes} output channels (classes)\n'
              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    elif params['model_name'] == 'UnetPlusPlus':
        print(f'Model:\n'
              f'\t{params["model_name"]}'
              f'Network:\n'
              f'\tencoder: resnet18\n'
              f'\t1 input channels\n'
              f'\t2 output channels (classes)\n')

    load = params['load']
    if load:
        saved_dict = torch.load(load, map_location=device)
        net.load_state_dict(saved_dict['network'])
        logging.info(f'Model loaded from {load}')
        print(f'Model loaded from {load}')
        load = saved_dict




    net.to(device=device)

    train_net(net=net,
              device=device,
              base_hardware=params['base_hardware'],
              model_name=params['model_name'],
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              learning_rate=params['lr'],
              val_percent=params['validation'] / 100,
              save_checkpoint=params['save_checkpoint'],
              img_scale=params['img_scale'],
              amp=params['amp'],
              dir_train_img=params['dir_train_img'],
              dir_train_mask=params['dir_train_mask'],
              dir_val_img=params['dir_val_img'],
              dir_val_mask=params['dir_val_mask'],
              dir_checkpoint=params['dir_checkpoint'],
              num_workers=params['num_workers'],
              segmentation_path=params['segmentation_path'],
              base_dir=params['base_dir'],
              n_channels=params['n_channels'],
              optimizer_type=params['optimizer_type'],
              scheduler_used=params['scheduler'],
              load=load,
              loss_fn=params['loss'],
              apply_augmentations=params['apply_augmentations'],
              padding=params['padding'])


if __name__ == '__main__':
    unet_train(sys.argv[1:])  # for use when debugging with pycharm
