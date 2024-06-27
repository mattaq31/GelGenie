"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import sys
import rich_click as click
import os

segmentation_folder = os.path.abspath(os.path.join(__file__, os.path.pardir))


@click.command()
@click.option('--config_name', '-c', default=None,
              help='[String] Name or path of the config file to be read in (without .toml)')
@click.option('--batch_file', '-b', default=None,
              help='[String] Name or path of the batch file to be generated (with/without .sh)')
def generate_eddie_batch_file(config_name, batch_file):
    """
    Generates a batch file for running model training on the eddie cluster.
    """

    default_batch_file = os.path.join(segmentation_folder, 'EDDIE_scripts', 'eddie_model_training_template.sh')

    with open(default_batch_file, 'r') as f:
        template_lines = f.readlines()

    for index, line in enumerate(template_lines):
        if 'YYYYYY' in line:
            if os.path.exists(config_name):
                template_lines[index] = 'gelseg_train --parameter_config %s\n' % config_name
            else:
                template_lines[index] = template_lines[index].replace('YYYYYY', config_name.replace('.toml', ''))
            break

    if '.sh' not in batch_file:
        batch_file = batch_file + '.sh'

    with open(batch_file, 'w') as f:
        for line in template_lines:
            f.write(line)


@click.command()
# config param
@click.option('--parameter_config', default=None, help='[Path] location of TOML parameters file, '
                                                       'containing configs for this experiment')
@click.option('--user_default_config', '-u', default=None, help='[String] Default user training config to use')
# model params
@click.option('--model_name', default=None, help='[String] Which model is used [milesial-UNet/UNetPlusPlus/smp-UNet]')
@click.option('--load_checkpoint', default=None,
              help='[Bool/String] Load model with specific epoch number from a .pth file in the model checkpoints folder')
@click.option('--classes', type=click.INT, default=None, help='[int] Number of classes/probabilities per pixel')
# processing params
@click.option('--base_hardware', default=None, help='[String] Where the program is run [EDDIE/PC]')
@click.option('--device', default=None, help='[String] Which processor is used [GPU/CPU]')
@click.option('--pe', type=click.INT, default=None, help='[int] How many parallel environments (cores) needed')
# training params
@click.option('--epochs', type=click.INT, default=None, help='[int] Number of epochs desired')
@click.option('--lr', type=click.FLOAT, default=None, help='[float] Learning Rate')
@click.option('--save_checkpoint', type=click.BOOL, default=None, help='[Bool] Whether checkpoints are saved')
@click.option('--checkpoint_frequency', type=click.INT, default=None, help='[int] How often checkpoints are saved')
@click.option('--model_cleanup_frequency', type=click.INT, default=None,
              help='[int] How often checkpoints are cleanup during training')
@click.option('--grad_scaler', type=click.BOOL, default=None,
              help='[Bool] Set to true to enable mixed precision gradient scaling (should improve performance)')
@click.option('--base_dir', default=None, help='[Path] Directory for output exports')
@click.option('--optimizer_type', default=None, help='[String] Type of optimizer to be used [adam/rmsprop]')
@click.option('--scheduler_type', default=None,
              help='[String (false for none)] Which scheduler is used during training')
@click.option('--loss', default=None, type=click.STRING,
              help='[String] Components of the Loss function [CrossEntropy/Dice/Both]')
@click.option('--wandb_track', default=None, type=click.BOOL, help='[Bool] Set to false to turn off wandb logging')
# data params
# essential
@click.option('--dir_train_img', type=click.STRING, default=None, help='[Path] Directory of training images')
@click.option('--dir_train_mask', default=None, help='[Path] Directory of training masks')
@click.option('--split_training_dataset', type=click.BOOL, default=None,
              help='[Bool] Turn on training dataset splitting')
@click.option('--dir_val_img', type=click.STRING, default=None, help='[Path] Directory of validation images')
@click.option('--dir_val_mask', type=click.STRING, default=None, help='[Path] Directory of validation masks')
# non-essential
@click.option('--num_workers', type=click.INT, default=None,
              help='[int] How many workers for dataloader simultaneously ,'
                   '(parallel dataloader threads, speed up data processing)')
@click.option('--batch_size', type=click.INT, default=None, help='[int] Batch size for dataloader')
@click.option('--validation', type=click.INT, default=None,
              help='[int] % of the data that is used as validation (0-100)')
@click.option('--in_channels', type=click.INT, default=None, help='[int] Number of colour channels for model input')
@click.option('--apply_augmentations', type=click.BOOL, default=None,
              help='[Bool] Whether augmentations are applied to training images')
@click.option('--padding', type=click.BOOL, default=None, help='[Bool] Whether padding is applied to training images')
def segmentation_network_trainer(parameter_config, user_default_config, **kwargs):
    import toml
    from os.path import join
    from gelgenie.segmentation.training.environment_setup import get_user_config, \
        cli_sort_and_classify, apply_defaults, environment_checks
    from gelgenie.segmentation.training.core_training import TrainingHandler

    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}  # filters out none values i.e. not specified params
    kwargs = cli_sort_and_classify(kwargs)  # sorts params into their respective category folders

    if user_default_config:  # update params with default user-specific config
        extracted_kwargs = get_user_config(user_default_config)
        extracted_kwargs.update(kwargs)
        kwargs = extracted_kwargs

    if parameter_config:  # updates all params from file
        combined_params = toml.load(parameter_config)
        combined_params.update(kwargs)  # prioritize command-line configuration over config file
    else:
        combined_params = kwargs

    combined_params = apply_defaults(combined_params)  # applies default values to params that are not specified

    environment_checks(combined_params)  # checks for any issues/problems
    # TODO: add click command that prints out defaults

    main_trainer = TrainingHandler(base_dir=combined_params['base_dir'],
                                   experiment_name=combined_params['experiment_name'],
                                   training_parameters=combined_params['training'],
                                   data_parameters=combined_params['data'],
                                   model_parameters=combined_params['model'],
                                   processing_parameters=combined_params['processing'])

    # Copies the config file to the experiment folder for safekeeping
    if combined_params['training']['load_checkpoint']:
        config_output_file = join(main_trainer.main_folder,
                                  'config_from_epoch_%s.toml' % combined_params['training']['load_checkpoint'])
    else:
        config_output_file = join(main_trainer.main_folder, 'config.toml')

    with open(config_output_file, "w") as f:
        toml.dump(combined_params, f)
        f.close()

    main_trainer.full_training()


if __name__ == '__main__':
    segmentation_network_trainer(sys.argv[1:])  # for use when debugging with pycharm
