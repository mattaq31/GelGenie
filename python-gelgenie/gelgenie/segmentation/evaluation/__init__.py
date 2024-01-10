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

import rich_click as click
import sys


def model_eval_load(exp_folder, eval_epoch):
    import toml
    import torch
    from os.path import join
    from gelgenie.segmentation.networks import model_configure
    from gelgenie.segmentation.helper_functions.stat_functions import load_statistics

    model_config = toml.load(join(exp_folder, 'config.toml'))['model']
    model, _, _ = model_configure(**model_config)
    if eval_epoch == 'best':
        stats = load_statistics(join(exp_folder, 'training_logs'), 'training_stats.csv', config='pd')
        sel_epoch = stats['Epoch'][stats['Dice Score'].idxmax()]
    else:
        sel_epoch = eval_epoch

    checkpoint = torch.load(f=join(exp_folder, 'checkpoints', 'checkpoint_epoch_%s.pth' % sel_epoch),
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['network'])
    model.eval()

    return model



@click.command()
@click.option('--model_and_epoch', '-me', multiple=True,
              help='Experiments and epochs to evaluate.', type=(str, str))
@click.option('--model_folder', '-p', default=None,
              help='Path to folder containing model config.')
@click.option('--input_folder', '-i', default=None,
              help='Path to folder containing input images.')
@click.option('--output_folder', '-o', default=None,
              help='Path to folder containing output images.')
@click.option('--multi_augment', is_flag=True,
              help='Set this flag to run test-time augmentation on input images.')
@click.option('--run_quant_analysis', is_flag=True,
              help='Set this flag to run quantitative analysis comparing output images with target masks.')
@click.option('--mask_folder', default=None,
              help='Path to ground truth mask data corresponding to input images.')
@click.option('--classical_analysis', is_flag=True,
              help='Set this flag to also run classical analyses for comparison purposes.')
@click.option('--map_colour', default=None, type=(int, int, int),
              help='Colour to use for output segmentation map.  Default is a brown/golden colour.')
@click.option('--add_map_from_file', default=None, type=(str, str), multiple=True,
              help='Tuple with 1) the name of a pre-computed model and 2) path to its precomputed segmentation maps '
                   'for the dataset in question. These maps will be added to the output '
                   'images and quantified as normal.')
def segmentation_pipeline(model_and_epoch, model_folder, input_folder, output_folder, multi_augment,
                          run_quant_analysis, mask_folder, classical_analysis, map_colour, add_map_from_file):

    from os.path import join
    from gelgenie.segmentation.evaluation.core_functions import segment_and_plot, segment_and_quantitate
    from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty

    experiment_names, eval_epochs = zip(*model_and_epoch)

    models = []

    for experiment, eval_epoch in zip(experiment_names, eval_epochs):
        exp_folder = join(model_folder, experiment)
        model = model_eval_load(exp_folder, eval_epoch)
        models.append(model)

    create_dir_if_empty(output_folder)

    if map_colour is None:
        map_colour = (163, 106, 13)

    if run_quant_analysis:
        segment_and_quantitate(models, list(experiment_names), input_folder, mask_folder, output_folder,
                               multi_augment=multi_augment, run_classical_techniques=classical_analysis,
                               map_pixel_colour=map_colour, nnunet_models_and_folders=add_map_from_file)
    else:
        segment_and_plot(models, list(experiment_names), input_folder, output_folder, multi_augment=multi_augment,
                         run_classical_techniques=classical_analysis, map_pixel_colour=map_colour,
                         nnunet_models_and_folders=add_map_from_file)


if __name__ == '__main__':
    segmentation_pipeline(sys.argv[1:])  # for use when debugging with pycharm
