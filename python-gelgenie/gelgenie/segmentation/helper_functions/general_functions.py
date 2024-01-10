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

import glob
import os
import re
from rich.table import Table
import rich_click as click
import subprocess
import shutil
from gelgenie.segmentation.helper_functions.stat_functions import load_statistics


@click.command()
@click.option('--loc', default='/exports/csce/eddie/eng/groups/DunnGroup/matthew/models_gelgenie',
              help='server source directory', show_default=True)
@click.option('--name', '-n', help='experiment folder')
@click.option('--server', default='eddie', help='server name', show_default=True)
@click.option('--out', default='/Users/matt/Desktop/', help='output directory', show_default=True)
@click.option('--verbose', is_flag=True)
@click.option('--pull_last_epoch_results', '-pl', is_flag=True)
@click.option('--pull_best_epoch_results', '-pb', is_flag=True)
def pull_server_data(loc, name, server, out, verbose, pull_last_epoch_results, pull_best_epoch_results):
    data_contents = ['training_logs/metric_plots.pdf', 'training_logs/training_stats.csv', 'time_log.txt',
                     'config.toml', 'model_summary.txt', 'model_structure.txt']

    data_combined = '{'
    for d in data_contents:
        data_combined = data_combined + d + ','
    data_combined = data_combined[:-1]
    data_combined += '}'

    data_combined = os.path.join(loc, name, data_combined)
    out_folder = os.path.join(out, name)
    results_folder = os.path.join(out_folder, 'training_logs')
    samples_folder = os.path.join(out_folder, 'segmentation_samples')

    create_dir_if_empty(out_folder, results_folder, samples_folder)

    command = 'scp %s:%s %s' % (server, data_combined, out_folder)
    if verbose:
        print('Command run:', command)
    process = subprocess.Popen(command,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if verbose:
        print(stdout)
        print(stderr)
    for stats_file in ['metric_plots.pdf', 'training_stats.csv']:
        shutil.move(os.path.join(out_folder, stats_file), os.path.join(results_folder, stats_file))

    if pull_last_epoch_results:
        summary_file = load_statistics(results_folder, 'training_stats.csv', config='pd')  # loads model training stats
        load_epoch = len(summary_file['Training Loss'])
        epoch_file = os.path.join(loc, name, 'segmentation_samples', 'sample_epoch_%s\*.pdf' % load_epoch)

        command = 'scp %s:%s %s' % (server, epoch_file, samples_folder)
        process = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

    if pull_best_epoch_results:
        summary_file = load_statistics(results_folder, 'training_stats.csv', config='pd')  # loads model training stats
        load_epoch = summary_file['Dice Score'].idxmax() + 1
        epoch_file = os.path.join(loc, name, 'segmentation_samples', 'sample_epoch_%s\*.pdf' % load_epoch)

        command = 'scp %s:%s %s' % (server, epoch_file, samples_folder)
        process = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()


def create_summary_table(title, columns, col_colors, data):
    """
    Creates a rich summary table.
    :param title: Table title (string)
    :param columns: Column names (list)
    :param col_colors: Colour of text within each column (list)
    :param data: Data to plot in each row.  Must be a list of tuples, where each tuple is a row
    :return: Formatted rich table
    """
    table = Table(title=title)

    for col, colour in zip(columns, col_colors):
        table.add_column(col, style=colour, no_wrap=True, justify='center')

    for param, val in data:
        table.add_row(param, str(val))
    return table


def create_dir_if_empty(*directories):
    """
    Creates a directory if it doesn't exist.
    :param directories: Single filepath or list of filepaths.
    :return: None
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def extract_image_names_from_folder(folder, sorted=True, recursive=False):
    filenames = []
    for extension in ['*.jpg', '*.png', '*.bmp', '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.jpeg', '*.JPEG', '*.JPG']:
        if recursive:
            glob_path = os.path.join(folder, '**', extension)
        else:
            glob_path = os.path.join(folder, extension)
        filenames.extend(glob.glob(glob_path, recursive=recursive))
    if sorted:
        # Sort file names in Natural Order so that numbers starting with 1s don't take priority
        filenames.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
    return filenames


def index_converter(ind, images_per_row, double_indexing=True):
    """
    Converts a singe digit index into a double digit system, if required.
    :param ind: The input single index
    :param images_per_row: The number of images per row in the output figure
    :param double_indexing: Whether or not double indexing is required
    :return: Two split indices or a single index if double indexing not necessary
    """
    if double_indexing:
        return int(ind / images_per_row), ind % images_per_row  # converts indices to double
    else:
        return ind
