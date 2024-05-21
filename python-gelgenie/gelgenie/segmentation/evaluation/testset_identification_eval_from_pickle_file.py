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

import os
from collections import defaultdict

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import imageio
import seaborn as sns

in_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/full_test_set_eval/metrics/band_level_accuracy_analysis'
eval_models = ['unet_dec_21', 'nnunet_final_fold_0', 'unet_dec_21_lsdb_only', 'watershed', 'multiotsu']
mg_1 = ['8', '13', '38', '62', '81', '105', '108', '114', '128', '132', '140', '143', '146', '161', '168', '179', '183',
        '187', '205', '214', '220', '230', '235', '242', '251', '257', '263', '292', '307', '312']
mg_2 = ['0', '7', '11', '32', '49', '176', '201', '216', 'mg2_214']
ng = ['UVP01944May172019', 'UVP01947May172019', 'UVP01949May172019', 'UVP02164June252019']
lsdb = ['C50-4', 'C30194', 'C51416', 'C53007', 'C60248', 'C61344', 'E864']
quantg = ['1_Thermo', '8_Thermo', '25_NEB', '29_NEB']

gel_level_dict = pickle.load(open(os.path.join(in_folder, 'gel_level_stats_no_ceiling.pkl'), 'rb'))

ceiling = False

for dataset_combo, data_name in zip([mg_1 + mg_2 + ng + quantg + lsdb], ['All Gels (n=54)']):
    model_level_accuracy_average = [0] * 5
    model_level_id_average = [0] * 5

    dataset_level_id_average = defaultdict(list)
    dataset_level_error_average = defaultdict(list)
    dataset_level_full_accuracy_list = defaultdict(list)

    for key, val in gel_level_dict.items():
        if key not in dataset_combo:
            continue
        for model_ind, model in enumerate(eval_models):
            id_list = val['identified'][model_ind]
            accuracy_list = [max(1 - x, 0) for x in val['error'][model_ind]]
            dataset_level_full_accuracy_list[model].extend(
                [max(1 - x, 0) if val['multi_band'][model_ind][ind] is False else 0 for ind, x in
                 enumerate(val['error'][model_ind])])
            if sum(id_list) == 0:
                percent_identified = 0
                average_accuracy_for_identified = 0
            else:
                percent_identified = (sum(id_list) / len(id_list)) * 100
                if ceiling:
                    average_accuracy_for_identified = sum([err for id, err in zip(id_list, accuracy_list) if id]) / sum(
                        id_list)
                else:
                    average_accuracy_for_identified = sum(accuracy_list) / len(accuracy_list)

            print(
                f'For gel {key}, {model} identified {percent_identified:.2f}% of bands, with an average accuracy of {average_accuracy_for_identified:.3f}')
            model_level_id_average[model_ind] += percent_identified
            model_level_accuracy_average[model_ind] += average_accuracy_for_identified
            dataset_level_id_average[model].append(percent_identified)
            dataset_level_error_average[model].append(average_accuracy_for_identified)
        print('----')
    print('---------')

    model_level_accuracy_average = [x / len(dataset_combo) for x in model_level_accuracy_average]
    model_level_id_average = [x / len(dataset_combo) for x in model_level_id_average]

    print('Final averages:')
    print(model_level_accuracy_average)
    print(model_level_id_average)

    id_dataset = pd.DataFrame.from_dict(dataset_level_id_average)
    accuracy_dataset = pd.DataFrame.from_dict(dataset_level_error_average)

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=id_dataset, showfliers=False)
    sns.stripplot(data=id_dataset, jitter=True, marker="o", palette='dark:black', size=8, alpha=0.6)
    plt.title('Identified')
    # plt.show()

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=accuracy_dataset, showfliers=False)
    sns.stripplot(data=accuracy_dataset, jitter=True, marker="o", palette='dark:black', size=8, alpha=0.6)
    plt.title('Accuracy')
    plt.ylim([-0.05, 1.05])
    # plt.show()

    full_accuracy_dataset = pd.DataFrame.from_dict(dataset_level_full_accuracy_list)
    full_accuracy_dataset.rename(columns={'unet_dec_21': 'Custom\n U-Net',
                                          'nnunet_final_fold_0': 'nnU-Net',
                                          'unet_dec_21_lsdb_only': 'LSDB-Only\n U-Net',
                                          'watershed': 'Watershed',
                                          'multiotsu': 'Multi-Otsu'}, inplace=True)

    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'Helvetica'

    fig, ax = plt.subplots(figsize=(10, 8))
    vplot = sns.violinplot(data=full_accuracy_dataset, cut=0,
                           # inner_kws=dict(box_width=10, whis_width=2),
                           palette='bright', inner='quartile')

    # Set the color of the spines to black
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel('Segmentation Accuracy', fontsize=40)
    plt.xlabel('Model', fontsize=40)
    plt.yticks(fontsize=22)
    vplot.tick_params(axis='x', labelsize=22)
    plt.tight_layout()
    plt.savefig('/Users/matt/Desktop/full_gel_level_accuracy_violin.pdf')
    plt.show()
