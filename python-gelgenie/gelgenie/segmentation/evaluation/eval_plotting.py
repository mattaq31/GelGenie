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

"""
This file plots out the test set metric data and compares different models and datasets.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# fixed terms
metrics = ['Dice Score', 'MultiClass Dice Score', 'True Negatives', 'False Positives', 'False Negatives','True Positives']
data_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/full_test_set_eval'
output_folder = '/Users/matt/Desktop'
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Helvetica'
title_fontsize = 20
axis_fontsize = 40
tick_fontsize = 24
label_fontsize = 32
metric_of_interest = 'Dice Score'

# images corresponding to each dataset
dataset_titles = ['Matthew Gels 1', 'Matthew Gels 2', 'Nathan Gels', 'LSDB Gels', 'Quantitation Gels']
mg_1 = ['8', '13', '38', '62', '81', '105', '108', '114', '128', '132', '140', '143', '146', '161', '168', '179', '183',
        '187', '205', '214', '220', '230', '235', '242', '251', '257', '263', '292', '307', '312']
mg_2 = ['0', '7', '11', '32', '49', '176', '201', '216', 'mg2_214']
ng = ['UVP01944May172019', 'UVP01947May172019', 'UVP01949May172019', 'UVP02164June252019']
lsdb = ['C50-4', 'C30194', 'C51416', 'C53007', 'C60248', 'C61344', 'E864']
quantg = ['1_Thermo', '8_Thermo', '25_NEB', '29_NEB']

# loading and gathering datasets
datasets = {}
for m in metrics:
    datasets[m] = pd.read_csv(os.path.join(data_folder, 'metrics', m + '.csv'), index_col=0)

# custom name for each column/model
models = datasets[m].columns
custom_names = ['Old U-Net', 'Custom\n U-Net', 'Custom U-Net\n (Extended)', 'LSDB-only\n U-Net',
                'LSDB-only U-Net\n (Extended)', 'Watershed', 'Multi-Otsu', 'nnU-Net', 'nnU-Net\n (Extended)']

# actual data for plotting
zoom_slice = [1, 7, 3, 5, 6]
tick_points = [0, 0.2, 0.4, 0.6, 0.8, 1]
zoom_names = [custom_names[i] for i in zoom_slice]
zoom_columns = [models[i] for i in zoom_slice]
target_df = datasets[metric_of_interest].iloc[:, zoom_slice]

# full bar plot of all models and datasets
fig, ax = plt.subplots(figsize=(15, 12))

[x.set_linewidth(2.5) for x in ax.spines.values()]  # makes border thicker
# Set the color of the spines to black
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

barplot = sns.barplot(x=zoom_columns, y=target_df.loc['mean'])
plt.title('Full Test Set (n=54)', fontsize=title_fontsize)
ax.set_xticks(ax.get_xticks())  # just to silence annoying warning
barplot.set_xticklabels(zoom_names, fontsize=label_fontsize)
barplot.tick_params(axis='y', labelsize=tick_fontsize)

# plt.xlabel('Model', fontsize=axis_fontsize)
plt.ylabel('Dice Score', fontsize=axis_fontsize)
plt.yticks(tick_points)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
# plt.savefig(os.path.join(output_folder, 'full_test_set_barplot.png'), dpi=300)
plt.show()

sns.set(style="whitegrid")
sel_palette = 'bright'

# box plots for selective figures
for selection, title in zip([mg_1 + mg_2 + ng + quantg + lsdb, mg_1 + mg_2 + ng + quantg, lsdb], ['All Gels (n=54)', 'Standard Gels (n=47)', 'LSDB Gels (n=7)']):

    fig, ax = plt.subplots(figsize=(15, 12))
    [x.set_linewidth(2.5) for x in ax.spines.values()]  # makes border thicker
    # Set the color of the spines to black
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    boxplot = sns.boxplot(data=target_df.loc[selection], showfliers=False,
                          notch=True, palette=sel_palette, linewidth=2.5)
    sns.stripplot(data=target_df.loc[selection], jitter=0.15, legend=False, dodge=True,
              alpha=1.0, marker="o", palette=sel_palette, linewidth=0.8)

    plt.title(title, fontsize=title_fontsize)  # seems to overlap with figure, making tiny for now then will replace in post
    # Set font size for ticks
    boxplot.tick_params(axis='x', labelsize=tick_fontsize)
    boxplot.tick_params(axis='y', labelsize=tick_fontsize)

    # Add labels and title
    # plt.xlabel('Model', fontsize=axis_fontsize)
    plt.ylabel('Dice Score', fontsize=axis_fontsize)
    plt.yticks(tick_points)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(ax.get_xticks())  # just to silence annoying warning

    boxplot.set_xticklabels(zoom_names, fontsize=label_fontsize)
    plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, f'{title}_boxplot.png'), dpi=300)
    plt.show()
    break
# confusion matrices seem mostly pointless due to the high class imbalance
# for m_ind, model in enumerate(models):
#     conf_matrix = np.zeros((2, 2))
#     for index, m in enumerate(['True Negatives', 'False Positives', 'False Negatives', 'True Positives']):
#         conf_matrix[index // 2, index % 2] = sum(datasets[m].iloc[:-1].values)[m_ind]
#     conf_matrix = conf_matrix / conf_matrix.sum(axis=0)
#     sns.heatmap(conf_matrix, annot=True)
#     plt.title(model)
#     plt.show()




# PLOTTING V1

    # fig, ax = plt.subplots(figsize=(15, 12))
    # [x.set_linewidth(2.5) for x in ax.spines.values()]  # makes border thicker
    # # Set the color of the spines to black
    # ax.spines['top'].set_color('black')
    # ax.spines['bottom'].set_color('black')
    # ax.spines['left'].set_color('black')
    # ax.spines['right'].set_color('black')
    # boxplot = sns.boxplot(data=target_df.loc[selection], showfliers=False)
    # sns.stripplot(data=target_df.loc[selection], jitter=True, marker="o", palette='dark:black', size=8, alpha=0.6)
    #
    # plt.title(title, fontsize=title_fontsize)  # seems to overlap with figure, making tiny for now then will replace in post
    # # Set font size for ticks
    # boxplot.tick_params(axis='x', labelsize=tick_fontsize)
    # boxplot.tick_params(axis='y', labelsize=tick_fontsize)
    #
    # # Add labels and title
    # # plt.xlabel('Model', fontsize=axis_fontsize)
    # plt.ylabel('Dice Score', fontsize=axis_fontsize)
    # plt.yticks(tick_points)
    # ax.set_ylim(-0.05, 1.05)
    #
    # ax.set_xticks(ax.get_xticks())  # just to silence annoying warning
    #
    # boxplot.set_xticklabels(zoom_names, fontsize=label_fontsize)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, f'{title}_boxplot.png'), dpi=300)
    # plt.show()
