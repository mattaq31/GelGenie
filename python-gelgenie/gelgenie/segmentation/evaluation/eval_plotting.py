import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

metrics = ['Dice Score', 'MultiClass Dice Score', 'True Negatives', 'False Positives', 'False Negatives',
           'True Positives']
data_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/full_test_set_eval'
output_folder = '/Users/matt/Desktop'

mg_1 = ['8', '13', '38', '62', '81', '105', '108', '114', '128', '132', '140', '143', '146', '161', '168', '179', '183',
        '187', '205', '214', '220', '230', '235', '242', '251', '257', '263', '292', '307', '312']
mg_2 = ['0', '7', '11', '32', '49', '176', '201', '216', 'mg2_214']
ng = ['UVP01944May172019', 'UVP01947May172019', 'UVP01949May172019', 'UVP02164June252019']
lsdb = ['C50-4', 'C30194', 'C51416', 'C53007', 'C60248', 'C61344', 'E864']
quantg = ['1_Thermo', '8_Thermo', '25_NEB', '29_NEB']
titles = ['Matthew Gels 1', 'Matthew Gels 2', 'Nathan Gels', 'LSDB Gels', 'Quantitation Gels']
datasets = {}
sns.set(style="whitegrid")
# Set the default font family to Helvetica
plt.rcParams['font.family'] = 'Helvetica'
for m in metrics:
    datasets[m] = pd.read_csv(os.path.join(data_folder, 'metrics', m + '.csv'), index_col=0)
models = datasets[m].columns
custom_names = ['Old UNet', 'Universal UNet', 'Universal UNet\n (Extended)', 'LSDB UNet', 'LSDB UNet\n (Extended)', 'Watershed', 'Multi-Otsu', 'nn-UNet', 'nn-UNet\n (Extended)']
title_fontsize = 18
axis_fontsize = 18
tick_fontsize = 10
label_fontsize = 12

met = 'Dice Score'

# full bar plot
plt.figure(figsize=(15, 10))
barplot = sns.barplot(x=models, y=datasets[met].loc['mean'])
# plt.xticks(rotation='vertical')
plt.title('Full Test Set', fontsize=title_fontsize)
barplot.set_xticklabels(custom_names, fontsize=label_fontsize)
# Add labels and title
plt.xlabel('Model', fontsize=axis_fontsize)
plt.ylabel('Dice Score', fontsize=axis_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'full_test_set_barplot.png'), dpi=300)
plt.show()

# box plots for selective figures
for selection, title in zip([mg_1 + mg_2 + ng + quantg + lsdb, mg_1 + mg_2 + ng + quantg, lsdb], ['All Gels', 'Standard Gels', 'LSDB Gels']):
    plt.figure(figsize=(15, 10))
    boxplot = sns.boxplot(data=datasets[met].loc[selection])
    # plt.xticks(rotation='vertical')
    plt.title(title, fontsize=title_fontsize)

    # Set font size for ticks
    boxplot.tick_params(axis='x', labelsize=tick_fontsize)
    boxplot.tick_params(axis='y', labelsize=tick_fontsize)

    # Add labels and title
    plt.xlabel('Model', fontsize=axis_fontsize)
    plt.ylabel('Dice Score', fontsize=axis_fontsize)
    boxplot.set_xticklabels(custom_names, fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{title}_boxplot.png'), dpi=300)
    plt.show()

# met = 'Dice Score'
# for selection, title in zip([mg_1, mg_2, ng, lsdb, quantg], titles):
#     plt.figure(figsize=(10, 10))
#     # plt.bar(models, datasets[met].loc[selection].mean(axis=0))
#     sns.boxplot(data=datasets[met].loc[selection])
#     plt.xticks(rotation='vertical')
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()



# confusion matrices seem mostly pointless due to the high class imbalance
# for m_ind, model in enumerate(models):
#     conf_matrix = np.zeros((2, 2))
#     for index, m in enumerate(['True Negatives', 'False Positives', 'False Negatives', 'True Positives']):
#         conf_matrix[index // 2, index % 2] = sum(datasets[m].iloc[:-1].values)[m_ind]
#     conf_matrix = conf_matrix / conf_matrix.sum(axis=0)
#     sns.heatmap(conf_matrix, annot=True)
#     plt.title(model)
#     plt.show()
