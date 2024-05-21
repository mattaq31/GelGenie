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

in_folder = '/Users/matt/Desktop/seg_analysis'
eval_models = ['unet_dec_21', 'nnunet_final_fold_0', 'unet_dec_21_lsdb_only', 'watershed', 'multiotsu']
mg_1 = ['8', '13', '38', '62', '81', '105', '108', '114', '128', '132', '140', '143', '146', '161', '168', '179', '183',
        '187', '205', '214', '220', '230', '235', '242', '251', '257', '263', '292', '307', '312']
mg_2 = ['0', '7', '11', '32', '49', '176', '201', '216', 'mg2_214']
ng = ['UVP01944May172019', 'UVP01947May172019', 'UVP01949May172019', 'UVP02164June252019']
lsdb = ['C50-4', 'C30194', 'C51416', 'C53007', 'C60248', 'C61344', 'E864']
quantg = ['1_Thermo', '8_Thermo', '25_NEB', '29_NEB']

gel_level_dict = pickle.load(open(os.path.join(in_folder, 'gel_level_stats_no_ceiling.pkl'), 'rb'))

ceiling = False

for dataset_combo, data_name in zip([mg_1 + mg_2 + ng + quantg, lsdb], ['Standard Gels (n=47)', 'LSDB Gels (n=7)']):
    model_level_error_average = [0] * 5
    model_level_id_average = [0] * 5

    dataset_level_id_average = defaultdict(list)
    dataset_level_error_average = defaultdict(list)

    for key, val in gel_level_dict.items():
        if key not in dataset_combo:
            continue
        for model_ind, model in enumerate(eval_models):
            id_list = val['identified'][model_ind]
            error_list = [max(1-x, 0) for x in val['error'][model_ind]]
            dataset_level_error_average[model].extend(error_list)
    #     print('----')
    # print('---------')

    id_dataset = pd.DataFrame.from_dict(dataset_level_id_average)
    err_dataset = pd.DataFrame.from_dict(dataset_level_error_average)

    plt.figure(figsize=(15, 10))
    sns.violinplot(data=err_dataset)
    # sns.boxplot(data=err_dataset, showfliers=False)
    # sns.stripplot(data=err_dataset, jitter=True, marker="o", palette='dark:black', size=8, alpha=0.6)
    plt.title(data_name)
    plt.ylim([-0.05, 1.05])
    plt.show()
