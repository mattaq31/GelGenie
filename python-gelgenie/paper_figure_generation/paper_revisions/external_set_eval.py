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

from os.path import join
from gelgenie.segmentation.evaluation.core_functions import segment_and_plot, segment_and_quantitate
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty
from gelgenie.segmentation.evaluation import model_eval_load


output_folder = '/Users/matt/Desktop/'
model_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023'

input_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/gels_for_paper_revisions/assembled_set/images']
mask_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/gels_for_paper_revisions/assembled_set/masks']

classical_analysis = True
multi_augment = False
model_and_epoch = [('unet_dec_21', 'best'),
                   ('unet_dec_21_finetune', '590')]
experiment_names, eval_epochs = zip(*model_and_epoch)

models = []

for experiment, eval_epoch in zip(experiment_names, eval_epochs):

    if 'nov_4' in experiment:
       exp_folder = join('/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/November 2023', experiment)
    else:
       exp_folder = join(model_folder, experiment)

    model = model_eval_load(exp_folder, eval_epoch)
    models.append(model)

segment_and_quantitate(models, list(experiment_names), input_folder, mask_folder, join(output_folder, 'external_eval_percentile_norm'),
                       multi_augment=multi_augment, run_classical_techniques=classical_analysis, percentile_norm=True)
segment_and_quantitate(models, list(experiment_names), input_folder, mask_folder, join(output_folder, 'external_eval_standard'),
                       multi_augment=multi_augment, run_classical_techniques=classical_analysis)

