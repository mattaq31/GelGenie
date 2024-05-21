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


output_folder = '/Users/matt/Desktop/finetune_test_set_eval_590'
model_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023'

input_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_images',
                '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_images',
                '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/test_images',
                '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/test_images',
                '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/test_images']

mask_folder = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/test_masks']

run_quant_analysis = True
classical_analysis = False
multi_augment = False

model_and_epoch = [('unet_dec_21_finetune', '590')]

experiment_names, eval_epochs = zip(*model_and_epoch)

models = []

for experiment, eval_epoch in zip(experiment_names, eval_epochs):

    if 'nov_4' in experiment:
       exp_folder = join('/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/November 2023', experiment)
    else:
       exp_folder = join(model_folder, experiment)

    model = model_eval_load(exp_folder, eval_epoch)
    models.append(model)

create_dir_if_empty(output_folder)

if run_quant_analysis:
    segment_and_quantitate(models, list(experiment_names), input_folder, mask_folder, output_folder,
                           multi_augment=multi_augment, run_classical_techniques=classical_analysis)
else:
    segment_and_plot(models, list(experiment_names), input_folder, output_folder, multi_augment=multi_augment,
                     run_classical_techniques=classical_analysis)
