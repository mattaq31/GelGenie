from os.path import join
from gelgenie.segmentation.evaluation.core_functions import segment_and_plot, segment_and_quantitate
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty
from gelgenie.segmentation.evaluation import model_eval_load


output_folder = '/Users/matt/Desktop/full_test_set_eval'
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
classical_analysis = True
multi_augment = False

model_and_epoch = [('unet_global_padding_nov_4', 'best'),
                   ('unet_dec_21', 'best'),
                   ('unet_dec_21_extended_set', '600'),
                   ('unet_dec_21_lsdb_only', 'best'),
                   ('unet_dec_21_lsdb_only_extended_set', '600')]

experiment_names, eval_epochs = zip(*model_and_epoch)

models = []

nnunet_models_and_folders = [['nnunet_final_fold_all', '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/test_inference/epoch_600_fold_all'],
                             ['nnunet_final_fold_0', '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/test_inference/epoch_best_fold_0']]


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
                           multi_augment=multi_augment, run_classical_techniques=classical_analysis,
                           nnunet_models_and_folders=nnunet_models_and_folders)
else:
    segment_and_plot(models, list(experiment_names), input_folder, output_folder, multi_augment=multi_augment,
                     run_classical_techniques=classical_analysis, nnunet_models_and_folders=nnunet_models_and_folders)
