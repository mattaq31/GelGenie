import os
import shutil


inference_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/test_inference/epoch_best_fold_0'
# inference_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/test_inference/epoch_600_fold_all'

conv_file = os.path.join(inference_folder, 'image_mapper.txt')

with open(conv_file, 'r') as f:
    lines = f.readlines()

lines = [(x.split(',')[0].split('.')[0] + '.tif', x.split(',')[1].replace('\n', '').replace('_0000.', '.')) for x in lines]

for i in lines:
    shutil.move(os.path.join(inference_folder, i[1]), os.path.join(inference_folder, i[0]))
