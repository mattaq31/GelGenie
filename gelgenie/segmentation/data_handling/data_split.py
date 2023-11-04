import os
from os.path import join
from segmentation.helper_functions.general_functions import extract_image_names_from_folder, create_dir_if_empty
import numpy as np
import shutil


base_dir = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set'

all_datasets = ['lsdb_gels', 'matthew_gels', 'matthew_gels_2', 'nathan_gels', 'neb_ladders']

val_percent = 0.1
test_percent = 0.1
global_count = 0

for dataset in all_datasets:
    image_folder = join(base_dir, dataset, 'images')
    mask_folder = join(base_dir, dataset, 'masks')
    val_folder = join(base_dir, dataset, 'val_images')
    val_mask_folder = join(base_dir, dataset, 'val_masks')
    test_folder = join(base_dir, dataset, 'test_images')
    test_mask_folder = join(base_dir, dataset, 'test_masks')

    create_dir_if_empty(val_folder, val_mask_folder, test_folder, test_mask_folder)

    all_images = extract_image_names_from_folder(image_folder)

    base_names = [os.path.basename(x) for x in all_images]
    mask_names = [os.path.basename(x).split('.')[0] + '.tif' for x in all_images]
    all_masks = [join(mask_folder, x) for x in mask_names]

    image_count = len(base_names)

    samples = np.random.choice(image_count, int(image_count * (val_percent+test_percent)), replace=False)
    val_samples = samples[:int(image_count * val_percent)]
    test_samples = samples[int(image_count * val_percent):]

    val_images = [base_names[y] for y in val_samples]
    val_masks = [mask_names[y] for y in val_samples]

    test_images = [base_names[y] for y in test_samples]
    test_masks = [mask_names[y] for y in test_samples]
    global_count += image_count

    for im_base, mask_base in zip(val_images, val_masks):
        shutil.move(join(image_folder, im_base), join(val_folder, im_base))
        shutil.move(join(mask_folder, mask_base), join(val_mask_folder, mask_base))

    for im_base, mask_base in zip(test_images, test_masks):
        shutil.move(join(image_folder, im_base), join(test_folder, im_base))
        shutil.move(join(mask_folder, mask_base), join(test_mask_folder, mask_base))



