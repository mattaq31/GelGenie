import os
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb
from tqdm import tqdm
import math
import imageio
import numpy as np


ref_data_folder = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)),
                               'data_analysis', 'ref_data')


def model_predict_and_process(model, image):
    with torch.no_grad():
        mask = model(image)
        one_hot = F.one_hot(mask.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
        ordered_mask = one_hot.numpy().squeeze()
    return mask, ordered_mask


def index_converter(ind, images_per_row):
    return int(ind / images_per_row), ind % images_per_row  # converts indices to double


def segment_and_analyze(models, model_names, input_folder, output_folder, minmax_norm=False):

    dataset = ImageDataset(input_folder, 1, padding=False, individual_padding=True, minmax_norm=minmax_norm)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
    images_per_row = 2
    double_indexing = True  # axes will have two indices rather than one

    if math.ceil((len(model_names) + 1)/images_per_row) == 1:  # axes will only have one index rather than 2
        double_indexing = False

    for mname in model_names:
        create_dir_if_empty(os.path.join(output_folder, mname))

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        np_image = batch['image'].detach().squeeze().cpu().numpy()
        all_model_outputs = []
        for model, mname in zip(models, model_names):
            _, mask = model_predict_and_process(model, batch['image'])

            labels, _ = ndi.label(mask.argmax(axis=0))
            rgb_labels = label2rgb(labels, image=np_image)
            all_model_outputs.append(rgb_labels)
            imageio.v2.imwrite(os.path.join(output_folder, mname, '%s.png' % batch['image_name'][0]), (rgb_labels*255).astype(np.uint8))

        # results preview
        fig, ax = plt.subplots(math.ceil((len(all_model_outputs) + 1)/images_per_row), images_per_row, figsize=(15, 15))

        if double_indexing:
            zero_ax_index = index_converter(0, images_per_row)
        else:
            zero_ax_index = 0

        ax[zero_ax_index].imshow(np_image, cmap='gray')
        ax[zero_ax_index].set_title('Reference Image')

        for index, (mask, name) in enumerate(zip(all_model_outputs, model_names)):
            if double_indexing:
                plot_index = index_converter(index+1, images_per_row)
            else:
                plot_index = index + 1

            ax[plot_index].imshow(mask)
            if len(name) > 14:
                title = name[:int(len(name)/2)] + '\n' + name[int(len(name)/2):]
            else:
                title = name
            ax[plot_index].set_title(title, fontsize=13)

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.suptitle('Segmentation result for image %s' % batch['image_name'][0])
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '%s segmentation.png' % batch['image_name'][0]), dpi=300)
        plt.close(fig)

