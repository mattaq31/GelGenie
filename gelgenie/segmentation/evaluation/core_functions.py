import os
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb
from tqdm import tqdm
import math


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


def segment_and_analyze(models, model_names, input_folder, output_folder):

    dataset = ImageDataset(input_folder, 1, padding=False, individual_padding=True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
    images_per_row = 2

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        np_image = batch['image'].detach().squeeze().cpu().numpy()
        all_model_outputs = []
        for model in models:
            _, mask = model_predict_and_process(model, batch['image'])

            labels, _ = ndi.label(mask.argmax(axis=0))
            rgb_labels = label2rgb(labels, image=np_image)
            all_model_outputs.append(rgb_labels)

        # results preview
        fig, ax = plt.subplots(math.ceil((len(all_model_outputs) + 1)/images_per_row), images_per_row, figsize=(15, 15))

        zero_ax_index = index_converter(0, images_per_row)
        ax[zero_ax_index].imshow(np_image, cmap='gray')
        ax[zero_ax_index].set_title('Reference Image')

        for index, (mask, name) in enumerate(zip(all_model_outputs, model_names)):
            plot_index = index_converter(index+1, images_per_row)
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

