import os
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb
from tqdm import tqdm

ref_data_folder = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)),
                               'data_analysis', 'ref_data')


def model_predict_and_process(model, image):
    with torch.no_grad():
        mask = model(image)
        one_hot = F.one_hot(mask.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
        ordered_mask = one_hot.numpy().squeeze()
    return mask, ordered_mask


def segment_and_analyze(model, input_folder, output_folder):
    dataset = ImageDataset(input_folder, 1, padding=True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        np_image = batch['image'].detach().squeeze().cpu().numpy()
        _, mask = model_predict_and_process(model, batch['image'])
        labels, _ = ndi.label(mask.argmax(axis=0))
        rgb_labels = label2rgb(labels, image=np_image)

        # results preview
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(np_image, cmap='gray')
        ax[1].imshow(rgb_labels)

        ax[0].set_title('Reference Image')
        ax[1].set_title('Segmented Image')

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.suptitle('Segmentation result for image %s' % batch['image_name'][0])
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '%s segmentation.pdf' % batch['image_name'][0]), dpi=300)
