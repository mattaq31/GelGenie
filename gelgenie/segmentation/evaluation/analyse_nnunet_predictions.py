import os
import imageio
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb


in_folder = '/Users/matt/Desktop/test_aug29'
ref_folder = '/Users/matt/Desktop/nnunet_data'
out_folder = '/Users/matt/Desktop/comparison_data'
for i, ifile in enumerate(extract_image_names_from_folder(in_folder)):
    basename = os.path.basename(ifile)
    ref_im_base = basename.split('.')[0] + '_0000.tif'

    im = imageio.v2.imread(ifile)
    ref_im = imageio.v2.imread(os.path.join(ref_folder, ref_im_base))

    labels, _ = ndi.label(im)
    rgb_labels = label2rgb(labels, image=ref_im)

    # results preview
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))

    ax[0].imshow(ref_im, cmap='gray')
    ax[0].set_title('Reference Image')

    ax[1].imshow(rgb_labels)
    ax[1].set_title('nnUNet Prediction')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, '%s.png' % basename), dpi=300)
    plt.close(fig)
