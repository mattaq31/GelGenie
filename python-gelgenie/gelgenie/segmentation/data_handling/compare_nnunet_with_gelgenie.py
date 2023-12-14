import os
import imageio
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder, create_dir_if_empty
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb
import cv2
from tqdm import tqdm


nnunet_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_results/test_aug29_nnunet/'
gelgenie_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_results/test_aug29/smp_unet_aug26_1'
ref_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/gel_testing'
out_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_results/test_aug29_nnunet/gelgenie_comparison'


nnunet_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_results/test_aug29_nnunet/neb_only'
gelgenie_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_results/test_aug29_neb/smp_unet_aug26_1'
ref_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/gel_testing/neb_only'
out_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_results/test_aug29_nnunet/gelgenie_comparison_neb'

nnunet_folder = '/Users/matt/Desktop/nnunet_output'
gelgenie_folder = '/Users/matt/Desktop/gelgenie_output/unet_global_padding_nov_4'
ref_folder = '/Users/matt/Desktop/input_data'
out_folder = '/Users/matt/Desktop/gelgenie_nnunet_comparison'

ref_files = extract_image_names_from_folder(ref_folder)
gelgenie_files = extract_image_names_from_folder(gelgenie_folder)
nnunet_files = extract_image_names_from_folder(nnunet_folder)

create_dir_if_empty(out_folder)

for i, (rfile, gfile, nfile) in tqdm(enumerate(zip(ref_files, gelgenie_files, nnunet_files))):
    basename = os.path.basename(rfile).split('.')[0]
    n_im = imageio.v2.imread(nfile)
    g_im = imageio.v2.imread(gfile)
    r_im = imageio.v2.imread(rfile)
    if r_im.shape[-1] == 4:  # Actual input: 4 channels
        r_im = cv2.cvtColor(r_im, cv2.COLOR_RGBA2RGB)

    labels, _ = ndi.label(n_im)
    n_labels = label2rgb(labels, image=r_im)

    # results preview
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))

    ax[0].imshow(r_im, cmap='gray')
    ax[0].set_title('Reference Image')

    ax[1].imshow(g_im)
    ax[1].set_title('GelGenie Prediction (4 Nov)')

    ax[2].imshow(n_labels)
    ax[2].set_title('nnUNet Prediction')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, '%s.png' % basename), dpi=300)
    plt.close(fig)

