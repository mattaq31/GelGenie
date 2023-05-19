from gel_tools import GelAnalysis
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
from scipy import ndimage as ndi
from skimage.color import label2rgb
import skimage
from collections import defaultdict
import pandas as pd
import matplotlib.patches as patches


from gel_tools.band_detection import watershed_seg, mask_expansion
from gel_tools.utils import convert_pil_image_base_64, convert_numpy_image_base_64

if __name__ == '__main__':

    analyser = GelAnalysis("../scratch_data/2_cropped.png", image_type='file')
    plt.figure()
    plt.imshow(analyser.np_image, cmap='gray')
    plt.show()
    im = analyser.gray_image
    max_pixel_val = 65535
    fg = (43/100) * max_pixel_val
    bg = (18/100) * max_pixel_val
    label_set = watershed_seg(im, fg, bg)

    final_segmentation = np.zeros_like(im, dtype="int32")
    final_segmentation += label_set

    plt.figure()
    plt.imshow(label_set, cmap='gray')
    plt.show()
    mask = mask_expansion(im, label_set, verbose=False, next_bg=bg-(0.005*max_pixel_val))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.show()

    labeled_fbands, _ = ndi.label(final_segmentation)
    actual_band_id = 1
    for i in range(1, labeled_fbands.max()+1):
        if (labeled_fbands == i).sum() < 50:
            labeled_fbands[labeled_fbands == i] = 0
        else:
            labeled_fbands[labeled_fbands == i] = actual_band_id
            actual_band_id += 1

    plt.figure()
    plt.imshow(labeled_fbands)
    plt.show()

    final_overlay = label2rgb(labeled_fbands, image=im, bg_label=0, bg_color=[0, 0, 0])
    plt.figure()
    plt.imshow(final_overlay)
    plt.show()

    # Invert image
    inverted_img = skimage.util.invert(im)
    # Overlay found bands on inverted image
    overlay_inverted = label2rgb(labeled_fbands, image=inverted_img, bg_label=0, bg_color=[1, 1, 1])
    plt.figure()
    plt.imshow(overlay_inverted)
    plt.show()

    # Find properties of bands
    props = skimage.measure.regionprops(labeled_fbands, im)
    props_table = skimage.measure.regionprops_table(labeled_fbands, im)

    overlay_im = Image.fromarray(np.uint8(final_overlay*255))
    overlay_im_inv = Image.fromarray(np.uint8(overlay_inverted*255))

    over_b64, over_invert_b64 = convert_pil_image_base_64(overlay_im, overlay_im_inv)
    a, b = convert_numpy_image_base_64(final_overlay, overlay_inverted)

    # Create band dictionary
    band_dict = defaultdict(list)

    # Filter bands with area less than 50, and find props of those larger than 50
    for band_no, region_object in enumerate(props, 1):

        if region_object.area >= 50:  # area threshold for a band
            # Calculate and append weighted area
            weighted_area = round(region_object.mean_intensity.item() * region_object.area.item() / (255 * 255))
            # Add relevant props to dictionary
            band_dict["id"].append(band_no)  # cannot be changed by user
            band_dict["label"].append(band_no)  # default label is band number, but can be changed
            band_dict["center_x"].append(region_object.centroid[1])
            band_dict["center_y"].append(region_object.centroid[0])
            band_dict["center"].append(region_object.centroid)
            band_dict["area"].append(region_object.area.item())
            band_dict["w_area"].append(weighted_area)
            band_dict["c_area"].append(weighted_area)
            band_dict["bbox"].append(region_object.bbox)
            # TODO: band dict shouldn't be saved as an attribute

    band_dataframe = pd.DataFrame.from_dict(band_dict).set_index(['id'])

    fig, ax = plt.subplots()
    plt.imshow(overlay_inverted)
    for id, bbox in zip(band_dict['id'], band_dict['bbox']):
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


    import base64
    import io

    profile = skimage.measure.profile_line(im, [0, 400], [1000, 400], linewidth=7, reduce_func=np.mean)

    fig = plt.figure(figsize=(4, 11), facecolor='black')
    plt.plot(profile, range(len(profile)))
    plt.gca().invert_yaxis()
    plt.axis("off")
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    plt.show()
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('utf-8')

    profile_image = my_stringIObytes

    with open('/Users/matt/Desktop/tst.tif', "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(profile_image.getbuffer())

