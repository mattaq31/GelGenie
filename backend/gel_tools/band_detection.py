import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage.filters import sobel
import skimage.util as util
from scipy import ndimage as ndi
from skimage.color import label2rgb



# Create histogram of gray values in image
# Can be used to determine bg and fg values in testing
def make_histogram(image):
    """
    # Create histogram of gray values in image
    # Can be used to determine bg and fg values in testing
    """
    # Actual making of histogram
    hist = np.histogram(image, bins=np.arange(0, 256))
    #print(hist)
    # Graphing the histogram next to the image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.plot(hist[1][:-1], hist[0], lw=2)
    ax2.set_title('histogram of grey values')
    plt.show()
    # Return histogram
    return hist


def watershed_seg(image, sure_fg, sure_bg, verbose=False):
    """
    Apply watershed algorithm to find bands in a given image.
    :param image: numpy array containing image raw data (grayscale, 16-bit)
    :param sure_fg: Threshold value above which a band is definitely present.
    :param sure_bg: Threshold value under which a band is definitely not present.
    :param verbose:  Set to true to print out figures containing results immediately.
    :return: 2D array of same shape as orig. image with unique band labels
    """

    # Use Sobel filter on original image to find elevation map
    elevation_map = sobel(image)

    # Define markers for the background and foreground of the image
    markers = np.zeros_like(image)
    markers[image < sure_bg] = 1
    markers[image > sure_fg] = 2

    # Apply the watershed algorithm itself, using the elevation map and markers
    segmentation = skimage.segmentation.watershed(elevation_map, markers)

    # Fill holes and relabel bands, giving each a unique label
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_bands, _ = ndi.label(segmentation)

    # Overlay labels on original image
    image_label_overlay = label2rgb(labeled_bands, image=image)

    if verbose:
        plt.figure()
        plt.title("Region-based Segmentation")
        plt.imshow(image_label_overlay)
        print('Band Labels:')
        print(labeled_bands)

    return labeled_bands





def mask_expansion(original_image, labeled_image, next_bg, verbose=False):
    """
    Expands the areas of found bands to those adjacent pixels which
    would potentially make bands on the next pass.
    :param original_image: Raw gel image (ndarray, 16-bit).
    :param labeled_image: Image with segmented bands.

    :param verbose: Set to true to print diagnostic results.
    :return: Image mask containing areas where bands are already present (or have been expanded via flood algorithm)
    """

    # Get properties of the labeled bands (will use to find centroid of bands)
    props = skimage.measure.regionprops(labeled_image)

    # Initialize output image
    output_image = np.zeros_like(original_image)

    for band in props:
        # Define centroid
        centroid = band.centroid
        # Set seed point to closest integer coord pair to centroid
        seed_point = (round(centroid[0]), round(centroid[1]))

        # Initialize flooding image as full of zeros
        image_for_flood = np.zeros_like(original_image)

        # Set every pixel in flooding image that could possibly make a new band next pass to 1
        image_for_flood[original_image > next_bg] = 1

        # Use floodfill on the flooding image using the found bands
        # This will ideally exclude any areas that neighbour found bands from being considered as bands
        output_image_part = skimage.morphology.flood(image_for_flood, seed_point, tolerance=None)

        # Add the excluded areas of each iteration to final excluded area
        output_image += output_image_part

        if verbose:
            print('Seed point for band %s:' % band.label, seed_point)
            print('=====')
            plt.figure()
            plt.title("Flood band %s" % band.label)
            plt.imshow(output_image_part)

    return output_image


# TODO: this function needs to be split up and made more robust
def find_bands(img, sure_fg, sure_bg, repetitions, background_jump=0.005, verbose=False, minimum_pixel_count=50):
    """
    # Function which brings it all together and actually finds bands.
    # Takes the source image and watershed segmentation algorithm parameters.
    # Returns white-on-black and black-on-white images with colored bands, and band properties.
    """
    if verbose:
        print('Band Finding Parameters:')
        print("sure fg: ", sure_fg)
        print("sure bg: ", sure_bg)
        print("repetitions: ", repetitions)
        print("===============")

    # Create copy of loaded image to apply mask to
    working_img = img.copy()

    # Create template for labels (each integer label corresponds to band ID)
    final_segmentation = np.zeros_like(img, dtype="int32")

    if img.dtype == 'uint16':
        max_pixel_val = 65535
    else:
        max_pixel_val = 255

    # Repeat the watershed algorithm "repetitions" times
    for i in range(0, repetitions):
        # Run watershed algorithm
        segmented_image = watershed_seg(working_img, sure_fg, sure_bg)

        # Add found bands from this iteration to all found bands
        final_segmentation += segmented_image

        # Create mask
        mask = mask_expansion(working_img, segmented_image,  next_bg=sure_bg-background_jump*max_pixel_val)

        # Apply mask (removing areas of image which should no longer be explored)
        working_img[mask > 0] = 0

        if verbose:
            plt.figure()
            plt.title("Remaining parts of image without bands after repetition %d" % i)
            plt.imshow(working_img)

        # Reduce fg and bg values on each pass
        sure_fg -= background_jump*max_pixel_val
        sure_bg -= background_jump*max_pixel_val

    # Relabel bands to ensure correct labelling
    labeled_fbands, _ = ndi.label(final_segmentation)

    actual_band_id = 1
    for i in range(1, labeled_fbands.max()+1):  # filtering step - removes all segmented sections which are too small to be usable
        if (labeled_fbands == i).sum() < minimum_pixel_count:
            labeled_fbands[labeled_fbands == i] = 0
        else:
            labeled_fbands[labeled_fbands == i] = actual_band_id
            actual_band_id += 1

    # Overlay found bands on original image
    final_overlay = label2rgb(labeled_fbands, image=img, bg_label=0, bg_color=[0, 0, 0])  # TODO: does this always assume one type of image?

    if verbose:
        plt.figure()
        plt.title("Overlayed image")
        plt.imshow(final_overlay)

    # Invert image
    inverted_img = util.invert(img)
    # Overlay found bands on inverted image
    overlay_inverted = label2rgb(labeled_fbands, image=inverted_img, bg_label=0, bg_color=[1, 1, 1])

    # Find properties of bands
    props = skimage.measure.regionprops(labeled_fbands, img)
    props_table = skimage.measure.regionprops_table(labeled_fbands, img)

    return final_overlay, props, overlay_inverted, props_table, labeled_fbands


