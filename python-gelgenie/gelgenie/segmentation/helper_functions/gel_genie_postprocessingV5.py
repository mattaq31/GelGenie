import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from tabulate import tabulate
from skimage.morphology import convex_hull_image



def create_dir_if_empty(*directories):
    """
    Creates a directory if it doesn't exist.
    :param directories: Single filepath or list of filepaths.
    :return: None
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def find_nearest_points(all_blob_centroids, median_length_blobs):
    struct_nearest = []

    for k in range(all_blob_centroids.shape[0]):
        A = all_blob_centroids[k, :]
        points = all_blob_centroids
        struct_nearest.append(find_nearest_vertical_point(A, points, median_length_blobs))

    return struct_nearest


def find_nearest_vertical_point(A, points, median_length_blobs):
    x = points[:, 0]
    y = points[:, 1]

    distance_x = x - A[0]
    distance_y = y - A[1]
    valid_indices = np.where(np.abs(distance_y) < (median_length_blobs / 2))[0]
    distances = np.sqrt(distance_x[valid_indices] ** 2 + distance_y[valid_indices] ** 2)

    the_ind = valid_indices[distances == 0]

    if len(valid_indices) == 1:
        nearest_point = the_ind
        min_index = the_ind
        col_a_points = the_ind
    else:
        min_index_in_valid = np.argsort(distances)[0:2]
        min_index = valid_indices[min_index_in_valid[1]]
        nearest_point = points[min_index, :]
        col_a_points = valid_indices

    return {
        'min_index': min_index,
        'nearest_point': nearest_point,
        'col_a_points': col_a_points
    }


def separate_into_columns(struct_nearest):
    indices = [item['min_index'] for item in struct_nearest]
    groups_indices = []
    group_count = 0

    for i in range(len(indices)):
        a_group = [i]
        friend = indices[i]

        while friend not in a_group:
            a_group.append(friend)
            friend = indices[friend]

        group_count += 1
        groups_indices.append(a_group)

    merged_groups = groups_indices.copy()
    idx = 0

    while idx < len(merged_groups):
        current_array = merged_groups[idx]

        for j in range(idx + 1, len(merged_groups)):
            if any(item in current_array for item in merged_groups[j]):
                current_array = list(set(current_array) | set(merged_groups[j]))
                merged_groups[idx] = current_array
                merged_groups.pop(j)
                idx -= 1
                break

        idx += 1

    return merged_groups


def merge_bounding_boxes(merged_groups, props):
    merged_bounding_boxes = []
    merged_bounding_boxes_centroids = []

    for group in merged_groups:
        if not group:
            continue  # Skip empty groups

        # Extract bounding boxes and centroids for each object in the group
        group_boxes = [props[i].bbox for i in group]
        group_centroids = [props[i].centroid for i in group]

        # Combine bounding boxes
        min_row = min(box[0] for box in group_boxes)
        min_col = min(box[1] for box in group_boxes)
        max_row = max(box[2] for box in group_boxes)
        max_col = max(box[3] for box in group_boxes)

        merged_box = (min_row, min_col, max_row, max_col)
        merged_bounding_boxes.append(merged_box)

        # Calculate centroid of the combined bounding box
        centroid_row = (min_row + max_row) / 2
        centroid_col = (min_col + max_col) / 2
        merged_centroid = (centroid_row, centroid_col)
        merged_bounding_boxes_centroids.append(merged_centroid)
    return merged_bounding_boxes, merged_bounding_boxes_centroids


def merge_close_centroids_bb(merged_bounding_boxes, sorted_centroids, sorted_inds, merged_groups, vertical_threshold):
    merged_groups_fin = []
    merged_boxes = []

    merged_bounding_boxes = np.array(merged_bounding_boxes)  # Convert to NumPy array

    current_group_indices = [sorted_inds[0]]
    merged_groups_inds = merged_groups[sorted_inds[0]]

    for i in range(1, len(sorted_centroids)):
        if abs(sorted_centroids[i][1] - sorted_centroids[i - 1][1]) <= vertical_threshold:
            current_group_indices.append(sorted_inds[i])
            merged_groups_inds += merged_groups[sorted_inds[i]]
        else:
            # Extract bounding boxes for each object in the current group
            group_boxes = merged_bounding_boxes[current_group_indices]

            # Combine bounding boxes to encompass the entire group
            min_row = np.min(group_boxes[:, 0])
            min_col = np.min(group_boxes[:, 1])
            max_row = np.max(group_boxes[:, 2])
            max_col = np.max(group_boxes[:, 3])

            merged_groups_fin.append(
                {'mergedBoxes': [min_row, min_col, max_row, max_col], 'indices': merged_groups_inds})

            current_group_indices = [sorted_inds[i]]
            merged_groups_inds = merged_groups[sorted_inds[i]]

    # Merge the last group
    group_boxes = merged_bounding_boxes[current_group_indices]
    min_row = np.min(group_boxes[:, 0])
    min_col = np.min(group_boxes[:, 1])
    max_row = np.max(group_boxes[:, 2])
    max_col = np.max(group_boxes[:, 3])

    merged_groups_fin.append({'mergedBoxes': [min_row, min_col, max_row, max_col], 'indices': merged_groups_inds})

    # Remove empty elements from the list
    merged_groups_fin = [item for item in merged_groups_fin if item]

    merged_boxes = [group['mergedBoxes'] for group in merged_groups_fin]

    return merged_groups_fin, merged_boxes


def mad(data, axis=None):
    median = np.median(data, axis=axis)
    mad_value = np.median(np.abs(data - median), axis=axis)
    return 1.4826 * mad_value


def is_outlier(data, threshold=3):
    scaled_mad = mad(data)
    median = np.median(data)
    return np.abs(data - median) > threshold * scaled_mad


def process_merged_bounding_boxes(merged_groups_fin, merged_boxes, props, horizontal_threshold):
    all_info = []
    # Extract all centroids of props
    all_props_centroids = [prop.centroid for prop in props]

    for i in range(len(merged_boxes)):
        current_info = {'Group': i, 'BoundingBox': merged_boxes[i], 'Blobs': merged_groups_fin[i]['indices'],
                        'Props': [], 'Centroids': [], 'Broken': [], 'Small': [], 'SmallTrue': [], 'BrokenTrue': []}

        props_group = [props[j] for j in merged_groups_fin[i]['indices']]
        current_info['Props'] = props_group
        blobsInds = merged_groups_fin[i]['indices']

        # Extract the horizontal length of each original bounding box
        horizontal_lengths = [prop.bbox[3] - prop.bbox[1] for prop in props_group]

        # Find the outliers
        tf = is_outlier(horizontal_lengths)
        true_outliers = np.array(blobsInds)[tf]
        blob_centroids = np.array([prop.centroid for prop in props_group])
        blob_centroids_original = blob_centroids
        current_info['Centroids'] = blob_centroids

        if np.sum(tf) > 0:
            if np.sum(tf) == 1:
                # only one abnormally small blob - count it as a normal blob, do
                # nothing but save the index
                broken = np.where(tf)[0]
                current_info['Small'] = broken.tolist()
                current_info['SmallTrue'] = true_outliers.tolist()

            elif np.sum(tf) == 2:
                broken = np.where(tf)[0]

                wherep1 = blobsInds[broken[0]]
                wherep2 = blobsInds[broken[1]]
                wherep1True = true_outliers[0]
                wherep2True = true_outliers[1]
                p1 = np.array(all_props_centroids[true_outliers[0]])
                p2 = np.array(all_props_centroids[true_outliers[1]])

                # check if they are in the same line, else do nothing
                if np.abs(p1[0] - p2[0]) <= horizontal_threshold:
                    current_info['Broken'] = broken.tolist()
                    current_info['BrokenTrue'].append(list(np.array([wherep1True, wherep2True])))
                    new_centroid = (p1 + p2) / 2
                    blob_centroids[broken[0]] = new_centroid
                    blob_centroids = np.delete(blob_centroids, broken[1], axis=0)
                else:
                    current_info['Small'] = broken.tolist()
                    current_info['SmallTrue'].append(true_outliers.tolist())


            else:
                # more than 2 abnormally small blobs, find if they have pairs
                broken = np.where(tf)[0]
                broken_true = true_outliers
                Centroids = blob_centroids[broken, :]

                while len(broken) > 0:
                    a = Centroids[0]

                    x = Centroids[:, 0]
                    y = Centroids[:, 1]

                    distance_x = x - a[0]
                    distance_y = y - a[1]
                    valid_indices = np.where(np.abs(distance_x) < horizontal_threshold)[0]

                    if len(valid_indices) == 1:
                        current_info['Small'].append(broken[0])
                        current_info['SmallTrue'].append(np.where((all_props_centroids == a).all(axis=1))[0])
                        broken = np.delete(broken, 0)
                        Centroids = np.delete(Centroids, 0, axis=0)
                    else:
                        distances = np.sqrt(distance_x[valid_indices] ** 2 + distance_y[valid_indices] ** 2)
                        min_index_in_valid = np.argsort(distances)[0:2]
                        min_index = valid_indices[min_index_in_valid[1]]

                        p1 = Centroids[min_index_in_valid[0]]
                        p2 = Centroids[min_index_in_valid[1]]
                        wherep1 = np.where((blob_centroids == p1).all(axis=1))[0]
                        wherep2 = np.where((blob_centroids == p2).all(axis=1))[0]
                        wherep1True = np.where((all_props_centroids == p1).all(axis=1))[0]
                        wherep2True = np.where((all_props_centroids == p2).all(axis=1))[0]

                        # check if they are in the same line, else do nothing
                        if np.abs(p1[0] - p2[0]) <= horizontal_threshold:
                            current_info['Broken'].append(broken[min_index_in_valid])
                            current_info['BrokenTrue'].append(
                                list(np.concatenate([wherep1True, wherep2True], dtype=np.int64)))
                            new_centroid = (p1 + p2) / 2
                            blob_centroids[wherep1] = new_centroid
                            blob_centroids = np.delete(blob_centroids, wherep2, axis=0)
                        else:
                            current_info['Small'].append(broken[min_index_in_valid])
                            current_info['SmallTrue'].append(
                                list(np.concatenate([wherep1True, wherep2True], dtype=np.int64)))

                        broken = np.delete(broken, min_index_in_valid)
                        Centroids = np.delete(Centroids, min_index_in_valid, axis=0)

            current_info['Centroids'] = blob_centroids.tolist()

        all_info.append(current_info)

    return all_info


# edit user settings here
# folder_path = r'D:/tranbajos/Edinburgh/Backup Edi June 8th/gel genie/probability_map_samples_v2/direct_model_outputs_test_set/214.2/unet_dec_21_extended_set'
folder_path = 'D:/tranbajos/Edinburgh/Backup Edi June 8th/gel genie/probability_map_samples/direct_model_outputs'
output_path = 'D:/tranbajos/Edinburgh/Backup Edi June 8th/gel genie/probability_map_samples/output_plots'
save_plots = True
verbose_output = False

for image_folder in os.listdir(folder_path):
    if not os.path.isdir(os.path.join(folder_path, image_folder)):  # non-folder files
        continue

    if save_plots:
        create_dir_if_empty(os.path.join(output_path, image_folder))

    binary_image = np.load(os.path.join(folder_path, image_folder, 'seg_mask_one_hot.npy'))
    binary_image = binary_image.argmax(axis=0)
    binary_image = binary_image.astype(np.uint8)

    labeled_image = label(binary_image, connectivity=2)

    # Colorize the labels for visualization
    colored_labels = label2rgb(labeled_image, bg_label=0)

    # Compute region properties for labeled blobs
    props = regionprops(labeled_image, binary_image)

    # Compute the median length of blobs
    horizontal_lengths = [prop.bbox[3] - prop.bbox[1] for prop in props]
    median_length_blobs = np.median(horizontal_lengths)

    # Compute the median height of blobs
    vertical_lengths = [prop.bbox[2] - prop.bbox[0] for prop in props]
    median_height_blobs = np.median(vertical_lengths)

    # Number of blobs
    number_of_blobs = len(props)

    # Compute boundaries of the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_blob_centroids = np.array([prop.centroid for prop in props])

    # Find nearest points and separate columns
    struct_nearest = find_nearest_points(all_blob_centroids, median_length_blobs)

    # Separate connected blobs into columns
    merged_groups = separate_into_columns(struct_nearest)

    # Merge bounding boxes of connected blobs in each column
    merged_bounding_boxes, merged_bounding_boxes_centroids = merge_bounding_boxes(merged_groups, props)

    # Further merge columns
    vertical_threshold = median_length_blobs / 3
    horizontal_threshold = median_height_blobs / 2

    # Sort centroids based on their x-coordinates
    sorted_centroids = sorted(merged_bounding_boxes_centroids, key=lambda x: x[1])
    sorted_inds = [i for i, _ in sorted(enumerate(merged_bounding_boxes_centroids), key=lambda x: x[1][1])]

    # Merge close centroids into groups
    merged_groups_fin, merged_boxes = merge_close_centroids_bb(merged_bounding_boxes, sorted_centroids, sorted_inds,
                                                               merged_groups, vertical_threshold)

    result = process_merged_bounding_boxes(merged_groups_fin, merged_boxes, props, horizontal_threshold)

    # Display information in a table
    if verbose_output:
        for info in result:
            print(f"Group: {info['Group']}")
            print(f"BoundingBox: {info['BoundingBox']}")
            print(f"Blobs: {info['Blobs']}")

            # print("Props:")
            # for prop in info['Props']:
            #     print(f"  - {prop}")

            # print(f"Centroids: {info['Centroids']}")
            print(f"BrokenTrue: {info['BrokenTrue']}")
            print(f"SmallTrue: {info['SmallTrue']}")

            print("\n" + "-" * 50 + "\n")  # Separation line

    # Display boundaries on the original image and blob centroids with index
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image with boundaries and blob centroids
    ax1.imshow(binary_image, cmap='gray')
    for contour in contours:
        ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=1.2)

    all_blob_centroids = np.array([prop.centroid for prop in props])
    for i, centroid in enumerate(all_blob_centroids):
        ax1.text(centroid[1] + 15, centroid[0], str(i), color='b', fontsize=8, ha='center', va='center')
        ax1.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=2)

    ax1.set_title('Original Image with Boundaries and Blob Centroids')

    # Plot final merged bounding boxes and amended centroids
    ax2.imshow(binary_image, cmap='gray')

    for box in merged_boxes:
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax2.add_patch(rect)

    for info in result:
        centroids = np.array(info['Centroids'])
        ax2.scatter(centroids[:, 1], centroids[:, 0], c='g', marker='+', label='Amended Centroids')

    ax2.set_title('Final Merged Bounding Boxes with Amended Centroids')
    plt.suptitle('Analysis STEP 1')
    plt.tight_layout()

    if save_plots:
        plt.savefig(os.path.join(output_path, image_folder, 'step1_merged_centroids.png'), dpi=300)
    if verbose_output:
        plt.show(block=False)

    plt.close()

    empty_image = np.zeros_like(binary_image)
    fixed_image = np.zeros_like(binary_image)

    # Iterate through each group in the result
    for info in result:
        # Extract the indices from the 'BrokenTrue' list
        broken_indices = info['BrokenTrue']

        # Iterate through each set of broken indices in the group
        for indices_set in broken_indices:
            # Iterate through each index in the set and paint the corresponding binary blob
            for index in indices_set:
                # Extract the binary blob for the current index
                current_blob = labeled_image == index + 1  # Adding 1 because label starts from 1

                # Paint the blob on the empty image
                empty_image[current_blob] = 255  # Assuming binary values (0 and 255)

    # Display the result
    plt.figure()
    plt.imshow(empty_image, cmap='gray')
    plt.title('Empty Image with broken Blobs (STEP 2)')

    if save_plots:
        plt.savefig(os.path.join(output_path, image_folder, 'step2_broken_blobs.png'), dpi=300)

    if verbose_output:
        plt.show(block=False)
    plt.close()

    empty_image = np.zeros_like(binary_image)

    # Create a copy of the original binary_image to store the fixed_image
    fixed_image = np.copy(binary_image)

    # Iterate through each group in the result
    for info in result:
        # Extract the indices from the 'BrokenTrue' list
        broken_indices = info['BrokenTrue']

        # Iterate through each set of broken indices in the group
        for indices_set in broken_indices:
            # Extract the binary blobs for the current indices_set
            blobs = [labeled_image == index + 1 for index in indices_set]

            # Create convex hull for the blobs
            convex_hull = convex_hull_image(np.logical_or(*blobs))

            # Paint the convex hull on the empty image
            empty_image[convex_hull] = 255  # Assuming binary values (0 and 255)

            # Paint the convex hull on the fixed_image
            fixed_image[convex_hull] = 255  # Assuming binary values (0 and 255)

    # Display the result for the empty image
    plt.figure()
    plt.imshow(empty_image, cmap='gray')
    plt.title('Empty Image with Painted Convex Hulls (STEP 3)')

    if save_plots:
        plt.savefig(os.path.join(output_path, image_folder, 'step3_convex_hulls.png'), dpi=300)

    if verbose_output:
        plt.show(block=False)
    plt.close()

    # Display the result for the fixed image
    plt.figure()
    plt.imshow(fixed_image, cmap='gray')

    # Plot rectangles for merged bounding boxes
    for box in merged_boxes:
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r',
                                 facecolor='none')
        plt.gca().add_patch(rect)
    plt.title('Fixed Image with Painted Convex Hulls (STEP 4)')
    if save_plots:
        plt.savefig(os.path.join(output_path, image_folder, 'step4_fixed_convex_hulls.png'), dpi=300)

    if verbose_output:
        plt.show(block=False)
    plt.close()

    final_labeled_image = label(fixed_image, connectivity=2)
    final_props = regionprops(final_labeled_image, fixed_image)
    final_all_blob_centroids = np.array([prop.centroid for prop in final_props])

    plt.figure()
    plt.imshow(fixed_image, cmap='gray')
    # Plot rectangles for merged bounding boxes
    for box in merged_boxes:
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r',
                                 facecolor='none')
        plt.gca().add_patch(rect)

    # Plot centroids
    for i, centroid in enumerate(final_all_blob_centroids):
        plt.text(centroid[1] + 15, centroid[0], str(i), color='b', fontsize=8, ha='center', va='center')
        plt.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=1)

    plt.title('Fixed Image with Painted Convex Hulls and New Centroids (STEP 5)')

    if save_plots:
        plt.savefig(os.path.join(output_path, image_folder, 'step5_convex_hulls_and_centroids.png'), dpi=300)

    if verbose_output:
        plt.show(block=False)
    plt.close()

    ## Fix the last broken bands:


    extra_fixed_image = fixed_image

    # Iterate through each group in the result
    for info in result:
        # Extract Bounding Box of each final group
        extra_empty_image = np.zeros_like(binary_image)
        bounding_box = info['BoundingBox']
        # Extract bounding box coordinates
        y_min, x_min, y_max, x_max = bounding_box
        # Set pixels within the bounding box to 1 (white)
        extra_empty_image[y_min:y_max, x_min:x_max] = 1
        masked_image = cv2.bitwise_and(fixed_image, fixed_image, mask=extra_empty_image)

        extra_final_labeled_image = label(masked_image, connectivity=2)
        extra_final_props = regionprops(extra_final_labeled_image, masked_image)

        imtest2 = np.zeros_like(binary_image)

        column_centroids = np.array([prop.centroid for prop in extra_final_props])

        y = column_centroids[:, 0]
        x = column_centroids[:, 1]

        # Check differences in vertical axes between centroids

        for k in range(column_centroids.shape[0]):
            current_ind = np.atleast_1d(k)
            A = column_centroids[k, :]
            distance_y = np.abs(y - A[0])
            valid_indices = np.where((distance_y > 0) & (distance_y < horizontal_threshold))[0]

            if len(valid_indices) > 0:
                indices_set = np.concatenate([current_ind, valid_indices])

                # Extract the binary blobs for the current indices_set
                blobs = [extra_final_labeled_image == index + 1 for index in indices_set]

                # Combine binary blobs to create a convex hull
                convex_hull = convex_hull_image(np.logical_or.reduce(blobs))

                # Paint the convex hull on the fixed_image
                extra_fixed_image[convex_hull] = 255  # Assuming binary values (0 and 255)


    extra_fixed_image = np.logical_or(fixed_image, extra_fixed_image).astype(np.uint8) * 255

    plt.figure()
    plt.imshow(extra_fixed_image, cmap='gray')
    plt.show(block=False)



    extra_final_labeled_image = label(extra_fixed_image, connectivity=2)
    extra_final_props = regionprops(extra_final_labeled_image, fixed_image)
    extra_final_all_blob_centroids = np.array([prop.centroid for prop in extra_final_props])


    plt.figure()
    plt.imshow(extra_fixed_image, cmap='gray')
    plt.show(block=False)


    # Plot rectangles for merged bounding boxes
    for box in merged_boxes:
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    # Plot centroids
    for i, centroid in enumerate(extra_final_all_blob_centroids):
        plt.text(centroid[1] + 15, centroid[0], str(i), color='b', fontsize=8, ha='center', va='center')
        plt.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=1)

    plt.title('Extra Fixed Image with Painted Convex Hulls and New Centroids')
    if save_plots:
        plt.savefig(os.path.join(output_path, image_folder, 'step6_extra_convex_hulls_and_centroids.png'), dpi=300)

    if verbose_output:
        plt.show(block=False)
    plt.close()

    print('Hello')









