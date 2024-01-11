import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from tabulate import tabulate

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
            continue # Skip empty groups

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
        if abs(sorted_centroids[i][1] - sorted_centroids[i-1][1]) <= vertical_threshold:
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

            merged_groups_fin.append({'mergedBoxes': [min_row, min_col, max_row, max_col], 'indices': merged_groups_inds})

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

    for i in range(len(merged_boxes)):
        current_info = {'Group': i, 'BoundingBox': merged_boxes[i], 'Blobs': merged_groups_fin[i]['indices'],
                        'Props': [], 'Centroids': [], 'Broken': [], 'Small': [], 'SmallTrue': [], 'BrokenTrue': []}

        props_group = [props[j] for j in merged_groups_fin[i]['indices']]
        current_info['Props'] = props_group
        blobsInds = merged_groups_fin[i]['indices']

        # Extract the horizontal length of each original bounding box
        horizontal_lengths = [prop.bbox[3]-prop.bbox[1] for prop in props_group]

        # Find the outliers
        tf = is_outlier(horizontal_lengths)
        blob_centroids = np.array([prop.centroid for prop in props_group])
        current_info['Centroids'] = blob_centroids

        if np.sum(tf) > 0:
            if np.sum(tf) == 1:
                # only one abnormally small blob - count it as a normal blob, do
                # nothing but save the index
                broken = np.where(tf)[0]
                current_info['Small'] = broken.tolist()
                current_info['SmallTrue'] = [blobsInds [index] for index in broken[min_index_in_valid]]

            elif np.sum(tf) == 2:
                broken = np.where(tf)[0]
                
                wherep1 = blobsInds[broken[0]]
                wherep2 = blobsInds[broken[1]]
                p1 = blob_centroids[broken[0]]
                p2 = blob_centroids[broken[1]]

                # check if they are in the same line, else do nothing
                if np.abs(p1[0] - p2[0]) <= horizontal_threshold:
                    current_info['Broken'] = broken.tolist()
                    current_info['BrokenTrue'].append(np.array([wherep1, wherep2]))
                    new_centroid = (p1 + p2) / 2
                    blob_centroids[broken[0]] = new_centroid
                    blob_centroids = np.delete(blob_centroids, broken[1], axis=0)
                else:
                    current_info['Small'] = broken.tolist()
                    current_info['SmallTrue'].append(np.array([wherep1, wherep2]))
                    

            else:
                # more than 2 abnormally small blobs, find if they have pairs
                broken = np.where(tf)[0]
                Centroids = blob_centroids[broken,:]
                

                while len(broken) > 0:
                    a = Centroids[0]

                    x = Centroids[:, 0]
                    y = Centroids[:, 1]

                    distance_x = x - a[0]
                    distance_y = y - a[1]
                    valid_indices = np.where(np.abs(distance_x) < horizontal_threshold)[0]

                    if len(valid_indices) == 1:
                        current_info['Small'].append(broken[0])
                        current_info['SmallTrue'].append(np.where((blob_centroids == a).all(axis=1))[0])
                        broken = np.delete(broken, 0)
                        Centroids = np.delete(Centroids, 0, axis=0)
                    else:
                        distances = np.sqrt(distance_x[valid_indices]**2 + distance_y[valid_indices]**2)
                        min_index_in_valid = np.argsort(distances)[0:2]
                        min_index = valid_indices[min_index_in_valid[1]]
                        current_info['Broken'].append(broken[min_index_in_valid])
        
                        current_info['BrokenTrue'].append(blobsInds [index] for index in broken[min_index_in_valid])

                        p1 = Centroids[min_index_in_valid[0]]
                        p2 = Centroids[min_index_in_valid[1]]
                        wherep1 = np.where((blob_centroids == p1).all(axis=1))[0]
                        wherep2 = np.where((blob_centroids == p2).all(axis=1))[0]
                        current_info['BrokenTrue'].append(np.concatenate([wherep1, wherep2], dtype=np.int64))



                        # check if they are in the same line, else do nothing
                        if np.abs(p1[0] - p2[0]) <= horizontal_threshold:
                            new_centroid = (p1 + p2) / 2

                            blob_centroids[wherep1] = new_centroid
                            blob_centroids = np.delete(blob_centroids, wherep2, axis=0)
                            

                        broken = np.delete(broken, min_index_in_valid)
                        Centroids = np.delete(Centroids, min_index_in_valid, axis=0)

            current_info['Centroids'] = blob_centroids.tolist()

        all_info.append(current_info)

    return all_info


# Specify the folder path
folder_path = r'D:/tranbajos/Edinburgh/Backup Edi June 8th/gel genie/probability_map_samples/direct_model_outputs/6.2'

# Read the true mask image
image_path = os.path.join(folder_path, 'true_mask.png')
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)[1]

# Label connected components in the binary image
labeled_image = label(binary_image, connectivity=2)

# Colorize the labels for visualization
colored_labels = label2rgb(labeled_image, bg_label=0)

# Compute region properties for labeled blobs
props = regionprops(labeled_image, binary_image)

# Compute the median length of blobs
horizontal_lengths = [prop.bbox[3]-prop.bbox[1] for prop in props]
median_length_blobs = np.median(horizontal_lengths)

# Compute the median height of blobs
vertical_lengths = [prop.bbox[2]-prop.bbox[0] for prop in props]
median_height_blobs = np.median(vertical_lengths)

# Number of blobs
number_of_blobs = len(props)

# Compute boundaries of the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
all_blob_centroids = np.array([prop.centroid for prop in props])
# # Display boundaries on the original image
# plt.figure()
# plt.imshow(binary_image, cmap='gray')

# for contour in contours:
#     plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)

# # Display blob centroids with index
 
# for i, centroid in enumerate(all_blob_centroids):
#     plt.text(centroid[1] + 15, centroid[0], str(i), color='b', fontsize=8, ha='center', va='center')
#     plt.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=2)

# plt.show()


# Find nearest points and separate columns
struct_nearest = find_nearest_points(all_blob_centroids, median_length_blobs)

# Separate connected blobs into columns
merged_groups = separate_into_columns(struct_nearest)

# Merge bounding boxes of connected blobs in each column
merged_bounding_boxes, merged_bounding_boxes_centroids = merge_bounding_boxes(merged_groups, props)

# # Plot the original binary image
# plt.figure()
# plt.imshow(binary_image, cmap='gray')

# # Plot rectangles for merged bounding boxes
# for box in merged_bounding_boxes:
#     rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r', facecolor='none')
#     plt.gca().add_patch(rect)

# # Plot centroids
# centroids = np.array(merged_bounding_boxes_centroids).T
# plt.scatter(centroids[1], centroids[0], c='b', marker='x', label='Centroids')

# plt.title('Merged Bounding Boxes and Centroids')
# plt.legend()
# plt.show()

# Further merge columns
vertical_threshold = median_length_blobs / 5
horizontal_threshold = median_height_blobs / 2

# Sort centroids based on their x-coordinates
sorted_centroids = sorted(merged_bounding_boxes_centroids, key=lambda x: x[1])
sorted_inds = [i for i, _ in sorted(enumerate(merged_bounding_boxes_centroids), key=lambda x: x[1][1])]


# Merge close centroids into groups
merged_groups_fin, merged_boxes = merge_close_centroids_bb(merged_bounding_boxes, sorted_centroids, sorted_inds, merged_groups, vertical_threshold)


# plt.figure()
# plt.imshow(binary_image, cmap='gray')

# # Plot rectangles for merged bounding boxes
# for box in merged_boxes:
#     rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r', facecolor='none')
#     plt.gca().add_patch(rect)


# plt.title('Final Merged Bounding Boxes')

# plt.show()


result = process_merged_bounding_boxes(merged_groups_fin, merged_boxes, props, horizontal_threshold)


# Display information in a table
for info in result:
    print(f"Group: {info['Group']}")
    print(f"BoundingBox: {info['BoundingBox']}")
    print(f"Blobs: {info['Blobs']}")
    
    # print("Props:")
    # for prop in info['Props']:
    #     print(f"  - {prop}")

    print(f"Centroids: {info['Centroids']}")
    print(f"Broken: {info['Broken']}")
    print(f"Small: {info['Small']}")
    print("\n" + "-" * 50 + "\n")  # Separation line

# Display boundaries on the original image and blob centroids with index
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image with boundaries and blob centroids
ax1.imshow(binary_image, cmap='gray')
for contour in contours:
    ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)

all_blob_centroids = np.array([prop.centroid for prop in props])
for i, centroid in enumerate(all_blob_centroids):
    ax1.text(centroid[1] + 15, centroid[0], str(i), color='b', fontsize=8, ha='center', va='center')
    ax1.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=2)

ax1.set_title('Original Image with Boundaries and Blob Centroids')

# Plot final merged bounding boxes and amended centroids
ax2.imshow(binary_image, cmap='gray')

for box in merged_boxes:
    rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

for info in result:
    centroids = np.array(info['Centroids'])
    ax2.scatter(centroids[:, 1], centroids[:, 0], c='g', marker='+', label='Amended Centroids')

ax2.set_title('Final Merged Bounding Boxes with Amended Centroids')

plt.show()

print('Hello world')