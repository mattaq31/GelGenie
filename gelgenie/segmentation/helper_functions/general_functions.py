import glob
import os
import re


def create_dir_if_empty(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def extract_image_names_from_folder(folder, sorted=True, recursive=False):
    filenames = []
    for extension in ['*.jpg', '*.png', '*.bmp', '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.jpeg', '*.JPEG', '*.JPG']:
        if recursive:
            glob_path = os.path.join(folder, '**', extension)
        else:
            glob_path = os.path.join(folder, extension)
        filenames.extend(glob.glob(glob_path, recursive=recursive))
    if sorted:
        # Sort file names in Natural Order so that numbers starting with 1s don't take priority
        filenames.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
    return filenames
