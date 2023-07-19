import glob
import os
import re
from rich.table import Table


def create_summary_table(title, columns, col_colors, data):
    """
    Creates a rich summary table.
    :param title: Table title (string)
    :param columns: Column names (list)
    :param col_colors: Colour of text within each column (list)
    :param data: Data to plot in each row.  Must be a list of tuples, where each tuple is a row
    :return: Formatted rich table
    """
    table = Table(title=title)

    for col, colour in zip(columns, col_colors):
        table.add_column(col, style=colour, no_wrap=True, justify='center')

    for param, val in data:
        table.add_row(param, str(val))
    return table


def create_dir_if_empty(*directories):
    """
    Creates a directory if it doesn't exist.
    :param directories: Single filepath or list of filepaths.
    :return: None
    """
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
