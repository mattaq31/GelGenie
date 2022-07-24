import os


def create_dir_if_empty(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
