""" Setup script for package. """
from setuptools import setup, find_packages

setup(
    name="GelGenie",
    author="FB, MA, KD",
    description="TBD",
    version="1.0.0",
    url="https://github.com/mattaq31/Automatic-Gel-Analysis/",
    packages=find_packages(),
    entry_points='''
    [console_scripts]
    gelseg_train=gelgenie.segmentation.routine_training:segmentation_network_trainer
    pull_model=gelgenie.segmentation.helper_functions.general_functions:pull_server_data
    gen_eddie_qsub=gelgenie.segmentation.routine_training:generate_eddie_batch_file
    quick_seg=gelgenie.segmentation.evaluation:segmentation_pipeline
    seg_data_interpret=gelgenie.segmentation.evaluation:segmentation_results_compare
''',
)

