"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from setuptools import setup, find_packages

setup(
    name="GelGenie",
    author="Matthew Aquilina, Kiros Kwan, Filip Buŝić",
    description="Set of tools for training and using deep learning models to carry out "
                "gel electrophoresis band segmentation using PyTorch.",
    version="1.0.0",
    url="https://github.com/mattaq31/GelGenie/",
    packages=find_packages(),
    entry_points='''
    [console_scripts]
    gelseg_train=gelgenie.segmentation.routine_training:segmentation_network_trainer
    pull_model=gelgenie.segmentation.helper_functions.general_functions:pull_server_data
    gen_eddie_qsub=gelgenie.segmentation.routine_training:generate_eddie_batch_file
    quick_seg=gelgenie.segmentation.evaluation:segmentation_pipeline
    export_model=gelgenie.segmentation.helper_functions:export_model
''',
)

