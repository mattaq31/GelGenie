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

import rich_click as click
import sys


@click.command()
@click.option('--model_folder', required=True, help='model directory for conversion.')
@click.option('--epoch', '-e', required=True, help='Epoch checkpoint to convert.')
@click.option('--export_type', '-et', required=True, multiple=True,
              help='Request either torchscript, onnx or both exports.')
@click.option('--visual_onnx', is_flag=True,
              help='Set this flag to generate a sample image using onnx and opencv for comparison purposes.'
                   '  Also requires an input image directory')
@click.option('--image_directory',
              help='Image directory with samples images that can be tested using exported model.')
def export_model(model_folder, epoch, export_type, visual_onnx, image_directory):
    from gelgenie.segmentation.helper_functions.onnx_converter import simple_onnx_export, visual_onnx_export
    from gelgenie.segmentation.helper_functions.torchscript_converter import torchscript_export

    if 'onnx' in export_type:
        print('Converting to onnx format...')
        if visual_onnx:
            visual_onnx_export(model_folder, epoch, image_directory)
        else:
            simple_onnx_export(model_folder, epoch)

    if 'torchscript' in export_type:
        print('Converting to torchscript format...')
        torchscript_export(model_folder, epoch)


if __name__ == '__main__':
    export_model(sys.argv[1:])  # for use when debugging with pycharm
