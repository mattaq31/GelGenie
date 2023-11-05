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
