import rich_click as click
import sys


def model_eval_load(exp_folder, eval_epoch):
    import toml
    import torch
    from os.path import join
    from gelgenie.segmentation.networks import model_configure
    from gelgenie.segmentation.helper_functions.stat_functions import load_statistics

    model_config = toml.load(join(exp_folder, 'config.toml'))['model']
    model, _, _ = model_configure(**model_config)
    if eval_epoch == 'best':
        stats = load_statistics(join(exp_folder, 'training_logs'), 'training_stats.csv', config='pd')
        sel_epoch = stats['Epoch'][stats['Dice Score'].idxmax()]
    else:
        sel_epoch = eval_epoch

    checkpoint = torch.load(f=join(exp_folder, 'checkpoints', 'checkpoint_epoch_%s.pth' % sel_epoch),
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['network'])
    model.eval()

    return model



@click.command()
@click.option('--model_and_epoch', '-me', multiple=True,
              help='Experiments and epochs to evaluate.', type=(str, str))
@click.option('--model_folder', '-p', default=None,
              help='Path to folder containing model config.')
@click.option('--input_folder', '-i', default=None,
              help='Path to folder containing input images.')
@click.option('--output_folder', '-o', default=None,
              help='Path to folder containing output images.')
@click.option('--multi_augment', is_flag=True,
              help='Set this flag to run test-time augmentation on input images.')
@click.option('--run_quant_analysis', is_flag=True,
              help='Set this flag to run quantitative analysis comparing output images with target masks.')
@click.option('--mask_folder', default=None,
              help='Path to ground truth mask data corresponding to input images.')
def segmentation_pipeline(model_and_epoch, model_folder, input_folder, output_folder, multi_augment,
                          run_quant_analysis, mask_folder):

    from os.path import join
    from gelgenie.segmentation.evaluation.core_functions import segment_and_plot, segment_and_quantitate
    from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty

    experiment_names, eval_epochs = zip(*model_and_epoch)

    models = []

    for experiment, eval_epoch in zip(experiment_names, eval_epochs):
        exp_folder = join(model_folder, experiment)
        model = model_eval_load(exp_folder, eval_epoch)
        models.append(model)

    create_dir_if_empty(output_folder)

    if run_quant_analysis:
        segment_and_quantitate(models, experiment_names, input_folder, mask_folder, output_folder, multi_augment=multi_augment)
    else:
        segment_and_plot(models, experiment_names, input_folder, output_folder, multi_augment=multi_augment)


if __name__ == '__main__':
    segmentation_pipeline(sys.argv[1:])  # for use when debugging with pycharm
