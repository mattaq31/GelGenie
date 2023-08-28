import rich_click as click
import sys


@click.command()
@click.option('--datafolder', '-d', default=None,
              help='Folder to import/export data.')
@click.option('--datafile', '-f', default=None,
              help='Specific csv file to be analyzed.')
def segmentation_results_compare(datafolder, datafile):
    from gelgenie.segmentation.evaluation.reference_image_analysis import segmentation_accuracy_comparison
    segmentation_accuracy_comparison(datafolder, datafile)


@click.command()
@click.option('--model_and_epoch', '-me', multiple=True,
              help='Experiments and epochs to evaluate.', type=(str, str))
@click.option('--model_folder', '-p', default=None,
              help='Path to folder containing model config.')
@click.option('--input_folder', '-i', default=None,
              help='Path to folder containing input images.')
@click.option('--output_folder', '-o', default=None,
              help='Path to folder containing output images.')
@click.option('--run_ref_analysis', is_flag=True,
              help='Set this flag to run analysis on standard reference images.')
def segmentation_pipeline(model_and_epoch, model_folder, input_folder, output_folder, run_ref_analysis):

    import torch
    from os.path import join
    from gelgenie.segmentation.networks import model_configure
    from gelgenie.segmentation.evaluation.core_functions import segment_and_analyze
    from gelgenie.segmentation.evaluation.reference_image_analysis import standard_ladder_analysis
    import toml
    from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty

    experiment_names, eval_epochs = zip(*model_and_epoch)

    models = []

    for experiment, eval_epoch in zip(experiment_names, eval_epochs):
        exp_folder = join(model_folder, experiment)
        model_config = toml.load(join(exp_folder, 'config.toml'))['model']
        model, _, _ = model_configure(**model_config)
        checkpoint = torch.load(f=join(exp_folder, 'checkpoints', 'checkpoint_epoch_%s.pth' % eval_epoch),
                                map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['network'])
        model.eval()
        models.append(model)

    create_dir_if_empty(output_folder)

    if run_ref_analysis:  # TODO: broken, needs a rewrite
        raise NotImplementedError('This function needs to be rewritten.')
        # standard_ladder_analysis(model, output_folder)
    else:
        segment_and_analyze(models, experiment_names, input_folder, output_folder)


if __name__ == '__main__':
    segmentation_pipeline(sys.argv[1:])  # for use when debugging with pycharm
