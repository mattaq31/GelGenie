import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
import wandb
import os
from pathlib import Path
import logging
import numpy as np

from segmentation.helper_functions.dice_score import dice_loss
from ..helper_functions.stat_functions import excel_stats
from ..helper_functions.display_functions import plot_stats, show_segmentation
from segmentation.evaluation.basic_eval import evaluate
from .training_setup import define_optimizer, define_scheduler
from ..data_handling import prep_dataloader


def train_net(net, device, base_hardware='PC', model_name='milesial-UNet', epochs=5, batch_size=8, learning_rate=1e-5,
              val_percent=0.1, save_checkpoint=True, img_scale=0.5, amp=False, dir_train_img=None,
              dir_train_mask=None, split_training_dataset=False, dir_val_img=None, dir_val_mask=None,
              dir_checkpoint=None, num_workers=1, segmentation_path='', base_dir='', n_channels=1,
              optimizer_type='adam', scheduler_used='false', load=False, loss_fn='both',
              apply_augmentations=False, padding=False):
    """
    TODO: documentation needs updating here.
    TODO: if not using val_percent anymore, this needs to be removed.
    Ideally, you should allow the user to either select a specific
    validation dataset or specify a percentage of the training set to be converted to the validation set

    :param padding:
    :param dir_train_img:
    :param n_channels:
    :param dir_val_img:
    :param dir_val_mask:
    :param model_name:
    :param base_hardware:
    :param dir_train_mask:
    :param scheduler_used:
    :param loss_fn:
    :param apply_augmentations:
    :param net: UNet Model
    :param device: Device being used (cpu/ cuda)
    :param epochs: Number of epochs
    :param batch_size: Batch size for loading data
    :param learning_rate: Learning rate
    :param val_percent: Validation set partition percentage
    :param save_checkpoint: Whether checkpoints are saved
    :param img_scale: Downscaling factor of the images (between 0 and 1)
    :param amp: Use mixed precision
    :param dir_checkpoint: Directory where checkpoints are stored
    :param num_workers: Number of workers for dataloader
    :param segmentation_path: Directory to save segmentation images
    :param base_dir: Path of base directory
    :param optimizer_type: Type of optimizer to use
    :param load: False if nothing is loaded, dictionary of state_dicts if loaded
    :return: None
    """

    train_loader, val_loader, n_train, n_val = \
        prep_dataloader(dir_train_img, dir_train_mask, split_training_dataset, dir_val_img, dir_val_mask,
                        n_channels, img_scale, val_percent, batch_size, num_workers,
                        apply_augmentations, padding)

    # (Initialize logging)
    if base_hardware == 'EDDIE':
        experiment = wandb.init(project='U-Net', entity='dunn-group', resume='allow',
                                name=os.path.basename(base_dir),
                                settings=wandb.Settings(start_method="fork"))
    else:
        experiment = wandb.init(project='U-Net', entity='dunn-group',
                                name=os.path.basename(base_dir),
                                resume='allow')

    experiment.config.update(
        dict(model_name=model_name, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
             amp=amp, dir_train_img=dir_train_img, dir_train_mask=dir_train_mask,
             split_training_dataset=split_training_dataset,
             dir_val_img=dir_val_img, dir_val_mask=dir_val_mask,
             dir_checkpoint=dir_checkpoint, num_workers=num_workers, segmentation_path=segmentation_path,
             base_dir=base_dir, n_channels=n_channels,
             optimizer_type=optimizer_type, scheduler_used=scheduler_used, load=load, loss_fn=loss_fn,
             apply_augmentations=apply_augmentations, padding=padding))

    print("Starting training:\n",
          f"Epochs:          {epochs}\n",
          f"Batch size:      {batch_size}\n",
          f"Learning rate:   {learning_rate}\n",
          f"Training size:   {n_train}\n",
          f"Validation size: {n_val}\n",
          f"Checkpoints:     {save_checkpoint}\n",
          f"Device:          {device.type}\n",
          f"Images scaling:  {img_scale}\n",
          f"Mixed Precision: {amp}\n",
          f"Optimizer Type:  {optimizer_type}\n",
          f"Scheduler Used:  {scheduler_used}")

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optimizer_type == 'rmsprop':
        optimizer = define_optimizer(net.parameters(), lr=learning_rate, optimizer_type='rmsprop',
                                     optimizer_params={'weight_decay': 1e-8, 'momentum': 0.9, 'alpha': 0.99})
    elif optimizer_type == 'adam':
        optimizer = define_optimizer(net.parameters(), lr=learning_rate, optimizer_type='adam')

    # Load will either be False or a dict of state_dicts
    if load:
        optimizer.load_state_dict(load['optimizer'])

    if scheduler_used == 'ReduceLROnPlateau' or scheduler_used == 'CosineAnnealingWarmRestarts':  # Scheduler will be used
        scheduler = define_scheduler(optimizer, scheduler_type=scheduler_used)  # goal: maximize Dice score
        # TODO: optimizer and scheduler need to be reloaded from previous checkpoint if continuing training.

        if load:
            scheduler.load_state_dict(load['scheduler'])

    criterion = nn.CrossEntropyLoss()
    global_step = 0

    train_loss_log = []
    val_loss_log = []

    columns = ['Epoch', 'Image', 'Mask Prediction', 'Separated bands',
               'Super-imposed mask prediction', 'True Mask']
    # table_list = []

    # Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    f'the images are loaded correctly, images.shape is  {images.shape}'

                images = images.to(device=device)  # TODO: why is there this torch.long dtype here?
                true_masks = true_masks.to(device=device, dtype=torch.long)

                masks_pred = net(images)

                if loss_fn == 'both':
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                elif loss_fn == 'CrossEntropy':
                    loss = criterion(masks_pred, true_masks)
                elif loss_fn == 'Dice':
                    loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                     F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                     multiclass=True)

                optimizer.zero_grad()  # this ensures that all weight gradients are zeroed before moving on to the next set of gradients
                loss.backward()  # this calculates the gradient for all weights (backpropagation)
                optimizer.step()  # here, the optimizer will calculate and make the change necessary for each weight based on its defined rules

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({  # TODO: are all these items necessary at this point?
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                continue

            # Evaluation round
            histograms = {}  # TODO: look at these results in Wandb

            val_score, show_image, show_mask_pred, show_mask_true = evaluate(net, val_loader, device)

            val_loss_log.append(val_score.item())  # Append dice score

            if scheduler_used:
                scheduler.step(val_score)

            image_array, threshold_mask_array, labelled_bands, combi_mask_array, mask_true_array = \
                show_segmentation(show_image.squeeze(), show_mask_pred.squeeze(), show_mask_true.squeeze(),
                                  epoch, dice_score=val_loss_log[-1], segmentation_path=segmentation_path,
                                  n_channels=n_channels)

            # table_list.append([f'Epoch {epoch}/{epochs}', wandb.Image(image_array, caption=f'Epoch {epoch}'),
            #                    wandb.Image(threshold_mask_array), wandb.Image(labelled_bands),
            #                    wandb.Image(combi_mask_array), wandb.Image(mask_true_array)])

            # table = wandb.Table(data=table_list, columns=columns)
            # TODO: This table is difficult to view on wandb - fix or remove

            # Creating arrays of training set to log into wandb
            if n_channels == 1:
                height = images[0].squeeze().size(dim=0)
                width = images[0].squeeze().size(dim=1)
                show_train_image_array = images[0].detach().squeeze().cpu().numpy()
            elif n_channels == 3:
                height = images[0].squeeze().size(dim=1)
                width = images[0].squeeze().size(dim=2)
                show_train_image_array = np.transpose(images[0].detach().squeeze().cpu().numpy(), (1, 2, 0))  # np array [H, W, C]

            train_pred_array = masks_pred[0].detach().squeeze().cpu().numpy()  # Copies prediction into np array [C, H, W]
            show_train_pred_thresholded_array = np.zeros((height, width), dtype=int)  # [H,W]
            for h in range(height):
                for w in range(width):
                    # The threshold is set to 0.8
                    if train_pred_array[0][h][w] < 0.2 and train_pred_array[1][h][w] > 0.8:
                        show_train_pred_thresholded_array[h][w] = 1

            show_train_mask_array = true_masks[0].detach().squeeze().cpu().numpy()  # Copies true mask[1] into np array [H,W]

            # Logging onto wandb
            logging.info('Validation Dice score: {}'.format(val_score))
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'train': {
                    'images': wandb.Image(show_train_image_array),
                    'masks': {
                        'true': wandb.Image(show_train_mask_array),
                        'pred': wandb.Image(show_train_pred_thresholded_array),
                    },
                },
                # 'train': {
                #     'images': wandb.Image(np.transpose(images[0].detach().squeeze().cpu().numpy(), (1, 2, 0))),
                #     'masks': {
                #         'true': wandb.Image(true_masks[0].float().cpu()),
                #         'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                #     },
                # },
                'val': {
                    'images': wandb.Image(image_array),
                    'masks': {
                        'true': wandb.Image(mask_true_array),
                        'pred': wandb.Image(threshold_mask_array, caption='Threshold = 0.8'),
                        'pred-superimposed': wandb.Image(combi_mask_array),
                        'labelled-bands': wandb.Image(labelled_bands),
                    },
                },
                # 'val': {
                #     'images': wandb.Image(show_image.cpu()),
                #     'masks': {
                #         'true': wandb.Image(show_mask_true.cpu()),
                #         'pred': wandb.Image(show_mask_pred.cpu()),
                #     },
                # },
                # 'table': table,
                'step': global_step,
                'epoch': epoch,
                **histograms  # TODO: remove this histogram
            })

            # All batches in the epoch iterated through, append loss values as string type
            train_loss_log.append(epoch_loss)

            plot_stats(train_loss_log, val_loss_log, base_dir)

            excel_stats(train_loss_log, val_loss_log, base_dir)

        if save_checkpoint and (epoch == epochs or epoch % 10 == 0):  # Only save checkpoints every 10 epochs
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            save_dict = {'network': net.state_dict(),
                         'optimizer': optimizer.state_dict()}
            if scheduler_used:
                save_dict['scheduler'] = scheduler.state_dict()

            torch.save(save_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
