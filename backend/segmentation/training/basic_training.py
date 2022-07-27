import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm

import wandb

from pathlib import Path
import logging


from segmentation.unet_utils.dice_score import dice_loss
from segmentation.unet_utils.utils import plot_stats, show_segmentation, excel_stats
from segmentation.evaluation.basic_eval import evaluate
from .training_setup import define_optimizer
from ..helper_functions.data_functions import prep_dataloader


def train_net(net,
              device,
              epochs=5,
              batch_size=8,
              learning_rate=1e-5,
              val_percent=0.1,
              save_checkpoint=True,
              img_scale=0.5,
              amp=False,
              dir_img=None,
              dir_mask=None,
              dir_checkpoint=None,
              num_workers=1,
              segmentation_path='',
              base_dir='',
              n_channels=1,
              optimizer_type='adam',
              scheduler_used=False):
    """
    TODO: fill in
    :param net: UNet Model
    :param device: Device being used (cpu/ cuda)
    :param epochs: Number of epochs
    :param batch_size: Batch size for loading data
    :param learning_rate: Learning rate
    :param val_percent: Validation set partition percentage
    :param save_checkpoint: Whether checkpoints are saved
    :param img_scale: Downscaling factor of the images (between 0 and 1)
    :param amp: Use mixed precision
    :param dir_img: Directory of original image
    :param dir_mask: Directory of true masks
    :param dir_checkpoint: Directory where checkpoints are stored
    :param num_workers: Number of workers for dataloader
    :param segmentation_path: Directory to save segmentation images
    :param base_dir: Path of base directory
    :param optimizer_type: Type of optimizer to use
    :return: None
    """

    train_loader, val_loader, n_train, n_val = \
        prep_dataloader(dir_img, dir_mask, n_channels, img_scale, val_percent, batch_size, num_workers)



    # (Initialize logging)  TODO: we have to make a decision - either we use wandb or completely discard it
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment = wandb.init(project='U-Net', entity='dunn-group', resume='allow')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp, n_channels=n_channels, optimizer_type=optimizer_type,
                                  scheduler_used=scheduler_used, base_dir=base_dir))

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

    if scheduler_used:  # Scheduler will be used
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    else:  # No scheduler will be used
        pass
    # TODO: I think we should try and remove this scheduler for now, then add later if necessary.
    #  Add a system to allow user to select whether to turn scheduler on/off

    criterion = nn.CrossEntropyLoss()
    global_step = 0

    train_loss_log = []
    val_loss_log = []

    table = wandb.Table(columns=['Image', 'Mask Prediction', 'Separated bands',
                                 'Super-imposed mask prediction', 'True Mask'])

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    f'the images are loaded correctly, images.shape is  {images.shape}'

                images = images.to(device=device)  # TODO: why is there this torch.long dtype here?
                true_masks = true_masks.to(device=device, dtype=torch.long)

                masks_pred = net(images)

                loss = criterion(masks_pred, true_masks) \
                       + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                   F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                   multiclass=True)

                optimizer.zero_grad()  # this ensures that all weight gradients are zeroed before moving on to the next set of gradients
                loss.backward()  # this calculates the gradient for all weights (backpropagation)
                optimizer.step()  # here, the optimizer will calculate and make the change necessary for each weight based on its defined rules

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # break  # TODO: delete


            # Evaluation round
            histograms = {}  # TODO: look at these results in Wandb
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score, show_image, show_mask_pred, show_mask_true = evaluate(net, val_loader, device)

            val_loss_log.append(val_score.item())  # Append dice score

            if scheduler_used:
                scheduler.step(val_score)  # TODO: also very important - should this be done once every epoch or every batch?

            image_array, threshold_mask_array, labelled_bands, combi_mask_array, mask_true_array = \
                show_segmentation(show_image.squeeze(), show_mask_pred.squeeze(), show_mask_true.squeeze(),
                              epoch, dice_score=val_loss_log[-1], segmentation_path=segmentation_path,
                              n_channels=n_channels)

            table.add_data(wandb.Image(image_array, caption=f'Epoch {epoch}'), wandb.Image(threshold_mask_array),
                           wandb.Image(labelled_bands), wandb.Image(combi_mask_array), wandb.Image(mask_true_array))

            # Logging onto wandb
            logging.info('Validation Dice score: {}'.format(val_score))  # TODO: is this useful?
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'train':{
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                    },
                },
                'val': {
                    'images': wandb.Image(show_image.cpu()),
                    'masks': {
                        'true': wandb.Image(show_mask_true.cpu()),
                        'pred': wandb.Image(show_mask_pred.cpu()),
                    },
                },
                'table': table,
                'step': global_step,
                'epoch': epoch,
                **histograms
            })

            # All batches in the epoch iterated through, append loss values as string type
            train_loss_log.append(epoch_loss)

            plot_stats(train_loss_log, val_loss_log, base_dir)

            excel_stats(train_loss_log, val_loss_log, base_dir)

        if save_checkpoint and (epoch == epochs or epoch % 10 == 0):  # Only save checkpoints every 10 epochs
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

