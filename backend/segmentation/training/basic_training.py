import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from segmentation.unet_utils.data_loading import BasicDataset
from segmentation.unet_utils.dice_score import dice_loss
from segmentation.unet_utils.utils import plot_stats
from segmentation.evaluation.basic_eval import evaluate_epoch, evaluate


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
              base_dir=''):
    """
    TODO: fill in
    :param net:
    :param device:
    :param epochs:
    :param batch_size:
    :param learning_rate:
    :param val_percent:
    :param save_checkpoint:
    :param img_scale:
    :param amp:
    :param dir_img:
    :param dir_mask:
    :param dir_checkpoint:
    :param num_workers:
    :param segmentation_path:
    :param base_dir:
    :return:
    """

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=1, pin_memory=True)

    # (Initialize logging)  TODO: we have to make a decision - either we use wandb or completely discard it
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    print("Starting training:\n",
          f"Epochs:          {epochs}\n",
          f"Batch size:      {batch_size}\n",
          f"Learning rate:   {learning_rate}\n",
          f"Training size:   {n_train}\n",
          f"Validation size: {n_val}\n",
          f"Checkpoints:     {save_checkpoint}\n",
          f"Device:          {device.type}\n",
          f"Images scaling:  {img_scale}\n",
          f"Mixed Precision: {amp}")

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

    if device != torch.device('cpu'):
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # TODO: investigate - is this necessary?  If it is, can we implement something for CPU?
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    train_loss_log = []
    val_loss_log = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                # print("loaded batch of images and masks")

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                if device != torch.device('cpu'):  # TODO: either find fix, or reduce amount of code here
                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)

                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                else:
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)


                optimizer.zero_grad(set_to_none=True)
                # TODO: what to do with the grad scaler and backward step in case of no CUDA being available?
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

            # All batches in the epoch iterated through, append loss values as string type
            train_loss_log.append(epoch_loss)
            # TODO: this function is repeating all the work the previous one did.  I would integrate both together.
            val_loss_log.append(evaluate_epoch(net, val_loader, device, epoch, segmentation_path).item())

            plot_stats(train_loss_log, val_loss_log, base_dir)

            loss_array = np.array([train_loss_log, val_loss_log]).T
            loss_dataframe = pd.DataFrame(loss_array, columns=['Training Set', 'Validation Set'])
            loss_dataframe.index.names = ['Epoch']
            loss_dataframe.index += 1
            loss_dataframe.to_csv(Path(base_dir + '/loss.csv'))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        print("another epoch done!")
