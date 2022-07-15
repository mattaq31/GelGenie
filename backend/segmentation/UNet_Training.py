# To be used in EDDIE


import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from segmentation.unet_utils.data_loading import BasicDataset
from segmentation.unet_utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from segmentation.unet import UNet

import click
import toml

import os
from time import strftime

import torchshow as ts

import numpy as np
import pandas as pd

#######################################################################################################################
# Path of base data, image directory, mask directory, and checkpoints
#######################################################################################################################

"""
"/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/",
                                             "Automatic-Gel-Analysis/backend/segmentation/configs/",
                                             "EDDIE_GPU_default.toml"
"""
@click.command()
@click.option('--parameters',       default=None,       help='[Path] location of TOML parameters file, '
                                                             'containing configs for this experiment')
@click.option('--server',           default=None,       help='[String] Where the program is run [EDDIE/PC]')
@click.option('--core',             default=None,       help='[String] Which processor is used [GPU/CPU]')
@click.option('--pe',               default=None,       help='[int] How many parallel environments (cores) needed')
@click.option('--memory',           default=None,       help='[int] Required memory per core in GBytes')
@click.option('--epochs',           default=None,       help='[int] Number of epochs desired')
@click.option('--num_workers',      default=None,       help='[int] How many workers for dataloader simultaneously ,'
                                                             '(parallel dataloader threads, speed up data processing)')
@click.option('--batch_size',       default=None,       help='[int] Batch size for dataloader')
@click.option('--lr',               default=None,       help='[float] Learning Rate')
@click.option('--validation',       default=None,       help='[int] % of the data that is used as validation (0-100)')
@click.option('--save_checkpoint',  default=None,       help='[Bool] Whether checkpoints are saved')
@click.option('--img_scale',        default=None,       help='[Float] Downscaling factor of the images')
@click.option('--amp',              default=None,       help='[Bool] Use mixed precision')
@click.option('--load',             default=None,       help='[Bool/Path] Load model from a .pth file')
@click.option('--classes',          default=None,       help='[int] Number of classes/probabilities per pixel')
@click.option('--bilinear',         default=None,       help='[Bool] Use bilinear upsampling')
def unet_train(parameters, **kwargs):

    params = setup(parameters, **kwargs)

    # Pre-trainined weights:
    # load = "/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/UNet_Training_With_Images/Pre-trained/unet_carvana_scale0.5_epoch2.pth"



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=params['classes'], bilinear=params['bilinear'])  # initializing random weights

    print(f'Network:\n'
          f'\t{net.n_channels} input channels\n'
          f'\t{net.n_classes} output channels (classes)\n'
          f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    load = params['load']
    if load:
        net.load_state_dict(torch.load(load, map_location=device))
        logging.info(f'Model loaded from {load}')

    net.to(device=device)

    train_net(net=net,
              epochs=int(params['epochs']),
              batch_size=int(params['batch_size']),
              learning_rate=float(params['lr']),
              device=device,
              img_scale=float(params['img_scale']),
              val_percent=int(params['validation']) / 100,
              amp=params['amp'],
              dir_img=params['dir_img'],
              dir_mask=params['dir_mask'],
              dir_checkpoint=params['dir_checkpoint'],
              num_workers=int(params['num_workers'],),
              segmentation_path=params['segmentation_path'],
              base_dir=params['base_dir'])


#######################################################################################################################
# Path of base data, image directory, mask directory, and checkpoints
#######################################################################################################################
def setup(parameters, **kwargs):

    # Getting manual input configurations (that are not default, ie. none)
    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
    # The default configuration if none is specified
    kwargs_default = {'parameters':        ("C:/2022_Summer_Intern/Automatic-Gel-Analysis/backend/segmentation/"
                                            "configs/PC_default.toml"),
                      'server':             "EDDIE",
                      'core':               "GPU",
                      'pe':                 1,
                      'memory':             64,
                      'epochs':             10,
                      'num_workers':        1,
                      'batch_size':         4,
                      'lr':                 1e-5,
                      'validation':         10,
                      'save_checkpoint':    True,
                      'img_scale':          0.5,
                      'amp':                False,
                      'load':               False,
                      'classes':            2,
                      'bilinear':           False}
    # Loading the parameter configuration file
    if parameters is not None:
        config_path = parameters
    else:
        config_path = kwargs_default['parameters']
    params = toml.load(config_path)

    params.update(kwargs) # prioritize manually entered configuration over config file
    kwargs_default.update(params) # prioritize (manually entered + config file) over default
    params = kwargs_default # saving the dict back to return output dictionary


    # Checking if number of workers exceed available threads when in EDDIE GPU, fixing it and alerting user
    if params['server'] == "EDDIE" and params['core'] == "GPU":
        if params['num_workers'] > params['pe']:
            params['num_workers'] = params['pe']
            print(f"Number of workers ({params['num_workers']}) exceeded available threads on GPU ({params['pe']}),",
                  "It is lowered to match it")
    # Alerts user if GPU is selected but is unavailable, and automatically switches to CPU
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params['core'] == "GPU":
        if params['device'] == 'cpu':
            print("GPU specified but cuda is unavailable, cpu will be used instead")



    if params['server'] == "PC":  # Paths for working on Kiros's PC
        params['dir_img'] = Path('C:/2022_Summer_Intern/UNet_Training_With_Images/Carvana/Input')
        params['dir_mask'] = Path('C:/2022_Summer_Intern/UNet_Training_With_Images/Carvana/Target/')


        # Make base directory for storing everything
        base_dir = strftime(
            "C:/2022_Summer_Intern/UNet_Training_With_Images"
            "/Model/%Y_%m_%d_%H;%M;%S")
        os.makedirs(base_dir, exist_ok=True)


    elif params['server'] == "EDDIE":  # Paths for working on EDDIE server
        params['dir_img'] = Path('/exports/csce/eddie/eng/groups/DunnGroup/kiros/'
                                 '2022_summer_intern/UNet_Training_With_Images/Carvana/Input/')
        params['dir_mask'] = Path('/exports/csce/eddie/eng/groups/DunnGroup/kiros/'
                                  '2022_summer_intern/UNet_Training_With_Images/Carvana/Target/')

        # Make base directory for storing everything
        base_dir = strftime(
            "/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/UNet_Training_With_Images"
            "/Model/%Y_%m_%d_%H;%M;%S")
        os.makedirs(base_dir, exist_ok=True)

    params['base_dir'] = base_dir
    params['dir_checkpoint'] = Path(base_dir +'/checkpoints/')
    os.makedirs(params['dir_checkpoint'], exist_ok=True)

    # Copies the config file
    # config_file_name = config_path.split('.')[-2].split('/')[-1]
    config_file_name = config_path.split('/')[-1]
    with open(base_dir + '/' + config_file_name, "w") as f:
        toml.dump(params, f)
        f.close()

    # Path for saving segmentation images
    params['segmentation_path'] = base_dir + '/segmentation_images'
    os.makedirs(params['segmentation_path'], exist_ok=True)

    return params
#######################################################################################################################
# Functions
#######################################################################################################################


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate_epoch(net, dataloader, device, epoch, segmentation_path):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    show_image = None
    show_mask_pred = None
    show_mask_true = None

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # gets the image, prediction mask, and true mask
            show_image = image
            show_mask_pred = mask_pred
            show_mask_true = mask_true



            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    show_combi_mask = torch.empty(1, 3, 640, 959)
    for i in range(640):
        for j in range(959):
            if (show_mask_pred[0][0][i][j] < 0.5 and show_mask_pred[0][1][i][j] > 0.5):
                show_combi_mask[0][0][i][j] = 1
                show_combi_mask[0][1][i][j] = 0
                show_combi_mask[0][2][i][j] = 0
            else:
                show_combi_mask[0][0][i][j] = image[0][0][i][j]
                show_combi_mask[0][1][i][j] = image[0][1][i][j]
                show_combi_mask[0][2][i][j] = image[0][2][i][j]
    ts.show([torch.squeeze(show_image), torch.squeeze(show_mask_pred),
             torch.squeeze(show_combi_mask), torch.squeeze(show_mask_true)])
    ts.save([torch.squeeze(show_image), torch.squeeze(show_mask_pred),
             torch.squeeze(show_combi_mask), torch.squeeze(show_mask_true)], segmentation_path + f'/epoch{epoch}.jpg')

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

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
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # print("created dataset")

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # print("alr split into train/validation partitions")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)  # num_workers=4
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=1, pin_memory=True)
    # print("created data loaders")

    # (Initialize logging)
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
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    # print("optimizer set up")

    train_loss_log = []
    val_loss_log = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        # print("Begin iteration")
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

                # print("images to device done")

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # print("predicted masks")
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                    # print("loss calculated")

                optimizer.zero_grad(set_to_none=True)
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
                #break #just for testing, DELETE!!!!!!!
            # All batches in the epoch iterated through, append loss values as string type
            train_loss_log.append(epoch_loss)
            val_loss_log.append(evaluate_epoch(net, val_loader, device, epoch, segmentation_path).item())

            plot_stats(train_loss_log, val_loss_log, base_dir)

            loss_array = np.array([train_loss_log, val_loss_log]).T
            loss_dataframe = pd.DataFrame(loss_array, columns=['Training Set', 'Validation Set'])
            loss_dataframe.index.names = ['Epoch']
            loss_dataframe.index += 1
            loss_dataframe.to_csv(Path(base_dir+'/loss.csv'))

            pass




        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        print("another epoch done!")


def plot_stats(train_loss_log, val_loss_log, base_dir):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,7))
    axs[0].plot([epoch+1 for epoch in range(len(train_loss_log))], train_loss_log, label='train set', linestyle='--', color='b')
    axs[0].plot([epoch+1 for epoch in range(len(val_loss_log))], val_loss_log, label='validation set', linestyle='--', color='r')
    axs[1].plot([epoch+1 for epoch in range(len(train_loss_log))], train_loss_log, label='train set', linestyle='--', color='b')
    axs[2].plot([epoch+1 for epoch in range(len(val_loss_log))], val_loss_log, label='validation set', linestyle='--', color='r')
    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.tight_layout()
    plt.savefig(Path(base_dir+'/loss.jpg'))
    plt.close(fig)
#######################################################################################################################
# Training the model
#######################################################################################################################
if __name__ == '__main__':
    unet_train(sys.argv[1:]) # for use when debugging with pycharm
