import torch
import torch.nn as nn
import torch.nn.functional as F
import toml

from tqdm import tqdm
import wandb
import os
from os.path import join
import numpy as np

from time import strftime
from rich import print as rprint
from collections import defaultdict

from gelgenie.segmentation.helper_functions.dice_score import dice_loss
from gelgenie.segmentation.helper_functions.depth_based_loss import unet_weight_map
from ..helper_functions.stat_functions import save_statistics, load_statistics
from ..helper_functions.display_functions import plot_stats, visualise_segmentation
from ..helper_functions.general_functions import create_dir_if_empty, create_summary_table
from .training_setup import core_setup
from ..data_handling import prep_train_val_dataloaders
from ..networks import model_configure
from gelgenie.segmentation.helper_functions.dice_score import multiclass_dice_coeff


class TrainingHandler:
    def __init__(self, experiment_name, base_dir,
                 training_parameters, processing_parameters,
                 data_parameters, model_parameters):

        self.main_folder = join(base_dir, experiment_name)

        # basic setup
        if training_parameters['load_checkpoint']:
            saved_config = toml.load(join(self.main_folder, 'config.toml'))
            unique_id = saved_config['training']['wandb_id']
        else:
            unique_id = wandb.util.generate_id()

        training_parameters['wandb_id'] = unique_id

        if os.path.exists(self.main_folder) and not training_parameters['load_checkpoint']:
            next_id = 1
            while os.path.exists(self.main_folder):  # continues to generate new names until an unused one is found
                self.main_folder = join(base_dir, '%s_%d' % (experiment_name, next_id))
                next_id += 1
            rprint('[bold red]Original folder name was already taken, so the id %d'
                   ' was appended to the current experiment name.[/ bold red]' % (next_id-1))

        self.checkpoints_folder = join(self.main_folder, 'checkpoints')
        self.example_output_folder = join(self.main_folder, 'segmentation_samples')
        self.logs_folder = join(self.main_folder, 'training_logs')

        self.device = torch.device('cuda' if processing_parameters['device'].lower() == 'gpu' else 'cpu')

        self.wandb_track = training_parameters['wandb_track']
        self.resumed_model = True if training_parameters['load_checkpoint'] else False

        # Initialise wandb logging
        if self.wandb_track:
            if processing_parameters['base_hardware'] == 'EDDIE':
                self.wandb_package = wandb.init(project='GelGenie Final Phase', entity='dunn-group', resume='allow',
                                                name=os.path.basename(self.main_folder), id=unique_id,
                                                settings=wandb.Settings(start_method="fork"))
            else:
                self.wandb_package = wandb.init(project='GelGenie Final Phase', entity='dunn-group',
                                                name=os.path.basename(self.main_folder), id=unique_id,
                                                resume='allow')

        create_dir_if_empty(self.main_folder, self.checkpoints_folder, self.example_output_folder, self.logs_folder)

        # model setup
        self.net, model_structure, model_docstring = model_configure(device=self.device, **model_parameters)

        if not self.resumed_model:
            with open(join(self.main_folder, 'model_structure.txt'), 'w', encoding='utf-8') as f:
                f.write(str(model_structure))
            with open(join(self.main_folder, 'model_summary.txt'), 'w', encoding='utf-8') as f:
                rprint(model_docstring, file=f)
        rprint(model_docstring)

        # training details setup
        self.optimizer, self.scheduler = core_setup(self.net, **training_parameters)
        self.current_epoch = 1
        self.max_epochs = training_parameters['epochs']
        self.checkpoint_saving = training_parameters['save_checkpoint']
        self.checkpoint_save_frequency = training_parameters['checkpoint_frequency']
        self.model_cleanup_frequency = training_parameters['model_cleanup_frequency']
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=training_parameters['grad_scaler'])  # CUDA only function
        self.use_amp_scaler = training_parameters['grad_scaler']

        # model loading
        if training_parameters['load_checkpoint']:
            self.load_checkpoint(training_parameters['load_checkpoint'])

        # data setup
        self.train_loader, self.val_loader, self.train_image_count, self.val_image_count = prep_train_val_dataloaders(
            **data_parameters)

        self.loss_definition = training_parameters['loss']  # list of all losses to be used
        if not isinstance(self.loss_definition, list):
            self.loss_definition = [self.loss_definition]

        # true if individual classes are weighted according to their frequency
        self.class_weighting_enabled = training_parameters['class_loss_weighting']

        # reduces effect of class balancing
        self.class_weighting_damper = torch.tensor(training_parameters['class_loss_weight_damper']).to(device=self.device)
        self.class_weighting = (torch.tensor(self.train_loader.dataset.class_weighting.astype(np.float32)).
                                to(device=self.device))  # class imbalance weighting
        self.class_weighting = self.class_weighting * self.class_weighting_damper

        # individual weighting for each component of loss fn
        self.loss_function_weighting = training_parameters['loss_component_weighting']

        # crossentropy loss reduction/weighting off when these are controlled separately
        self.crossentropy_loss_fn = nn.CrossEntropyLoss(
            reduction='none' if 'weighted' in ' '.join(self.loss_definition) else 'mean',
            weight=self.class_weighting if ('weighted' not in ' '.join(self.loss_definition) and
                                            self.class_weighting_enabled) else None)

        time_started = strftime("%Y_%m_%d_%H;%M;%S")
        # diagnostic strings
        diagnostic_info = [['Starting epoch', self.current_epoch],
                           ['Epochs to run', self.max_epochs - self.current_epoch + 1],
                           ['Device', str(self.device)],
                           ['Learning rate', str(self.optimizer.param_groups[0]['lr'])],
                           ['Training set images', str(self.train_image_count)],
                           ['Validation set images', str(self.val_image_count)],
                           ['Augmentations', str(data_parameters['apply_augmentations'])],
                           ['Checkpoints', str(self.checkpoint_saving)],
                           ['Optimizer', training_parameters['optimizer_type']],
                           ['Scheduler', training_parameters['scheduler_type']],
                           ['Network', model_parameters['model_name']],
                           ['Date/time started', time_started],
                           ]

        time_log = join(self.main_folder, 'time_log.txt')
        with open(time_log, 'a' if os.path.exists(time_log) else 'w') as f:
            if self.resumed_model:
                f.write('Time resumed training run: %s\n' % time_started)
            else:
                f.write('Time initiated first training run: %s\n' % time_started)

        if self.wandb_track and not self.resumed_model:
            # Logging parameters for training
            self.wandb_package.config.update(dict(training_parameters=training_parameters,
                                                  processing_parameters=processing_parameters,
                                                  data_parameters=data_parameters,
                                                  model_parameters=model_parameters))

        summary_table = create_summary_table("Training Summary", ['Parameter', 'Value'], ['cyan', 'green'],
                                             diagnostic_info)
        rprint(summary_table)

    def load_checkpoint(self, checkpoint):
        """
        Loads checkpoint model, optimizer and scheduler weights
        :param checkpoint: Model checkpoint epoch number (must be stored in checkpoints folder)
        :return: None
        """
        filepath = join(self.checkpoints_folder, 'checkpoint_epoch_%s.pth' % checkpoint)
        saved_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(saved_dict['network'])  # Load in state dictionary of model network
        self.optimizer.load_state_dict(saved_dict['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(saved_dict['scheduler'])
        self.current_epoch = saved_dict['epoch'] + 1
        rprint(f'[bold orange] Model, optimizer and scheduler weights loaded from '
               f'(epoch {self.current_epoch})[/bold orange]')

    def save_checkpoint(self, name):
        """
        Saves model to a checkpoint, along with optimizer and scheduler weights
        :param name: Filename for checkpoint
        :return: None
        """
        full_state_dict = {'network': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
        if self.scheduler:
            full_state_dict['scheduler'] = self.scheduler.state_dict()
        full_state_dict['epoch'] = self.current_epoch
        torch.save(full_state_dict, join(self.checkpoints_folder, name))
        rprint(f'[bold orange]Model, optimizer and scheduler weights saved to {name}.[/bold orange]')

    def loss_calculation(self, masks_pred, true_masks, epoch_metrics):

        full_losses = []
        for lossfn, loss_weight in zip(self.loss_definition, self.loss_function_weighting):
            if lossfn == 'dice':
                loss_dice = dice_loss(F.softmax(masks_pred, dim=1).float(),  # TODO: re-inspect here
                                      F.one_hot(true_masks, self.net.n_classes).permute(0, 3, 1, 2).float(),
                                      multiclass=True)
                epoch_metrics['Dice Loss'] += loss_dice.detach().cpu().numpy()
                full_losses.append(loss_dice*loss_weight)
            elif lossfn == 'crossentropy':
                loss_ce = self.crossentropy_loss_fn(masks_pred, true_masks)
                epoch_metrics['Cross-Entropy Loss'] += loss_ce.detach().cpu().numpy()
                full_losses.append(loss_ce*loss_weight)
            elif lossfn == 'unet_weighted_crossentropy':
                loss_ce = self.crossentropy_loss_fn(masks_pred, true_masks)
                weighting = np.zeros(true_masks.shape)
                for i in range(true_masks.shape[0]):
                    weighting[i, ...] = unet_weight_map(
                        true_masks[i, ...].cpu().numpy(), wc=self.class_weighting.cpu().numpy() if self.class_weighting_enabled else None)
                weighting = torch.from_numpy(weighting).to(device=self.device)

                loss_ce = (loss_ce * weighting).sum()/weighting.sum()
                full_losses.append(loss_ce*loss_weight)
                epoch_metrics['Cross-Entropy Loss'] += loss_ce.detach().cpu().numpy()
            elif lossfn == 'simple_weighted_crossentropy':
                raise NotImplementedError('Simple weighted crossentropy loss not implemented yet')
            else:
                raise RuntimeError('Loss definition not recognised')

        final_loss = sum(full_losses)

        epoch_metrics['Training Loss'] += final_loss.detach().cpu().numpy()

        return final_loss

    def train_epoch(self, epoch):
        """
        Oversees a single training epoch.
        :param epoch: Epoch number (for logging purposes)
        :return: Average loss values and learning rate for this epoch
        """

        epoch_metrics = defaultdict(float)
        self.net.train()
        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.max_epochs} training', unit='batch') as pbar:
            for batch in self.train_loader:
                images = batch['image']
                true_masks = batch['mask']
                images = images.to(device=self.device)
                true_masks = true_masks.to(device=self.device, dtype=torch.long)

                # Use autocast if amp is used
                with torch.cuda.amp.autocast(enabled=self.use_amp_scaler):
                    masks_pred = self.net(images)
                    loss = self.loss_calculation(masks_pred, true_masks, epoch_metrics)

                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        for key, val in epoch_metrics.items():  # averaging results across the full epoch
            epoch_metrics[key] = val / len(self.train_loader)
        epoch_metrics['Learning Rate'] = self.optimizer.param_groups[0]['lr']

        return epoch_metrics

    def eval_epoch(self, epoch):
        """
        Oversees a single validation epoch.
        :param epoch: Current epoch number (for logging purposes)
        :return: All validation metrics for this epoch
        """

        epoch_metrics = defaultdict(float)
        self.net.eval()

        # iterate over the validation set
        with tqdm(total=len(self.val_loader), desc=f'Epoch {epoch}/{self.max_epochs} validation', unit='batch',
                  leave=False) as pbar:
            for b_index, batch in enumerate(self.val_loader):
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to device, set the type of pixel values
                image = image.to(device=self.device, dtype=torch.float32)
                mask_true = mask_true.to(device=self.device, dtype=torch.long)

                # one_hot format has only a single 1 bit and the rest are 0 bits
                # i.e. if n_classes is 3 will transform [0] to [1,0,0], [1] to [0,1,0], [2] to [0,0,1]
                # The permute() function changes it from [N, H, W, C] to [N, C, H, W]
                mask_true = F.one_hot(mask_true, self.net.n_classes).permute(0, 3, 1, 2).float()

                with torch.no_grad():
                    # predict the mask
                    mask_pred = self.net(image)

                    # convert to one-hot format if 3 channels, else apply sigmoid function
                    # Calculate dice score
                    if self.net.n_classes == 1:
                        raise RuntimeError('Dice score not implemented for single class segmentation.')
                    else:
                        mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.net.n_classes).permute(0, 3, 1, 2).float()
                        # compute the Dice score, ignoring background
                        current_score = multiclass_dice_coeff(mask_pred[:, 1:, ...],
                                                              mask_true[:, 1:, ...],
                                                              reduce_batch_first=False).cpu().numpy()
                pbar.update(1)
                pbar.set_postfix(**{'Dice score (batch)': current_score})
                epoch_metrics['Dice Score'] += current_score
                if b_index < 3:  # prepares sample outputs
                    image_array, threshold_mask_array, combi_mask_array, mask_true_array = \
                        visualise_segmentation(image.squeeze(), mask_pred.squeeze(),
                                               mask_true.squeeze(),
                                               epoch, dice_score=current_score,
                                               optional_name=batch['image_name'][0],
                                               segmentation_path=self.example_output_folder)
                    if b_index == 0:  # only one sample sent to wandb
                        seg_sample_package = {'image': image_array, 'threshold_mask': threshold_mask_array,
                                              'combi_mask': combi_mask_array, 'mask_true': mask_true_array}

        for key, val in epoch_metrics.items():  # averaging results across the full epoch
            epoch_metrics[key] = val / len(self.val_loader)

        return epoch_metrics, seg_sample_package

    def full_training(self):
        """
        Runs the full training process, including training, validation, log saving and model checkpointing.
        :return: None
        """

        if self.resumed_model:
            old_metrics = load_statistics(self.logs_folder, 'training_stats.csv', config='pd')
            total_metrics = defaultdict(list, old_metrics.to_dict(orient='list'))
        else:
            total_metrics = defaultdict(list)

        # Begin training
        for epoch in range(self.current_epoch, self.max_epochs + 1):

            train_metrics = self.train_epoch(epoch)
            if self.val_loader is None:
                # if no validation data provided, eval is turned off
                val_metrics = {}
                seg_sample_package = {}
            else:
                val_metrics, seg_sample_package = self.eval_epoch(epoch)

            current_epoch_metrics = {**train_metrics, **val_metrics}  # combines all metrics

            for key in current_epoch_metrics.keys():
                total_metrics[key].append(current_epoch_metrics[key])
            total_metrics['Epoch'].append(epoch)

            if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(current_epoch_metrics['Dice Score'])
            elif type(self.scheduler).__name__ == 'CosineAnnealingWarmRestarts':
                self.scheduler.step()

            if self.val_loader is None:
                stat_plotting = [['Training Loss'], ['Learning Rate']]
            else:
                stat_plotting = [['Training Loss', 'Dice Score'], ['Learning Rate']]
            if 'Dice Loss' in current_epoch_metrics and 'Cross-Entropy Loss' in current_epoch_metrics:
                stat_plotting += [['Dice Loss', 'Cross-Entropy Loss']]

            # plot training metrics
            plot_stats(total_metrics, stat_plotting, self.logs_folder, 'metric_plots.pdf')

            # save results to file
            save_statistics(experiment_log_dir=self.logs_folder, filename='training_stats.csv',
                            stats_dict=total_metrics,
                            selected_data=epoch - 1 if epoch > 1 else None,
                            append=True if epoch > 1 else False)

            if self.wandb_track:
                if self.val_loader is None:
                    self.wandb_package.log({
                        'Learning rate': current_epoch_metrics['Learning Rate'],
                        'Train loss': current_epoch_metrics['Training Loss'],
                        'epoch': epoch,
                    })
                else:
                    # log eval sample to wandb, if available
                    self.wandb_package.log({
                        'Learning rate': current_epoch_metrics['Learning Rate'],
                        'Validation Dice': current_epoch_metrics['Dice Score'],
                        'Train loss': current_epoch_metrics['Training Loss'],
                        'Validation sample': {
                            'Input': wandb.Image(seg_sample_package['image']),
                            'Segmentation': {
                                'True': wandb.Image(seg_sample_package['mask_true']),
                                'Predicted': wandb.Image(seg_sample_package['threshold_mask']),
                                'Predicted-superimposed': wandb.Image(seg_sample_package['combi_mask']),
                            },
                        },
                        'epoch': epoch,
                    })

            if self.checkpoint_saving and (epoch == self.max_epochs or epoch % self.checkpoint_save_frequency == 0):
                self.save_checkpoint('checkpoint_epoch_%s.pth' % epoch)

            if self.model_cleanup_frequency > 0 and epoch % self.model_cleanup_frequency == 0:
                top_epoch_idx = sorted(range(len(total_metrics['Dice Score'])),
                                       key=lambda i: total_metrics['Dice Score'][i])[-2:]
                top_epochs = [total_metrics['Epoch'][i] for i in top_epoch_idx]
                deleted_epochs = []
                for epoch_id in total_metrics['Epoch']:
                    if epoch_id not in top_epochs and epoch_id % 100 != 0:
                        model_file = join(self.checkpoints_folder, 'checkpoint_epoch_%s.pth' % epoch_id)
                        if os.path.isfile(model_file):
                            os.remove(model_file)
                            deleted_epochs.append(epoch_id)
                rprint('[bold red]Deleted checkpoints for epochs: %s[/bold red]' % deleted_epochs)
            self.current_epoch += 1
            print('--------------------------')
        if self.wandb_track:
            self.wandb_package.finish()
