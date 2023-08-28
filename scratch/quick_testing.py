import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.rich import tqdm
import wandb
import os
from os.path import join
from pathlib import Path
import logging
import numpy as np
from time import strftime
from rich import print as rprint

from gelgenie.segmentation.helper_functions.dice_score import dice_loss


class TrainingHandler:
    def __init__(self, experiment_name, base_dir,
                 training_parameters, processing_parameters,
                 data_parameters, model_parameters):
        # basic setup
        self.main_folder = join(base_dir, experiment_name + '_' + strftime("%Y_%m_%d_%H;%M;%S"))
        self.checkpoints_folder = join(self.main_folder, 'checkpoints')
        self.example_output_folder = join(self.main_folder, 'segmentation_samples')
        self.device = processing_parameters['device']
        self.device = torch.device('cuda' if self.device.lower() == 'gpu' else 'cpu')

        if os.path.exists(self.main_folder):
            raise RuntimeError('Main experiment folder already exists - make sure to not overwrite!')
        # create_dir_if_empty(self.main_folder, self.checkpoints_folder, self.example_output_folder)

        # model setup
        # self.net, model_structure, model_docstring = model_configure(device=self.device, **model_parameters)

        # with open(join(self.main_folder, 'model_structure.txt'), 'w', encoding='utf-8') as f:
        #     f.write(str(model_structure))
        # with open(join(self.main_folder, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        #     rprint(model_docstring, file=f)
        # rprint(model_docstring)

        # training details setup
        # self.optimizer, self.scheduler = core_setup(self.net, **training_parameters)
        self.current_epoch = 0
        self.max_epochs = training_parameters['epochs']
        self.checkpoint_saving = training_parameters['save_checkpoint']
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=training_parameters['grad_scaler'])
        self.use_amp_scaler = training_parameters['grad_scaler']
        self.main_loss_fn = nn.CrossEntropyLoss()
        self.loss_definition = training_parameters['loss']

        # model loading
        if training_parameters['load_checkpoint']:
            self.load_checkpoint(training_parameters['load_checkpoint'])

        # data setup
        # self.train_loader, self.val_loader, self.train_image_count, self.val_image_count = prep_dataloader(**data_parameters)

        diagnostic_info = [['Starting epoch', self.current_epoch],
                           ['Epochs to run', self.max_epochs - self.current_epoch],
                           ['Device', str(self.device)],
                           ['Learning rate', str(self.optimizer.param_groups[0]['lr'])],
                           ['Training set images', str(self.train_image_count)],
                           ['Validation set images', str(self.val_image_count)],
                           ['Checkpoints', str(self.checkpoint_saving)],
                           ['Optimizer', training_parameters['optimizer_type']],
                           ['Scheduler', training_parameters['scheduler_type']],
                           ['Network', model_parameters['model_name']]
                           ]
        # summary_table = create_summary_table("Training Summary", ['Parameter', 'Value'], ['cyan', 'green'], diagnostic_info)
        # rprint(summary_table)
       # TODO: add wandb logging

    def load_checkpoint(self, checkpoint):
        """
        Loads checkpoint model, optimizer and scheduler weights
        :param checkpoint: Name of model checkpoint (must be stored in checkpoints folder)
        :return: None
        """
        filepath = join(self.checkpoints_folder, checkpoint)
        saved_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(saved_dict['network'])  # Load in state dictionary of model network
        self.optimizer.load_state_dict(saved_dict['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(saved_dict['scheduler'])
        self.current_epoch = saved_dict['epoch'] + 1
        rprint(f'[bold orange] Model, optimizer and scheduler weights loaded from {checkpoint} '
               f'(epoch {self.current_epoch})[/bold orange]')

    def save_checkpoint(self, name):
        full_state_dict = {}
        full_state_dict['network'] = self.net.state_dict()
        full_state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler:
            full_state_dict['scheduler'] = self.scheduler.state_dict()
        full_state_dict['epoch'] = self.current_epoch
        torch.save(full_state_dict, join(self.checkpoints_folder, name))
        rprint(f'[bold orange]Model, optimizer and scheduler weights saved to {name}.[/bold orange]')

   def full_training(self):
        train_loss_log = []
        val_loss_log = []

        # TODO: add more intricate train loss logging
        # TODO: add WnB again
        # TODO: complete training loop
        # TODO: check averaging of epoch losses

        # Begin training
        for epoch in range(self.current_epoch, self.max_epochs + 1):
            self.net.train()
            epoch_loss = 0
            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.max_epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images = batch['image']
                    true_masks = batch['mask']
                    images = images.to(device=self.device)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)

                    # Use autocast if amp is used
                    with torch.cuda.amp.autocast(enabled=self.use_amp_scaler):
                        masks_pred = self.net(images)
                        if self.loss_definition == 'both':
                            loss = self.crossentropy_loss_fn(masks_pred, true_masks) \
                                   + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                               F.one_hot(true_masks, self.net.n_classes).permute(0, 3, 1, 2).float(),
                                               multiclass=True)
                        elif self.loss_definition == 'CrossEntropy':
                            loss = self.crossentropy_loss_fn(masks_pred, true_masks)
                        elif self.loss_definition == 'Dice':
                            loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                             F.one_hot(true_masks, self.net.n_classes).permute(0, 3, 1, 2).float(),
                                             multiclass=True)
                        else:
                            raise RuntimeError('Loss definition not recognised')

                    self.optimizer.zero_grad()  # this ensures that all weight gradients are zeroed before moving on to the next set of gradients
                    self.grad_scaler.scale(loss).backward()  # this calculates the gradient for all weights (backpropagation)
                    self.grad_scaler.step(self.optimizer)  # here, the optimizer will calculate and make the change necessary for each weight based on its defined rules
                    self.grad_scaler.update()

                    pbar.update(1)
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    break
