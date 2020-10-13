import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import logging

from sklearn.manifold import TSNE

from src.models.optim.Loss_Functions import InfoNCE_loss
from src.utils.utils import print_progessbar

class Contrastive_trainer:
    """
    Define a trainer of the encoder network for the contrastice task.
    """
    def __init__(self, tau, n_epoch=100, batch_size=32,lr=1e-3, weight_decay=1e-6,
                 lr_milestones=(), n_job_dataloader=0, device='cuda',
                 print_batch_progress=False):
        """
        Build a contrastive trainer.
        ----------
        INPUT
            |---- tau (float) the temperature hyperparameter.
            |---- n_epoch (int) the number of epoch.
            |---- lr (float) the learning rate.
            |---- lr_milestones (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_job_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        self.tau = tau
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones
        self.n_job_dataloader = n_job_dataloader
        self.device = device
        self.print_batch_progress = print_batch_progress

        self.train_time = None
        self.train_loss = None
        self.eval_repr = None

    def train(self, dataset, net, valid_dataset=None):
        """
        Train the network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return two transformed version
            |           of an image.
            |---- net (nn.Module) The Encoder to train.
            |---- valid_dataset (torch.utils.data.Dataset) the dataset on which
            |           to validate the network at each epoch. Not validated if
            |           not provided.
        OUTPUT
            |---- net (nn.Module) The trained Encoder.
        """
        logger = logging.getLogger()

        # make dataloader (with drop_last = True to ensure that the loss can be computed)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=self.n_job_dataloader,
                                        drop_last=True)

        # put net on device
        net = net.to(self.device)

        # define loss function
        loss_fn = InfoNCE_loss(self.tau, self.batch_size, device=self.device)

        # define the optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define the learning rate scheduler : 90% reduction at each steps
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Start Contrastive Training.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch = len(train_loader)

        for epoch in range(self.n_epoch):
            net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get data on device
                input_1, input_2, _, _ = data
                input_1 = input_1.to(self.device).float().requires_grad_(True)
                input_2 = input_2.to(self.device).float().requires_grad_(True)

                # Update by Backpropagation : Fowrad + Backward + step
                optimizer.zero_grad()
                _, z_1 = net(input_1)
                _, z_2 = net(input_2)

                # normalize embeddings
                z_1 = F.normalize(z_1, dim=1)
                z_2 = F.normalize(z_2, dim=1)

                loss = loss_fn(z_1, z_2)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=40, erase=True)

            # compute valid_loss if required
            valid_loss = ''
            if valid_dataset:
                loss = self.evaluate(valid_dataset, net, save_tSNE=False, return_loss=True,
                                     print_to_logger=False)
                valid_loss = f' Valid Loss {loss:.6f} |'

            # display epoch statistics
            logger.info(f'----| Epoch {epoch + 1:03}/{self.n_epoch:03} '
                            f'| Time {time.time() - epoch_start_time:.3f} [s]'
                            f'| Loss {epoch_loss / n_batch:.6f} |' + valid_loss)
            # store loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update learning rate if milestone is reached
            scheduler.step()
            if epoch + 1 in self.lr_milestones:
                logger.info(f'---- LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

        # Save results
        self.train_time = time.time() - start_time
        self.train_loss = epoch_loss_list
        logger.info(f'---- Finished Contrastive Training in {self.train_time:.3f} [s].')

        return net

    def evaluate(self, dataset, net, save_tSNE=False, return_loss=True, print_to_logger=True):
        """
        Evaluate the Contrative network on the provided dataset.
        ----------
        INPUT
            |---- net (nn.Module) The Encoder network to validate.
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is evaluated.
            |---- print_to_logger (bool) whether to print in the logger.
            |---- save_tSNE (bool) whether to save a 2D t-SNE representation of
            |           the embeded data points
            |---- return_loss (bool) whether to return the validation loss.
        OUTPUT
            |---- (auc) (float) the validation loss if required.
        """
        if print_to_logger:
            logger = logging.getLogger()
        # make dataloader (with drop_last = True to ensure that the loss can be computed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=self.n_job_dataloader,
                                        drop_last=True)

        # put net on device
        net = net.to(self.device)

        # define loss function
        loss_fn = InfoNCE_loss(self.tau, self.batch_size, device=self.device)

        if print_to_logger:
            logger.info("Start Evaluating Contrastive.")

        net.eval()
        with torch.no_grad():
            sum_loss = 0.0
            idx_h_z = []
            n_batch = len(loader)

            for b, data in enumerate(loader):
                # get input
                input_1, input_2, _, idx = data
                input_1 = input_1.to(self.device).float()
                input_2 = input_2.to(self.device).float()
                idx = idx.to(self.device)
                # forward
                h_1, z_1 = net(input_1)
                h_2, z_2 = net(input_2)
                # normalize
                z_1 = F.normalize(z_1, dim=1)
                z_2 = F.normalize(z_2, dim=1)
                # compute loss
                loss = loss_fn(z_1, z_2)
                sum_loss += loss.item()
                # save embeddings
                if save_tSNE:
                    idx_h_z += list(zip(idx.cpu().data.numpy().tolist(),
                                        h_1.cpu().data.numpy().tolist(),
                                        z_1.cpu().data.numpy().tolist()))

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tEvaluation Batch', Size=40, erase=True)

        if save_tSNE:
            if print_to_logger:
                logger.info("Computing the t-SNE representation.")
            # Apply t-SNE transform on embeddings
            index, h, z = zip(*idx_h_z)
            h, z = np.array(h), np.array(z)
            h = TSNE(n_components=2).fit_transform(h)
            z = TSNE(n_components=2).fit_transform(z)
            self.eval_repr = list(zip(index, h.tolist(), z.tolist()))

            if print_to_logger:
                logger.info("Succesfully computed the t-SNE representation ")

        if return_loss:
            return loss / n_batch
