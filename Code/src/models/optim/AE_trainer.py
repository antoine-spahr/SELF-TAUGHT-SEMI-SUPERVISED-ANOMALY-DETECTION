import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

from src.models.optim.Loss_Functions import MaskedMSELoss
from src.utils.utils import print_progessbar

class AE_trainer:
    """
    Class to Train an AutoEncoder.
    """
    def __init__(self, n_epoch=150, lr=1e-4, weight_decay=1e-6, lr_milestone=(),
                 batch_size=64, n_job_dataloader=0, device='cuda', print_batch_progress=False):
        """
        Constructor of the AutoEncoder trainer.
        ----------
        INPUT
            |---- n_epoch (int) the number of epoch.
            |---- lr (float) the learning rate.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        self.n_epoch = n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.n_job_dataloader = n_job_dataloader
        self.device = device
        self.print_batch_progress = print_batch_progress
        # Outputs
        self.train_time = None
        self.train_loss = None
        self.eval_repr = None

    def train(self, net, dataset, valid_dataset=None):
        """
        Train the autoencoder network on the provided dataset.
        ----------
        INPUT
            |---- net (nn.Module) The autoencoder to train. It must return two
            |           embedding (after the convolution and after the MLP) as
            |           well as the reconstruction
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image , the label,
            |           a mask, semi-supervised label and the index.
            |---- valid_dataset (torch.utils.data.Dataset) the optional dataset
            |           on which to validate the model at each epoch.
        OUTPUT
            |---- net (nn.Module) The trained autoencoder.
        """
        logger = logging.getLogger()

        # make train dataloader using image and mask
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.n_job_dataloader)
        # define loss_fn
        loss_fn = MaskedMSELoss()

        # set network on device
        net = net.to(self.device)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # set the learning rate scheduler (multiple phase learning)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Training
        logger.info('Start Training AE.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch = len(train_loader)

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            net.train()

            for b, data in enumerate(train_loader):
                input, _, mask, semi_label, _ = data
                # put inputs to device
                input = input.to(self.device).float().requires_grad_(True)
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)

                # zero the network gradients
                optimizer.zero_grad()

                # Update network paramters by backpropagation by considering only the loss on the mask
                _, _, rec = net(input)
                loss = loss_fn(rec, input, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=40, erase=True)

            valid_auc = ''
            if valid_dataset:
                auc = self.evaluate(net, valid_dataset, save_tSNE=False, return_auc=True,
                                     print_to_logger=False)
                valid_auc = f' Valid AUC {auc:.6f} |'

            # display epoch statistics
            logger.info(f'----| Epoch {epoch + 1:03}/{self.n_epoch:03} '
                            f'| Time {time.time() - epoch_start_time:.3f} [s]'
                            f'| Loss {epoch_loss / n_batch:.6f} |' + valid_auc)

            # store loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update learning rate if milestone is reached
            scheduler.step()
            if epoch + 1 in self.lr_milestone:
                logger.info(f'---- LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

        # Save results
        self.train_time = time.time() - start_time
        self.train_loss = epoch_loss_list
        logger.info(f'---- Finished Training AE in {self.train_time:.3f} [s].')

        return net


    def evaluate(self, net, dataset, print_to_logger=True, return_auc=False, save_tSNE=True):
        """
        Evaluate the natwork on the provided dataset.
        ----------
        INPUT
            |---- net (nn.Module) The autoencoder to train. It must return two
            |           embedding (after the convolution and after the MLP) as
            |           well as the reconstruction
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image and a mask
            |           of where the loss is to be computed.
            |---- print_to_logger (bool) whether to print info in logger.
            |---- return_auc (bool) whether to return the computed AUC.
            |---- save_tSNE (bool) whether to save the intermediate representation
            |           as a 2D vector using tSNE.
        OUTPUT
            |---- None
        """
        if print_to_logger:
            logger = logging.getLogger()
        # make dataloader (with drop_last = True to ensure that the loss can be computed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=self.n_job_dataloader)

        # put net on device
        net = net.to(self.device)

        # define loss function
        loss_fn = MaskedMSELoss(reduction='none')

        if print_to_logger:
            logger.info("Start Evaluating AE.")

        idx_label_scores = []
        n_batch = len(loader)

        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, label, mask, semi_label, idx  = data
                # put inputs to device
                input = input.to(self.device).float().requires_grad_(True)
                label = label.to(self.device)
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                idx = idx.to(self.device)

                h, z, rec = net(input)

                # compute score as mean loss over by sample
                rec_loss = loss_fn(rec, input, mask)
                score = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # append scores : idx label score h z
                idx_label_scores += list(zip(idx.cpu().data.numpy().tolist(),
                                             label.cpu().data.numpy().tolist(),
                                             score.cpu().data.numpy().tolist(),
                                             h.cpu().data.numpy().tolist(),
                                             z.cpu().data.numpy().tolist()))

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tEvaluation Batch', Size=40, erase=True)

        if save_tSNE:
            if print_to_logger:
                logger.info("Computing the t-SNE representation.")
            # Apply t-SNE transform on embeddings
            index, label, scores, h, z = zip(*idx_label_scores)
            h, z = np.array(h), np.array(z)
            h = TSNE(n_components=2).fit_transform(h)
            z = TSNE(n_components=2).fit_transform(z)
            self.eval_repr = list(zip(index, label, scores, h.tolist(), z.tolist()))

            if print_to_logger:
                logger.info("Succesfully computed the t-SNE representation ")

        if return_auc:
            _, label, scores, _, _ = idx_label_scores
            auc = roc_auc_score(np.array(label), np.array(scores))
            return auc
