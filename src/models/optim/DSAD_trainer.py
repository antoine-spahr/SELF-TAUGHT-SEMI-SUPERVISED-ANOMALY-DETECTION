import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

from src.models.optim.Loss_Functions import DeepSADLoss
from src.utils.utils import print_progessbar

class DSAD_trainer:
    """
    Trainer for the DSAD.
    """
    def __init__(self, c, eta, lr=1e-4, n_epoch=150, lr_milestone=(), batch_size=64,
                 weight_decay=1e-6, device='cuda', n_job_dataloader=0, print_batch_progress=False):
        """
        Constructor of the DeepSAD trainer.
        ----------
        INPUT
            |---- c (torch.Tensor) the hypersphere center.
            |---- eta (float) the deep SAD parameter weighting the importance of
            |           unkonwn/known sample in learning.
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        # learning parameters
        self.lr = lr
        self.n_epoch = n_epoch
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_job_dataloader = n_job_dataloader
        self.print_batch_progress = print_batch_progress

        # DeepSAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.train_loss = None

        self.eval_auc = None
        self.eval_time = None
        self.eval_scores = None

    def train(self, dataset, net, valid_dataset=None):
        """
        Train the DeepSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSAD to train.
            |---- valid_dataset (torch.utils.data.Dataset) the dataset on which
            |           to validate the network at each epoch. Not validated if
            |           not provided.
        OUTPUT
            |---- net (nn.Module) The trained DeepSAD.
        """
        logger = logging.getLogger()

        # make the train dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_job_dataloader)

        # put net to device
        net = net.to(self.device)

        # initialize hypersphere center
        if self.c is None:
            logger.info(' Initializing the hypersphere center.')
            self.c = self.initialize_hypersphere_center(train_loader, net)
            logger.info(' Center succesfully initialized.')

        # define loss criterion
        loss_fn = DeepSADLoss(self.c, self.eta, eps=self.eps)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('Start Training the DeepSAD.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch = len(train_loader)

        for epoch in range(self.n_epoch):
            net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get input and semi-supervized labels
                input, _, _, semi_label, _ = data
                # put them to device
                input = input.to(self.device).float().requires_grad_(True)
                semi_label = semi_label.to(self.device)

                # zero the network's gradients
                optimizer.zero_grad()
                # optimize by backpropagation
                _, embed = net(input)
                loss = loss_fn(embed, semi_label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\t Train Batch', Size=40, erase=True)

            # validate if required
            valid_auc = ''
            if valid_dataset:
                auc = self.evaluate(net, valid_dataset, return_auc=True, print_to_logger=False, save_tSNE=False)
                valid_auc = f' Valid AUC {auc:.3%} |'

            # log the epoch statistics
            logger.info(f'----| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train Time: {time.time() - epoch_start_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} |' + valid_auc)

            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update scheduler
            scheduler.step()
            if epoch + 1 in self.lr_milestone:
                logger.info(f'---- LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

        # End training
        self.train_loss = epoch_loss_list
        self.train_time = time.time() - start_time
        logger.info(f'---- Finished Training DSAD in {self.train_time:.3f} [s]')

        return net

    def evaluate(self, net, dataset, return_auc=False, print_to_logger=True, save_tSNE=True):
        """
        Evaluate the DSAD network on the provided dataset.
        ----------
        INPUT
            |---- net (nn.Module) The DeepSAD network to validate.
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is evaluated.
            |---- return_auc (bool) whether to return the computed auc or not.
            |---- print_to_logger (bool) whether to print in the logger.
            |---- save_tSNE (bool) whether to save a 2D t-SNE representation of
            |           the embeded data points
        OUTPUT
            |---- (auc) (float) the validation auc if required.
        """
        if print_to_logger:
            logger = logging.getLogger()

        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.n_job_dataloader)
        # put net on device
        net = net.to(self.device)

        # Evaluating
        if print_to_logger:
            logger.info('Start Evaluating the DSAD.')
        start_time = time.time()
        idx_label_score = []

        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data on device
                input, label, _, semi_label, idx = data
                input = input.to(self.device).float()
                label = label.to(self.device)
                semi_label = semi_label.to(self.device)
                idx = idx.to(self.device)

                # Embed input and compute anomaly  score
                _, embed = net(input)
                score = torch.norm(self.c - embed, p=2, dim=1)

                # append idx, scores, label and embeding
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist(),
                                            embed.cpu().data.numpy().tolist()))

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\t Evaluation Batch', Size=40, erase=True)

        # compute AUCs
        index, label, score, embed = zip(*idx_label_score)
        label, score = np.array(label), np.array(score)
        auc = roc_auc_score(label, score)

        if save_tSNE:
            embed = np.array(embed)
            embed = TSNE(n_components=2).fit_transform(embed)
            idx_label_score = list(zip(index, label.tolist(), score.tolist(), embed.tolist()))

        self.eval_time = time.time() - start_time
        self.eval_scores = idx_label_score
        self.eval_auc = auc

        if print_to_logger:
            logger.info(f'Evaluation Time : {self.eval_time}')
            logger.info(f'Evaluation AUC : {self.eval_auc:.3%}')
            logger.info('Finished Evaluating the DSAD.')

        if return_auc:
            return auc

    def initialize_hypersphere_center(self, loader, net, eps=0.1):
        """
        Initialize the hypersphere center as the mean output of the network over
        one forward pass.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DeepSAD network. The output must be a vector
            |           embedding of the input.
            |---- eps (float) the epsilon representing the minimum value of the
            |           component of the center.
        OUTPUT
            |---- c (torch.Tensor) the initialized center.
        """
        n_sample = 0
        net.eval()
        with torch.no_grad():
            # get embdedding dimension with one forward pass of one batch
            sample = next(iter(loader))[0].float()
            embed_dim = net(sample.to(self.device))[1].shape[1]
            # initialize c
            c = torch.zeros(embed_dim, device=self.device)
            # get the output of all samples and accumulate them
            for b, data in enumerate(loader):
                input, _, _, semi_label, _ = data
                input = input.to(self.device).float()
                semi_label = semi_label.to(self.device)
                # take normal samples only
                input = input[semi_label != -1]

                _, embed = net(input)
                n_sample += embed.shape[0]
                c += torch.sum(embed, dim=0)

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\t Center Initialization Batch', Size=40, erase=True)

        # take the mean of accumulated c
        c /= n_sample
        # check if c_i are epsilon too close to zero to avoid them to be rivialy matched to zero
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c > 0)] = eps

        return c
