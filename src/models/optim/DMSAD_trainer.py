import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from src.models.optim.Loss_Functions import DMSADLoss
from src.utils.utils import print_progessbar

class DMSAD_trainer:
    """
    Trainer for the DMSAD.
    """
    def __init__(self, c, R, eta=1.0, gamma=0.05, n_sphere_init=100, n_epoch=150,
                 lr=1e-4, lr_milestone=(), batch_size=64, weight_decay=1e-6,
                 device='cuda', n_job_dataloader=0, print_batch_progress=False):
        """
        Constructor of the DMSAD trainer.
        ----------
        INPUT
            |---- c (array like N_sphere x Embed dim) the centers of the hyperspheres.
            |           If None, the centers are initialized using Kmeans.
            |---- R (1D array) the radii associated with the centers.
            |---- eta (float) the weight of semi-supervised labels in the loss.
            |---- gamma (float) the fraction of allowed outlier when setting the
            |           radius of each sphere in the end.
            |---- n_sphere_init (int) the number of initial hypersphere.
            |---- n_epoch (int) the number of epoch.
            |---- lr (float) the learning rate.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_job_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        # learning parameters
        self.n_epoch = n_epoch
        self.lr = lr
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_job_dataloader = n_job_dataloader
        self.print_batch_progress = print_batch_progress

        # DMSAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.R = torch.tensor(R, device=self.device) if R is not None else None
        self.eta = eta
        self.gamma = gamma
        self.n_sphere_init = n_sphere_init

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
        Train the DMSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, label, mask
            |           semi-supervized labels and the index.
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
            logger.info(' Initializing the hypersphere centers.')
            self.initialize_centers(train_loader, net)
            logger.info(f' {self.c.shape[0]} centers successfully initialized.')

        # define loss criterion
        loss_fn = DMSADLoss(self.eta, eps=self.eps)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('Start Training the DMSAD.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch = len(train_loader)

        for epoch in range(self.n_epoch):
            net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            n_k = torch.zeros(self.c.shape[0], device=self.device)

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
                loss = loss_fn(embed, self.c, semi_label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # get the closest sphere and count the number of normal samples per sphere
                idx = torch.argmin(torch.norm(self.c.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=2), dim=1)
                for i in idx[semi_label != -1]:
                    n_k[i] += 1

                if self.print_batch_progress:
                    print_progessbar(b, len(train_loader), Name='\t\tTrain Batch', Size=40, erase=True)

            # remove centers with less than gamma fraction of largest hypersphere number of sample
            self.c = self.c[n_k >= self.gamma * torch.max(n_k)]

            # validate if required
            valid_auc = ''
            if valid_dataset:
                auc = self.evaluate(net, valid_dataset, return_auc=True, print_to_logger=False, save_tSNE=False)
                valid_auc = f' Valid AUC {auc:.3%} |'

            # log the epoch statistics
            logger.info(f'----| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train Time: {time.time() - epoch_start_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} '
                        f'| N sphere {self.c.shape[0]:03} |' + valid_auc)

            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update scheduler
            scheduler.step()
            if epoch + 1 in self.lr_milestone:
                logger.info(f'---- LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

        # Set the radius of each sphere as 1-gamma quantile of normal samples distances
        logger.info(f'---- Setting the hyperspheres radii as the {1-self.gamma:.1%} quantiles of normal sample distances.')
        self.set_radius(train_loader, net)
        logger.info(f'---- {self.R.shape[0]} radii successufully defined.')

        # End training
        self.train_loss = epoch_loss_list
        self.train_time = time.time() - start_time
        logger.info(f'---- Finished Training DMSAD in {self.train_time:.3f} [s]')

        return net

    def evaluate(self, net, dataset, return_auc=False, print_to_logger=True, save_tSNE=True):
        """
        Evaluate the DSAD network on the provided dataset.
        ----------
        INPUT
            |---- net (nn.Module) The DMSAD network to validate.
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
            logger.info('Start Evaluating the DMSAD.')
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
                # find closest sphere
                score, sphere_idx = torch.min(torch.norm(self.c.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=2), dim=1)

                # append idx, scores, label and embeding
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist(),
                                            sphere_idx.cpu().data.numpy().tolist(),
                                            embed.cpu().data.numpy().tolist()))

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\t Evaluation Batch', Size=40, erase=True)

        # compute AUCs
        index, label, score, sphere_index, embed = zip(*idx_label_score)
        label, score = np.array(label), np.array(score)
        auc = roc_auc_score(label, score)

        if save_tSNE:
            embed = np.array(embed)
            embed = TSNE(n_components=2).fit_transform(embed)
            idx_label_score = list(zip(index, label.tolist(), score.tolist(), sphere_index, embed.tolist()))

        self.eval_time = time.time() - start_time
        self.eval_scores = idx_label_score
        self.eval_auc = auc

        if print_to_logger:
            logger.info(f'Evaluation Time : {self.eval_time}')
            logger.info(f'Evaluation AUC : {self.eval_auc:.3%}')
            logger.info('Finished Evaluating the DMSAD.')

        if return_auc:
            return auc

    def initialize_centers(self, loader, net, eps=0.1):
        """
        Initialize the multiple centers using the K-Means algorithm on the
        embedding of all the normal samples.
        ----------
        INPUT
            |---- loader (torch.utils.data.Dataloader) the loader of the data.
            |---- net (nn.Module) the DMSAD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- eps (float) minimal value for center coordinates, to avoid
            |           center too close to zero.
        OUTPUT
            |---- None
        """
        # Get sample embedding
        repr = []
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data
                input, _, _, semi_label, _ = data
                input = input.to(self.device).float()
                semi_label = semi_label.to(self.device)
                # keep only normal samples
                input = input[semi_label != -1]
                # get embdeding of batch
                _, embed = net(input)
                repr.append(embed)

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\tBatch', Size=40, erase=True)

            repr = torch.cat(repr, dim=0).cpu().numpy()

        # Apply Kmeans algorithm on embedding
        kmeans = KMeans(n_clusters=self.n_sphere_init).fit(repr)
        self.c = torch.tensor(kmeans.cluster_centers_).to(self.device)

        # check if c_i are epsilon too close to zero to avoid them to be trivialy matched to zero
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps

    def set_radius(self, loader, net):
        """
        compute radius as 1-gamma quatile of normal sample distance to center.
        Then anomaly score is ||net(x) - c_j||^2 - R_j^2 <--- negative if in, positive if out.
        ----------
        INPUT
            |---- loader (torch.utils.data.Dataloader) the loader of the data.
            |---- net (nn.Module) the DMSAD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- None
        """
        dist_list = [[] for _ in range(self.c.shape[0])] # initialize N_sphere lists
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data
                input, _, _, semi_label, _ = data
                input = input.to(self.device).float()
                semi_label = semi_label.to(self.device)
                # keep only normal samples
                input = input[semi_label != -1]
                # get embdeding of batch
                _, embed = net(input)

                # get the closest sphere and count the number of normal samples per sphere
                dist, idx = torch.min(torch.norm(self.c.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=2), dim=1)
                for i, d in zip(idx, dist):
                    dist_list[i].append(d)

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\tBatch', Size=40, erase=True)

            # compute the radius as 1-gamma quantile of the normal distances of each spheres
            self.R = torch.zeros(self.c.shape[0], device=self.device)
            for i, dist in enumerate(dist_list):
                dist = torch.stack(dist, dim=0)
                self.R[i] = torch.kthvalue(dist, k=int((1 - self.gamma) * dist.shape[0]))[0]
