import torch
import json
import sys
import logging
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

from src.utils.utils import print_progessbar

class NearestNeighbors:
    """
    Define a model that classify normal/abnormal based on the nearest neighbors
    in an embedding space of a trained model.
    """
    def __init__(self, net, kmeans_reduction=False, batch_size=16, n_job_dataloader=0,
                 device='cuda', print_batch_progress=True):
        """
        Build a nearest neighbors model.
        ----------
        INPUT
            |---- net (nn.Module) The trained Encoder.
            |---- kmeans_reduction (bool) whether to simplify the data using kmeans.
            |---- batch_size (int) the batch_size to use.
            |---- n_job_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        self.net = net
        self.centers = None
        self.kmeans_reduction = kmeans_reduction

        self.device = device
        self.print_batch_progress = print_batch_progress
        self.batch_size = batch_size
        self.n_job_dataloader = n_job_dataloader

        self.results = {
            'KMean_centers': None,
            'valid': {
                'auc': None,
                'scores': None
            },
            'test': {
                'auc': None,
                'scores': None
            }
        }

    def train(self, dataset, n_cluster=500):
        """
        Train the nearest neighbors for the ablation study. The training consist
        in summarising the train data into n_cluster points obtained by Kmeans.
        The distance from normal points is obtained by Nearest neighbors with
        those centers.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is evaluated.
            |---- n_cluster (int) the number of center to summaries the training data.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()

        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.n_job_dataloader)

        # get representation of normal samples with net
        logger.info('Getting normal train sample representation for the Ablation Study.')
        repr = []
        self.net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data
                input, _, mask, semi_label, _ = data
                input = input.to(self.device).float()
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                # mask input and keep only normal samples
                input = (input * mask)[semi_label != -1]
                # get embdeding of batch
                embed = self.net(input)[0] # first element returned is the transfered representation
                repr.append(embed)

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\tBatch', Size=40, erase=True)

            repr = torch.cat(repr, dim=0).cpu().numpy()

        # Apply Kmeans algorithm on embedding
        if self.kmeans_reduction:
            logger.info(f'Performing KMeans Clustering to summaries the data in {n_cluster} points.')
            kmeans = KMeans(n_clusters=n_cluster).fit(repr)
            self.centers = torch.tensor(kmeans.cluster_centers_).to(self.device)
        else:
            logger.info(f'Using all the {repr.shape[0]} normal samples representations for the nearest neighbors.')
            self.centers = torch.tensor(repr).to(self.device)
        logger.info(f'{self.centers.shape[0]} train points successfully generated')

    def evaluate(self, dataset, n_neighbors=100, mode='test'):
        """
        Evaluate the representation classification capabilitities by using as score
         the mean distance to the n_neighbors nearest centers.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is evaluated.
            |---- n_neighbors (int) the number of nearest centers to consider.
            |---- mode (str) define the set used. Either validation or test.
        OUTPUT
            |---- None
        """
        assert mode in ['valid', 'test'], f'Invalid mode provided : {mode} was given. Expected either valid or test.'

        logger = logging.getLogger()
        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.n_job_dataloader)

        logger.info('Start Evaluating the Ablation Study.')
        idx_label_score = []

        self.net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data on device
                input, label, mask, semi_label, idx = data
                input = input.to(self.device).float()
                label = label.to(self.device)
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                idx = idx.to(self.device)

                # mask input
                input = input * mask

                # Embed input and compute anomaly  score
                embed = self.net(input)[0] # first element returned is the transfered representation

                # get nearest centers of samples and take mean(distance as scores)
                dist = torch.norm(self.centers.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=2)
                min_dist = dist.topk(n_neighbors, dim=1, largest=False)[0]
                score = torch.mean(min_dist, dim=1)

                # append idx, scores, label and embeding
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist()))

                if self.print_batch_progress:
                    print_progessbar(b, len(loader), Name='\t\t Evaluation Batch', Size=40, erase=True)

        # compute AUCs
        _, label, score = zip(*idx_label_score)
        label, score = np.array(label), np.array(score)
        auc = roc_auc_score(label, score)
        self.results[mode]['auc'] = auc
        self.results[mode]['score'] = idx_label_score

        logger.info(f'{mode.title()} AUC : {auc:.3%}')
        logger.info('Finished Evaluating the Ablation Study.')

    def save_results(self, export_fn):
        """
        Save the model results in JSON.
        ----------
        INPUT
            |---- export_fn (str) path where to get the results.
        OUTPUT
            |---- None
        """
        with open(export_fn, 'w') as fn:
            json.dump(self.results, fn)

    def save_model(self, export_fn):
        """
        Save the ablation model (centers).
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save({'centers': self.centers}, export_fn)
