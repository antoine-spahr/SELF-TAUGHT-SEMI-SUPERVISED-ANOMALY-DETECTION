import torch
import json
import sys

from src.models.optim.AE_trainer import AE_trainer
from src.models.optim.DSAD_trainer import DSAD_trainer

class AE_DSAD:
    """
    Define a Deep SAD encoder with a AE pretraining of the encoder.
    """
    def __init__(self, ae_net, AD_net, eta=1.0):
        """
        Build a AE_DSAD.
        ----------
        INPUT
            |---- ae_net (nn.Module) the network to use for the AE learning.
            |---- AD_net (nn.Module) the DSAD network for anomaly detection.
            |---- eta (float) the semi-supervised importance in the DSAD loss.
        OUTPUT
            |---- None
        """
        self.ae_net = ae_net
        self.AD_net = AD_net
        self.c = None
        self.eta = eta

        self.ae_trainer = None
        self.AD_trainer = None

        self.results = {
            'AE':{
                'train':{
                    'time': None,
                    'loss': None
                },
                'valid':{
                    'embedding': None
                },
                'test':{
                    'embedding': None
                }
            },
            'AD':{
                'train':{
                    'time': None,
                    'loss': None
                },
                'valid':{
                    'time': None,
                    'auc': None,
                    'scores': None
                },
                'test':{
                    'time': None,
                    'auc': None,
                    'scores': None
                }
            }
        }

    def train_AE(self, dataset, valid_dataset=None, n_epoch=100, batch_size=32,
                     lr=1e-3, weight_decay=1e-6, lr_milestone=(), n_job_dataloader=0,
                     device='cuda', print_batch_progress=False):
        """
        Pretrain the encoder with AE learning.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) The dataset on which to train
            |           the ae_net. It must return the input image the mask and
            |           the semi-supervised label.
            |---- n_epoch (int) the number of epoch.
            |---- batch_size (int) the batch_size to use.
            |---- lr (float) the learning rate.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- lr_milestone (tuple) the lr update steps.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        self.ae_trainer = AE_trainer(n_epoch=n_epoch, batch_size=batch_size,
                         lr=lr, weight_decay=weight_decay, lr_milestone=lr_milestone,
                         n_job_dataloader=n_job_dataloader, device=device,
                         print_batch_progress=print_batch_progress)
        # train SimCLR
        self.ae_net = self.ae_trainer.train(self.ae_net, dataset, valid_dataset=valid_dataset)
        # get results
        self.results['AE']['train']['time'] = self.ae_trainer.train_time
        self.results['AE']['train']['loss'] = self.ae_trainer.train_loss

    def evaluate_AE(self, dataset, batch_size=32, n_job_dataloader=0,
                    device='cuda', print_batch_progress=False, set='test'):
        """
        Evaluate the AE to get the embedding.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) The dataset on which to evaluate
            |           the ae_net. It must return the input, label, mask, semi-
            |           supervised label and the index.
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
            |---- set (str) The nature of the set evaluated. Must be either 'valid' or 'test'.
        OUTPUT
            |---- None
        """
        assert set in ['valid', 'test'], f'Invalid set provided : {set} was given. Expected either valid or test.'

        if self.ae_trainer is None:
            self.ae_trainer = AE_trainer(batch_size=batch_size,
                        n_job_dataloader=n_job_dataloader, device=device,
                        print_batch_progress=print_batch_progress)

        # Evaluate model
        self.ae_trainer.evaluate(self.ae_net, dataset, save_tSNE=True,
                                   return_auc=False, print_to_logger=True)
        # get results
        self.results['AE'][set]['embedding'] = self.ae_trainer.eval_repr

    def save_ae_net(self, export_fn):
        """
        Save the AE model.
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save({'ae_net_dict': self.ae_net.state_dict()}, export_fn)

    def load_ae_net(self, import_fn, map_location='cpu'):
        """
        Load the AE model.
        ----------
        INPUT
            |---- import_fn (str) path where to get the model.
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        model = torch.load(import_fn, map_location=map_location)
        self.ae_net.load_state_dict(model['ae_net_dict'])

    def transfer_encoder(self):
        """
        Transfer the weight of the encoder learnt by AE learning to the
        encoder of DSAD.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        # get both encoder state dicts
        AD_encoder_dict = self.AD_net.encoder.state_dict()
        AE_encoder_dict = self.ae_net.encoder.state_dict()
        # keep common keys
        new_encoder_dict = {k:v for k, v in AE_encoder_dict.items() if k in AD_encoder_dict}
        # update classifer network encoder weights with the ae_net encoder ones.
        AD_encoder_dict.update(new_encoder_dict)
        self.AD_net.encoder.load_state_dict(AD_encoder_dict)

    def train_AD(self, dataset, valid_dataset=None, n_epoch=100, batch_size=32,
                 lr=1e-3, weight_decay=1e-6, lr_milestone=(), n_job_dataloader=0,
                 device='cuda', print_batch_progress=False):
        """
        Train the encoder on the DSAD objective.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) The dataset on which to train
            |           the AD_net. It must return the input, label, mask, semi-
            |           supervised label and the index.
            |---- valid_dataset (torch.utils.data.Dataset) The dataset on which
            |           to validate the model at each epoch. No validation is
            |           performed if not provided.
            |---- n_epoch (int) the number of epoch.
            |---- batch_size (int) the batch_size to use.
            |---- lr (float) the learning rate.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- lr_milestone (tuple) the lr update steps.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        self.AD_trainer = DSAD_trainer(self.c, self.eta, n_epoch=n_epoch, batch_size=batch_size,
                         lr=lr, weight_decay=weight_decay, lr_milestone=lr_milestone,
                         n_job_dataloader=n_job_dataloader, device=device,
                         print_batch_progress=print_batch_progress)
        # Train classifer
        self.AD_net = self.AD_trainer.train(dataset, self.AD_net, valid_dataset=valid_dataset)
        # get results
        self.results['AD']['train']['time'] = self.AD_trainer.train_time
        self.results['AD']['train']['loss'] = self.AD_trainer.train_loss
        self.c = self.AD_trainer.c.cpu().data.numpy().tolist()

    def evaluate_AD(self, dataset, batch_size=32, n_job_dataloader=0,
                    device='cuda', print_batch_progress=False, set='test'):
        """
        Evaluate the encoder on the DSAD objective.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) The dataset on which to evaluate
            |           the AD_net. It must return the input, label, mask, semi-
            |           supervised label and the index.
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
            |---- set (str) The nature of the set evaluated. Must be either 'valid' or 'test'.
        OUTPUT
            |---- None
        """
        assert set in ['valid', 'test'], f'Invalid set provided : {set} was given. Expected either valid or test.'

        if self.AD_trainer is None:
            self.AD_trainer = DSAD_trainer(self.c, self.eta, batch_size=batch_size,
                        n_job_dataloader=n_job_dataloader, device=device,
                        print_batch_progress=print_batch_progress)

        # Evaluate model
        self.AD_trainer.evaluate(self.AD_net, dataset, return_auc=False, print_to_logger=True, save_tSNE=True)
        # get results
        self.results['AD'][set]['time'] = self.AD_trainer.eval_time
        self.results['AD'][set]['auc'] = self.AD_trainer.eval_auc
        self.results['AD'][set]['scores'] = self.AD_trainer.eval_scores

    def save_AD(self, export_fn):
        """
        Save the DSAD model (center and state dictionary).
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save({'c': self.c,
                    'AD_net_dict': self.AD_net.state_dict()}, export_fn)

    def load_AD(self, import_fn, map_location='cpu'):
        """
        Load the DSAD model from location.
        ----------
        INPUT
            |---- import_fn (str) path where to get the model.
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        model = torch.load(import_fn, map_location=map_location)
        self.c = model['c']
        self.AD_net.load_state_dict(model['AD_net_dict'])

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
