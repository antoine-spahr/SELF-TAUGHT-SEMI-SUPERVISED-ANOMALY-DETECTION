import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """
    Compute the MSE loss only on the masked region.
    """
    def __init__(self, reduction='mean'):
        """
        Loss Constructor.
        ----------
        INPUT
            |---- reduction (str) the reduction to use on the loss. ONLY 'mean' or 'none'.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input, target, mask):
        """
        Forard pass of the loss. The loss is computed only where the wask is non-null.
        ----------
        INPUT
            |---- input (torch.Tensor) the input tensor.
            |---- target (torch.Tensor) the target tensor.
            |---- mask (torch.Tensor) the binary mask defining where the loss is
            |           computed (loss computed where mask != 0).
        OUTPUT
            |---- loss (torch.Tensor) the masked MSE loss.
        """
        # compute loss where mask = 1
        loss = self.criterion(input * mask, target * mask)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / torch.sum(mask)
        return loss

class DeepSADLoss(nn.Module):
    """
    Implementation of the DeepSAD loss proposed by Lukas Ruff et al. (2019)
    """
    def __init__(self, c, eta, eps=1e-6):
        """
        Constructor of the DeepSAD loss.
        ----------
        INPUT
            |---- c (torch.Tensor) the center of the hypersphere as a multidimensional vector.
            |---- eta (float) control the importance given to known or unknonw
            |           samples. 1.0 gives equal weights, <1.0 gives more weight
            |           to the unknown samples, >1.0 gives more weight to the
            |           known samples.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.c = c
        self.eta = eta
        self.eps = eps

    def forward(self, input, semi_target):
        """
        Forward pass of the DeepSAD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere
            |           center. (must thus have the same dimension (B x c.dim)).
            |---- semi_target (torch.Tensor) the semi-supervized label (0 -> unknown ;
            |           1 -> known normal ; -1 -> knonw abnormal)
        OUTPUT
            |---- loss (torch.Tensor) the DeepSAD loss.
        """
        # distance between center c and the input
        dist = torch.norm(self.c - input, p=2, dim=1)
        # compute the loss depening on the semi-supervized label
        # keep distance if semi-label is 0 or 1 (normal sample or unknonw (assumed) normal)
        # inverse distance if semi-label = -1 (known abnormal)
        losses = torch.where(semi_target == 0, dist**2, self.eta * ((dist**2 + self.eps) ** semi_target.float()))
        loss = torch.mean(losses)
        return loss

class DMSADLoss(nn.Module):
    """
    Implementation of the DMSAD loss inspired by Ghafoori et al. (2020) and Ruff
    et al. (2020)
    """
    def __init__(self, eta, eps=1e-6):
        """
        Constructor of the DMSAD loss.
        ----------
        INPUT
            |---- eta (float) control the importance given to known or unknonw
            |           samples. 1.0 gives equal weights, <1.0 gives more weight
            |           to the unknown samples, >1.0 gives more weight to the
            |           known samples.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.eta = eta
        self.eps = eps

    def forward(self, input, c, semi_target):
        """
        Forward pass of the DMSAD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere.
            |           center.
            |---- c (torch.Tensor) the centers of the hyperspheres as a multidimensional matrix (Centers x Embdeding).
            |---- semi_target (torch.Tensor) the semi-supervized label (0 -> unknown ;
            |           1 -> known normal ; -1 -> knonw abnormal)
        OUTPUT
            |---- loss (torch.Tensor) the DMSAD loss.
        """
        # distance between the input and the closest center
        dist, _ = torch.min(torch.norm(c.unsqueeze(0) - input.unsqueeze(1), p=2, dim=2), dim=1) # dist and idx by batch
        # compute the loss
        losses = torch.where(semi_target == 0, dist**2, self.eta * ((dist**2 + self.eps) ** semi_target.float()))
        loss = torch.mean(losses)
        return loss

class InfoNCE_loss(nn.Module):
    """
    Normalized temperature scaled cross-entropy loss.
    (https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py)
    """
    def __init__(self, tau, batch_size, device='cuda'):
        """
        Build a InfoNCE loss modules.
        ----------
        INPUT
            |---- tau (float) the temperature hyperparameter.
            |---- batch_size (int) the size of the batch.
            |---- device (str) the device to use.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.negative_mask = self.get_neg_mask(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def get_neg_mask(self, batch_size):
        """
        Generate a mask of the negative pairs positions.
        ----------
        INPUT
            |---- batch_size (int) the size of the btach.
        OUTPUT
            |---- mask (torch.Tensor) the mask of negtaive positions (2 batch_size x 2 batch_size).
        """
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Forward pass of the infoNCE loss.
        ----------
        INPUT
            |---- z_i (torch.Tensor) the representation of the first transformed
            |           input of the batch (B x Embed).
            |---- z_j (torch.Tensor) the representation of the second transformed
            |           input of the batch (B x Embed).
        OUTPUT
            |---- loss (torch.Tensor) the infoNCE loss.
        """
        # concat both represenation to get a (2*Batch x Embed)
        p = torch.cat((z_i, z_j), dim=0)
        # Compute the similarity matrix between elements --> (2Batch x 2Batch)
        sim = self.similarity_f(p.unsqueeze(1), p.unsqueeze(0)) / self.tau
        # Get positive pair similarity --> diag of upper right and lower left quarter of sim
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # get positive and negative vector for the two batches
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*self.batch_size, 1)
        negative_samples = sim[self.negative_mask].reshape(2*self.batch_size, -1)
        # generate lables (the first element represent the correct pair --> zero is correct label)
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # Compute the CE loss
        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.batch_size)
        return loss
