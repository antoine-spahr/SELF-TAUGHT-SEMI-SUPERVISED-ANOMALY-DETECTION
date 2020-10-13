import torch
import torch.nn as nn

from src.models.networks.Network_Modules import ResNet18_Encoder, ResNet18_Decoder, MLPHead

class Encoder(nn.Module):
    """
    Module defining an Encoder composed of a ResNet18 convolutional network
    followed by a MLP Projection head.
    """
    def __init__(self, MLP_Neurons_layer=[512,256,128]):
        """
        Build a Encoder module.
        ----------
        INPUT
            |---- Neurons_layer (list of int) list of number of neurones at each
            |           layers of the projection head.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer)

    def forward(self, x):
        """
        Forward pass of the Encoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x C x H x W).
        OUTPUT
            |---- h (torch.Tensor) the output of the convolutional network (B x 512).
            |---- z (torch.Tensor) the output of the projection head (B x Neurons_layer[-1]).
        """
        h = self.encoder(x)
        z = self.head_enc(h)

        return h, z

class AE_net(nn.Module):
    """
    Module defining an Auto-encoder composed of a ResNet18 convolutional network
    followed by a MLP Projection head. Then recostructed by a symetric MLP and
    mirrored ResNet18 decoder.
    """
    def __init__(self, MLP_Neurons_layer_enc=[512,256,128], MLP_Neurons_layer_dec=[128,256,512], output_channels=3):
        """
        Build an Auto-encoder module.
        ----------
        INPUT
            |---- Neurons_layer_enc (list of int) list of number of neurones at each
            |           layers of the encoding projection head.
            |---- Neurons_layer_dec (list of int) list of number of neurones at each
            |           layers of the decoding projection head.
            |---- output_channels (int) the nubmer of channel of the input.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer_enc)
        self.head_dec = MLPHead(MLP_Neurons_layer_dec)
        self.decoder = ResNet18_Decoder(output_channels=output_channels)

    def forward(self, x):
        """
        Forward pass of the Encoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x C x H x W).
        OUTPUT
            |---- h (torch.Tensor) the output of the convolutional network (B x 512).
            |---- z (torch.Tensor) the output of the projection head (B x Neurons_layer[-1]).
            |---- rec (torch.Tensor) the reconstructed sample (B x C x H x W).
        """
        h = self.encoder(x)
        z = self.head_enc(h)
        # reconstruct
        rec = self.decoder(self.head_dec(z))

        return h, z, rec
