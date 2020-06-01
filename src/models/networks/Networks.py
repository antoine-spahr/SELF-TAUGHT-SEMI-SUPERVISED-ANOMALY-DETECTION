import torch
import torch.nn as nn

from src.models.networks.Network_Modules import ResNet18_Encoder, ResNet18_Decoder, MLPHead

class Encoder(nn.Module):
    """

    """
    def __init__(self, MLP_Neurons_layer=[512,256,128]):
        """

        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer)

    def forward(self, x):
        """

        """
        h = self.encoder(x)
        z = self.head_enc(h)

        return h, z

class AE_net(nn.Module):
    """

    """
    def __init__(self, MLP_Neurons_layer_enc=[512,256,128], MLP_Neurons_layer_dec=[128,256,512], output_channels=3):
        """

        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer_enc)
        self.head_dec = MLPHead(MLP_Neurons_layer_dec)
        self.decoder = ResNet18_Decoder(output_channels=output_channels)

    def forward(self, x):
        """

        """
        h = self.encoder(x)
        z = self.head_enc(h)
        # reconstruct
        rec = self.decoder(self.head_dec(z))

        return h, z, rec
