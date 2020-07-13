import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.utils

class DownResBlock(nn.Module):
    """
    Residual Block for the ResNet18 (without bottleneck).
    """
    def __init__(self, in_channel, out_channel, downsample=False):
        """
         in ->[Conv3x3]->[BN]->[ReLU]->[Conv3x3]->[BN]-> + -> out
            |                                            |
            |________________[downLayer]_________________|
        ----------
        INPUT
            |---- in_channel (int) the number of input channels.
            |---- out_channel (int) the number of output channels.
            |---- downsample (bool) whether the block downsample the input.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.downsample = downsample
        # convolution 1
        conv1_stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=conv1_stride, \
                               bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel, affine=False)
        self.relu = nn.ReLU(inplace=True)
        # convolution 2
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel, affine=False)
        # the module for the pass through
        self.downLayer = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                                       nn.BatchNorm2d(out_channel, affine=False))

    def forward(self, x):
        """
        Forward pass of the Residual Block.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W) with C = in_channel
        OUTPUT
            |---- out (torch.Tensor) the output tensor (B x C x H' x W') with
            |           C = out_channel. H and W are changed if the stride is
            |           bigger than one.
        """
        # get the residual
        if self.downsample:
            residual = self.downLayer(x)
        else:
            residual = x

        # convolution n°1 with potential down sampling
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # convolution n°2
        out = self.conv2(out)
        out = self.bn2(out)
        # sum convolution with shortcut
        out += residual
        out = self.relu(out)
        return out

class UpResBlock(nn.Module):
    """
    Up Residual Block for the ResNet18-like decoder (without bottleneck).
    """
    def __init__(self, in_channel, out_channel, upsample=False):
        """
         in ->[Conv3x3]->[BN]->[ReLU]->[Conv3x3 / ConvTransp3x3]->[BN]-> + -> out
            |                                                            |
            |__________________________[upLayer]_________________________|
        ----------
        INPUT
            |---- in_channel (int) the number of input channels.
            |---- out_channel (int) the number of output channels.
            |---- upsample (bool) whether the block upsample the input.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.upsample = upsample
        # convolution 1
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel, affine=False)
        self.relu = nn.ReLU(inplace=True)

        # convolution 2. If block upsample
        if upsample:
            self.conv2 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                       nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False, dilation=1))
        else:
            self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel, affine=False)

        # module for the pass-through
        self.upLayer = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                     nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(out_channel, affine=False))

    def forward(self, x):
        """
        Forward pass of the UP Residual Block.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W) with C = in_channel
        OUTPUT
            |---- out (torch.Tensor) the output tensor (B x C x H' x W') with
            |           C = out_channel. H and W are changed if the stride is
            |           bigger than one.
        """
        # convolution n°1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # convolution n°2 or transposed convolution
        out = self.conv2(out)
        out = self.bn2(out)
        # get the residual
        if self.upsample:
            residual = self.upLayer(x)
        else:
            residual = x
        # sum convolution with shortcut
        out += residual
        out = self.relu(out)
        return out

class ResNet18_Encoder(nn.Module):
    """
    Combine multiple Residual block to form a ResNet18 up to the Average poolong
    layer. The size of the embeding dimension can be different than the one from
    ResNet18.
    """
    def __init__(self):
        """
        Build the Encoder from the layer's specification. The encoder is composed
        of an initial 7x7 convolution that halves the input dimension (h and w)
        followed by several layers of residual blocks. Each layer is composed of
        k Residual blocks. The first one reduce the input height and width by a
        factor 2 while the number of channel is increased by 2.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        # First convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual layers
        self.layer1 = nn.Sequential(DownResBlock(64, 64, downsample=False),
                                    DownResBlock(64, 64, downsample=False))
        self.layer2 = nn.Sequential(DownResBlock(64, 128, downsample=True),
                                    DownResBlock(128, 128, downsample=False))
        self.layer3 = nn.Sequential(DownResBlock(128, 256, downsample=True),
                                    DownResBlock(256, 256, downsample=False))
        self.layer4 = nn.Sequential(DownResBlock(256, 512, downsample=True),
                                    DownResBlock(512, 512, downsample=False))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        """
        Forward pass of the Encoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W). The input
            |           image can be grayscale or RGB. If it's grayscale it will
            |           be converted to RGB by stacking 3 copy.
        OUTPUT
            |---- out (torch.Tensor) the embedding of the image in dim 512.
        """
        # if grayscale (1 channel) convert to RGB by duplicating on 3 channel
        # assuming shape : (... x C x H x W)
        if x.shape[-3] == 1:
            x = torch.cat([x]*3, dim=1)
        # first 1x1 convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 4 layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Average pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNet18_Decoder(nn.Module):
    """
    Combine multiple Up Residual Blocks to form a ResNet18 like decoder.
    """
    def __init__(self, output_channels=3):
        """
        Build the ResNet18-like decoder. The decoder is composed of a Linear layer.
        The linear layer is interpolated (bilinear) to 512x16x16 which is then
        processed by several Up-layer of Up Residual Blocks. Each Up-layer is
        composed of k Up residual blocks. The first ones are without up sampling.
        The last one increase the input size (h and w) by a factor 2 and reduce
        the number of channels by a factor 2.
        ---------
        INPUT
            |---- output_size (tuple) the decoder output size. (C x H x W)
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)

        self.interp_layer = nn.Upsample(size=(16,16), mode='bilinear', align_corners=True)
        self.uplayer1 = nn.Sequential(UpResBlock(512, 512, upsample=False),
                                      UpResBlock(512, 256, upsample=True))
        self.uplayer2 = nn.Sequential(UpResBlock(256, 256, upsample=False),
                                      UpResBlock(256, 128, upsample=True))
        self.uplayer3 = nn.Sequential(UpResBlock(128, 128, upsample=False),
                                      UpResBlock(128, 64, upsample=True))
        self.uplayer4 = nn.Sequential(UpResBlock(64, 64, upsample=False),
                                      UpResBlock(64, 64, upsample=True))

        self.uplayer_final = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                           nn.Conv2d(64, output_channels, kernel_size=1, stride=1, bias=False))
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the decoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x embed_dim).
        OUTPUT
            |---- out (torch.Tensor) the reconstructed image (B x C x H x W).
        """
        x = x.view(-1, 512, 1, 1)
        x = self.interp_layer(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_final(x)
        x = self.final_activation(x)
        return x

class MLPHead(nn.Module):
    """
    Module defining the MLP Projection head.
    """
    def __init__(self, Neurons_layer=[512,256,128]):
        """
        Build a MLP projection head module.
        ----------
        INPUT
            |---- Neurons_layer (list of int) list of number of neurones at each
            |           layers of the projection head.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.fc_layers = nn.ModuleList(nn.Linear(in_features=n_in, out_features=n_out) for n_in, n_out in zip(Neurons_layer[:-1], Neurons_layer[1:]))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the projection head.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x Neurons_layer[0]).
        OUTPUT
            |---- x (torch.Tensor) the compressed image (B x Neurons_layer[-1]).
        """
        for linear in self.fc_layers[:-1]:
            x = self.relu(linear(x))
        x = self.fc_layers[-1](x)
        return x
