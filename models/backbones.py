import torch
from torch import nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3, resnet18
from models.block import *
import torch.nn.functional as F
import numpy as np
from monai.networks.nets import UNet
from torchvision.models import resnet, resnet18, resnet34
from models.block import gather


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_c, out_c, batchnorm=True, activation=True, k=3):
        super().__init__()
        if k == 3:
            self.conv = conv3x3(in_c, out_c)
        elif k == 1:
            self.conv = conv1x1(in_c, out_c)
        else:
            raise ValueError()

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_c)
        else:
            self.bn = nn.Identity()

        if activation:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_c, out_c, upsampling_method):
        super().__init__()
        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1),
            )

    def forward(self, x):
        return self.upsample(x)


class ResNetEncoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.inconv = ConvBlock(chan_in, 64, k=3)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer1
        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)  # -> 64 x H/2 x W/2
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)

        return x

class ResNetEncoder_modified(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        resnet2 = resnet18(pretrained=pretrained)
        self.inconv = ConvBlock(chan_in, 64, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet2.layer1
        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)  # -> 64 x H/2 x W/2
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)

        return x

class UNetEncoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.unet = UNet(
            spatial_dims=2,
            in_channels=chan_lifting,
            out_channels=chan_out,
            channels=(128, 256, 512),
            strides=(2, 2),
            num_res_units=3)
        # self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)  # -> 64 x H/2 x W/2
        x = self.unet(x)  # -> 64 x H/2 x W/2
        # x = self.outconv(x)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

# pure resnet18 as backbone
class Res18Encoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3

        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)

        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)
        x = self.layer2(x) # -> 128 x H/2 x W/2
        x = self.layer3(x) # -> 256 x H/4 x W/4
        x = self.up1(x)    # -> 128 x H/2 x W/2
        x = self.up2(x)    # -> 64 x H x W
        x = self.outconv(x)

        return x

# pure resnet18 as backbone
class Res34Encoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet34(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3

        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)

        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)
        x = self.layer2(x) # -> 128 x H/2 x W/2
        x = self.layer3(x) # -> 256 x H/4 x W/4
        x = self.up1(x)    # -> 128 x H/2 x W/2
        x = self.up2(x)    # -> 64 x H x W
        x = self.outconv(x)

        return x


# combine resnet18 with Unet
class URes18Encoder1(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3

        self.up1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # -> 128 x H/2 x W/2
        x3 = self.layer3(x2) # -> 256 x H/4 x W/4
        x3_up = F.interpolate(x3, scale_factor=2., mode='bilinear', align_corners=True)    # -> 256 x H/2 x W/2
        x = torch.cat((x3_up, x2), dim = 1)
        x = self.up1(x)
        x2_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)   # -> 128 x H x W
        x = torch.cat((x2_up, x1), dim=1)
        x = self.up2(x)
        x = self.outconv(x)

        return x

# combine resnet18 with Unet
class URes18Encoder2(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3

        self.up1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # -> 128 x H/2 x W/2
        x3 = self.layer3(x2) # -> 256 x H/4 x W/4
        x3_up = F.interpolate(x3, scale_factor=2., mode='bilinear', align_corners=True)    # -> 256 x H/2 x W/2
        x = torch.cat((x3_up, x2), dim = 1)
        x = self.up1(x)
        x2_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)   # -> 128 x H x W
        x = torch.cat((x2_up, x1), dim=1)
        x = self.up2(x)
        x = self.outconv(x)

        return x


# combine resnet18 with Unet (3 subsampling)
class URes18EncoderD1(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3
        self.layer4 = resnet1.layer4

        self.up1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            # nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.PReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.Conv2d(128, 128, 3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.PReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.PReLU(),
        )


        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # -> 128 x H/2 x W/2
        x3 = self.layer3(x2) # -> 256 x H/4 x W/4
        x4 = self.layer4(x3) # -> 512 x H/8 x W/8

        x4_up = F.interpolate(x4, scale_factor=2., mode='bilinear', align_corners=True)    # -> 512 x H/4 x W/4
        x = torch.cat((x4_up, x3), dim = 1)
        x = self.up1(x)
        x3_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)    # -> 256 x H/2 x W/2
        x = torch.cat((x3_up, x2), dim = 1)
        x = self.up2(x)
        x2_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)   # -> 128 x H x W
        x = torch.cat((x2_up, x1), dim=1)
        x = self.up3(x)
        x = self.outconv(x)

        return x

#
class URes18EncoderD2(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet18(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3
        self.layer4 = resnet1.layer4

        self.up1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )


        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # -> 128 x H/2 x W/2
        x3 = self.layer3(x2) # -> 256 x H/4 x W/4
        x4 = self.layer4(x3) # -> 512 x H/8 x W/8

        x4_up = F.interpolate(x4, scale_factor=2., mode='bilinear', align_corners=True)    # -> 512 x H/4 x W/4
        x = torch.cat((x4_up, x3), dim = 1)
        x = self.up1(x)
        x3_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)    # -> 256 x H/2 x W/2
        x = torch.cat((x3_up, x2), dim = 1)
        x = self.up2(x)
        x2_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)   # -> 128 x H x W
        x = torch.cat((x2_up, x1), dim=1)
        x = self.up3(x)
        x = self.outconv(x)

        return x


class URes34EncoderD(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet34(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3
        self.layer4 = resnet1.layer4

        self.up1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            # nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.PReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.Conv2d(128, 128, 3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.PReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.PReLU(),
        )


        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # -> 128 x H/2 x W/2
        x3 = self.layer3(x2) # -> 256 x H/4 x W/4
        x4 = self.layer4(x3) # -> 512 x H/8 x W/8

        x4_up = F.interpolate(x4, scale_factor=2., mode='bilinear', align_corners=True)    # -> 512 x H/4 x W/4
        x = torch.cat((x4_up, x3), dim = 1)
        x = self.up1(x)
        x3_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)    # -> 256 x H/2 x W/2
        x = torch.cat((x3_up, x2), dim = 1)
        x = self.up2(x)
        x2_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)   # -> 128 x H x W
        x = torch.cat((x2_up, x1), dim=1)
        x = self.up3(x)
        x = self.outconv(x)

        return x


# combine resnet34 with Unet
class URes34Encoder2(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet1 = resnet34(pretrained=pretrained)
        chan_lifting = 64
        self.inconv = ConvBlock(chan_in, chan_lifting, k=3)
        self.layer1 = resnet1.layer1
        self.layer2 = resnet1.layer2
        self.layer3 = resnet1.layer3

        self.up1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # -> 128 x H/2 x W/2
        x3 = self.layer3(x2) # -> 256 x H/4 x W/4
        x3_up = F.interpolate(x3, scale_factor=2., mode='bilinear', align_corners=True)    # -> 256 x H/2 x W/2
        x = torch.cat((x3_up, x2), dim = 1)
        x = self.up1(x)
        x2_up = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)   # -> 128 x H x W
        x = torch.cat((x2_up, x1), dim=1)
        x = self.up2(x)
        x = self.outconv(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(self, chan_in, chan_out, non_linearity, pretrained=False):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        resnet.inplanes = chan_in
        self.layer1 = resnet._make_layer(BasicBlock, 64, 2)
        resnet.inplanes = 64
        self.layer2 = resnet._make_layer(BasicBlock, 64, 2)

        self.upconv1 = UpConv(64, 64, "bilinear")
        self.outconv = ConvBlock(64, chan_out, batchnorm=False, activation=False)

        if non_linearity is None:
            self.non_linearity = nn.Identity()
        else:
            self.non_linearity = non_linearity

        # Initialize all the new layers
        self.resnet_init()

    def resnet_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.layer1(x)  # -> 128 x H/4 x W/4
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)  # -> C_out x H x W
        x = self.non_linearity(x)
        return x


class KPFCN(nn.Module):

    def __init__(self, config):
        super(KPFCN, self).__init__()

        ############
        # Parameters
        ############
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim

        #####################
        # List Encoder blocks
        #####################
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architectures):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architectures):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architectures[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architectures[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        print()

    def forward(self, batch, phase = 'encode'):
        # Get input features

        x = batch['features'].clone().detach()
        # 1. joint encoder part
        self.skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                self.skip_x.append(x)
            x = block_op(x, batch)  # [N,C]

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, self.skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # features = F.normalize(x, p=2, dim=-1)
        return x


class Fusion_CATL(nn.Module):
    def __init__(self, feat_dim):
        super(Fusion_CATL, self).__init__()
        self.feat_dim = feat_dim
        self.map = nn.Linear(2*feat_dim, feat_dim)


    def forward(self, feats_I, feats_P, batch):
        length_split = batch['stack_lengths'][0]

        B, C, H, W = feats_I[0].shape
        feats_I_src = feats_I[0].reshape(B, C, H*W)
        feats_I_tgt = feats_I[1].reshape(B, C, H*W)

        feats_P = torch.cat((feats_P, torch.zeros_like(feats_P[:1, :])), 0)
        feats_p2i = feats_P[batch['p2i_list'][0].squeeze()]

        feats_p2i_src = []
        feats_p2i_tgt = []
        for i in range(2*B):
            if i%2 == 0:
                feats_p2i_src.append(feats_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                feats_p2i_tgt.append(feats_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
        feats_src = torch.cat([feats_I_src.permute(0, 2, 1), torch.vstack(feats_p2i_src)], -1)
        feats_tgt = torch.cat([feats_I_tgt.permute(0, 2, 1), torch.vstack(feats_p2i_tgt)], -1)

        return [self.map(feats_src), self.map(feats_tgt)]



class Fusion_CAT(nn.Module):
    def __init__(self, feat_dim):
        super(Fusion_CAT, self).__init__()
        self.feat_dim = feat_dim
        self.map = nn.Linear(2*feat_dim, feat_dim)


    def forward(self, feats_I, feats_P, batch):
        length_split = batch['stack_lengths'][0]

        B, C, H, W = feats_I[0].shape
        feats_I_src = feats_I[0].reshape(B, C, H*W)
        feats_I_tgt = feats_I[1].reshape(B, C, H*W)

        feats_P = torch.cat((feats_P, torch.zeros_like(feats_P[:1, :])), 0)
        feats_p2i = feats_P[batch['p2i_list'][0].squeeze()]

        feats_p2i_src = []
        feats_p2i_tgt = []
        for i in range(2*B):
            if i%2 == 0:
                feats_p2i_src.append(feats_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                feats_p2i_tgt.append(feats_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
        return [torch.vstack(feats_p2i_src), torch.vstack(feats_p2i_tgt)]
        # return [feats_I_src.permute(0, 2, 1), feats_I_tgt.permute(0, 2, 1)]







