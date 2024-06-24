import torch
import torch.nn as nn
from .resnet import ResNetEncoder, ResnetDecoder, ResNetBasicBlock, ResNetBottleNeckBlock
from .unet import Unet_Encoder, Unet_Decoder, Unet_Classification_head


class Segmentation_Network_full(nn.Module): # 477,155 params, used in paper
    def __init__(self, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv3d(1, 32, kernel_size = (2,2,2), stride = (2,2,2)),
            nn.ReLU(),
            nn.Conv3d(32,64, kernel_size = (2,2,2), stride = (2,2,2)),
            nn.ReLU(),
            
            nn.Conv3d(64, 128, kernel_size = (2,2,2), stride = (2,2,2)),
            nn.ReLU(),
            nn.Conv3d(128,256, kernel_size = (2,2,2), stride = (2,2,2)),
            nn.ReLU(),
            
            nn.Conv3d(256,256, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.ReLU(),
            nn.Conv3d(256,256, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.ReLU(),
            
            nn.Conv3d(256,n_classes, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.Sigmoid(),
            nn.Flatten()
        )
    
    def forward(self, data):
        return self.network(data)

class UNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128, 256, 512,), dec_chs=(512, 256, 128, 64), n_classes=3):
        super().__init__()
        self.encoder     = Unet_Encoder(enc_chs)
        self.decoder     = Unet_Decoder(dec_chs)
        self.head        = Unet_Classification_head(n_classes)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out, enc_ftrs[-1])
        return out
    


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def resnet1(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 1, 1])

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])