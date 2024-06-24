import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 2, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 2, padding=1)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Unet_Encoder(nn.Module):
    def __init__(self, chs=(1,64,128, 256, 512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool3d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Unet_Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i+1], 2, 1) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, tensor, output_size):
        left_1 = (tensor.shape[-1] - output_size.shape[-1]) // 2
        left_2 = (tensor.shape[-2] - output_size.shape[-2]) // 2
        left_3 = (tensor.shape[-3] - output_size.shape[-3]) // 2
        
        
        left_1 = left_1 if left_1 > 0 else 0
        left_2 = left_2 if left_2 > 0 else 0
        left_3 = left_3 if left_3 > 0 else 0
        
        right_1 = left_1 + output_size.shape[-1]
        right_2 = left_2 + output_size.shape[-2]
        right_3 = left_3 + output_size.shape[-3]
        
        return tensor[..., left_1:right_1, left_2:right_2, left_3:right_3]


class Unet_Classification_head(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.b_net = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size = (2,2,2), stride = (1,1,1)),
            nn.ReLU(),
            nn.Conv3d(256, 64, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.ReLU(),
            )
        
        self.o_net = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size = (4,4,4), stride = (2,2,2)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size = (3,3,3), stride = (1,1,1)),
            nn.ReLU(),
            )

        self.classifier = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size = (2,2,2), stride = (2,2,2)),
            nn.ReLU(),
            
            nn.Conv3d(128, 64, kernel_size = (2,2,2), stride = (1,1,1)),
            nn.ReLU(),
            
            nn.Conv3d(64, 64, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.ReLU(),
            
            nn.Conv3d(64, 64, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.ReLU(),
        
            nn.Conv3d(64, n_classes, kernel_size = (1,1,1), stride = (1,1,1)),
            nn.Sigmoid(),
            nn.Flatten()
        )
        
    def forward(self, x, enc_ftrs):
        enc_ftrs = self.b_net(enc_ftrs)
        x = self.o_net(x)
        x = torch.cat([x, enc_ftrs], dim=1)
        x = self.classifier(x)

        return x
