import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# Discriminator
########################################

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, init_block):
        super().__init__()
        if init_block:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, padding, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, padding, padding_mode="reflect"),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
    def forward(self, x):
        return self.conv_block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, filter_size=[64, 128, 256, 512]):
        super().__init__()

        layers = []
        layers.append(DiscBlock(in_channels, filter_size[0], stride=2, padding=1, init_block=True))
        in_channels = filter_size[0]

        for size in filter_size[1:]:
            layers.append(DiscBlock(in_channels, size, stride=1 if size == filter_size[-1] else 2, padding=1, init_block=False))
            in_channels = size
        
        last_layer = nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode='reflect')
        layers.append(last_layer)

        self.disc = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.disc(x))

################################################################
#   Generator
# In paper, c7s1-k denotes a 7x7 convolution-InstanceNorm-ReLU layer with k filters and stride 1. 
# dk denotes 3x3 conv-InstanceNorm-RelU layer with k filters and stride 2. Reflection padding was used.
# Rk denotes residual block that contains two 3x3 conv layers with the same number of filters on both layer.
#uk denotes a 3x3 fractional-strided-conv-InstanceNorm-ReLU layer with k layers and stride 1/2.
# THe network with 6 residual blocks consists of => c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
################################################################
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, kernel_size=3, stride=1, padding=1, output_padding=0):
        super().__init__()
        if down == True:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act==True else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.residual = nn.Sequential(
            GenBlock(in_channels,in_channels,down=True,use_act=True,kernel_size=3,stride=1,padding=1),
            GenBlock(in_channels,in_channels,down=True,use_act=False,kernel_size=3,stride=1,padding=1),
        )
    
    def forward(self,x):
        return x + self.residual(x)

class Generator(nn.Module):
    def __init__(self,img_channels,filter_size = 64,num_residuals=9): # number of residuals is 6 for 128x128 image and 9 for 256x256 image
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(img_channels,filter_size,kernel_size=7,stride=1,padding=3,padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                GenBlock(filter_size,filter_size*2,kernel_size=3,stride=2,padding=1),
                GenBlock(filter_size*2,filter_size*4,kernel_size=3,stride=2,padding=1),
            ]
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filter_size*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                GenBlock(filter_size*4,filter_size*2,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
                GenBlock(filter_size*2,filter_size,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
            ]
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(filter_size,img_channels,kernel_size=7,stride=1,padding=3,padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.first_block(x)
        for block in self.down_blocks:
            x = block(x)
            
        x = self.res_blocks(x)
        for block in self.up_blocks:
            x = block(x)
            
        x = self.last_block(x)
            
        return torch.tanh(x)
