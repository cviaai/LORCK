import torch
import torch.nn as nn

import sys
from UNet import UNet 

class UNet_Original(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet(in_channels=in_channels, 
                              out_channels=out_channels, 
                              depth=4, activation="relu",
                              channels_sequence=[32, 128, 256, 512], 
                              conv_type="double",
                              dilation=1)
    def forward(self, x):
        return self.model(x)
    
    
class UNet_Original_with_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet(in_channels=in_channels, 
                              out_channels=out_channels, 
                              depth=4, activation="prelu",
                              channels_sequence=[32, 128, 256, 512], 
                              conv_type="double",
                              dilation=1,
                              batchnorm=True)
    def forward(self, x):
        return self.model(x)

class UNet_Baseline(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet(in_channels=in_channels, 
                        out_channels=out_channels, 
                        depth=4, 
                        channels_sequence=[32, 128, 256, 512], 
                        conv_type="triple",
                        dilation=1, 
                        batchnorm=True,
                        residual_bottleneck=True,
                        downsample_type='conv_stride',
                        activation="prelu",
                        big_upsample=True,
                        advanced_bottleneck=True)
        
    def forward(self, x):
        return self.model(x)
    
    
class UNet_Dilated(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet(in_channels=in_channels, 
                        out_channels=out_channels, 
                        depth=4, 
                        channels_sequence=[32, 128, 256, 512], 
                        conv_type="double",
                        dilation=[[1, 1], [2, 2], [4, 4], [8, 8], [1, 1]], 
                        batchnorm=True,
                        residual_bottleneck=True,
                        downsample_type='conv_stride',
                        activation="prelu",
                        big_upsample=True,
                        advanced_bottleneck=True
                             )
    def forward(self, x):
        return self.model(x)

class UNet_ProgressiveDilated(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet(in_channels=in_channels, 
                        out_channels=out_channels, 
                        depth=4, 
                        channels_sequence=[32, 128, 256, 512], 
                        conv_type="triple",
                        dilation=[[1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 1, 1]], 
                        batchnorm=True,
                        residual_bottleneck=True,
                        downsample_type='conv_stride',
                        activation="prelu",
                        big_upsample=True,
                        advanced_bottleneck=True)
    def forward(self, x):
        return self.model(x)



