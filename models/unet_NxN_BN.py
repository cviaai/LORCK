import torch
import torch.nn as nn

from UNet import UNet as UNet_new


class UNet_10x10(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet_new(in_channels=in_channels, 
                              out_channels=out_channels, 
                              depth=1, activation="prelu",
                              channels_sequence=[16], 
                              batchnorm=True,
                              conv_type="double",
                              dilation=1)
    def forward(self, x):
        return self.model(x)
    
class UNet_20x20(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet_new(in_channels=in_channels, 
                              out_channels=out_channels, 
                              depth=2, activation="prelu",
                              channels_sequence=[16, 32], 
                              batchnorm=True,
                              conv_type="double",
                              dilation=1)
    def forward(self, x):
        return self.model(x)
    
    
class UNet_40x40(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.model = UNet_new(in_channels=in_channels, 
                              out_channels=out_channels, 
                              depth=3, activation="prelu",
                              channels_sequence=[32, 64, 128], 
                              batchnorm=True,
                              conv_type="double",
                              dilation=1)
    def forward(self, x):
        return self.model(x)
    
    