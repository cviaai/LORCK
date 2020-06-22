import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from UNet_PD import UNet_Dilated


class UNet_Dilated_2levels_config1(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.model = UNet_Dilated(1, n_class)
        self.model.model.conv_down[0].conv_layers = nn.ModuleList(
                [nn.Conv2d(1, 32, 10, 
                           padding=4,
                           dilation=1, 
                           bias=False),
                nn.Conv2d(32, 32, 4, padding=5, dilation=3)])
    
        self.model.model.conv_down[1].conv_layers = nn.ModuleList(
                [nn.Conv2d(32, 128, 10 ,
                           stride=2, dilation=2, 
                           padding=5 + 4 - 1, 
                           bias=False),
                nn.Conv2d(128, 128, 4, padding=5, dilation=3)])
        
    def forward(self, x):
        x = self.model(x)
   
        return x



class UNet_Dilated_2levels_config2(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.model = UNet_Dilated(1, n_class)
        self.model.model.conv_down[0].conv_layers = nn.ModuleList(
                [nn.Conv2d(1, 32, 10, 
                           padding=4,
                           dilation=1, 
                           bias=False),
                nn.Conv2d(32, 32, 4, padding=5, dilation=3)])
    
        self.model.model.conv_down[1].conv_layers = nn.ModuleList(
                [nn.Conv2d(32, 128, 20 ,
                           stride=2, dilation=1, 
                           padding=8, 
                           bias=False),
                nn.Conv2d(128, 128, 4, padding=5, dilation=3)])
        
    def forward(self, x):
        x = self.model(x)
   
        return x


class UNet_Dilated_2levels_config3(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.model = UNet_Dilated(1, n_class)
        self.model.model.conv_down[0].conv_layers = nn.ModuleList(
                [nn.Conv2d(1, 32, 10, 
                           padding=4,
                           dilation=1, 
                           bias=False),
                nn.Conv2d(32, 32, 4, padding=5, dilation=3)])
    
        self.model.model.conv_down[1].conv_layers = nn.ModuleList(
                [nn.Conv2d(32, 128, 10 ,
                           stride=2, 
                           padding=5,
                           dilation=1, 
                           bias=False),
                nn.Conv2d(128, 128, 4, padding=4, dilation=3)])
        
    def forward(self, x):
        x = self.model(x)
        
        return x


