import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from UNet_PD import UNet_Dilated

def double_conv(in_channels, out_channels, bias=True, kernel=[3, 3], padding=[1, 1], dilation=[1, 1]):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel[0], padding=padding[0], dilation=dilation[0], bias=bias),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel[1], padding=padding[1], dilation=dilation[1]),
        nn.PReLU()
    )  

class UNet_config1(nn.Module):
    def __init__(self, n_class, kernel_size=20):
        super().__init__()
        
        self.dconv_down1 = double_conv(1, 32, bias=False, kernel=[kernel_size, 4], 
                                       padding=[int(kernel_size/2)-1, 2])
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.dconv_down2 = double_conv(32, 128)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.dconv_down3 = double_conv(128, 256)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        
        self.dconv_down4 = double_conv(256, 512) 
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, n_class, 1)
        
        
    def forward(self, x):

        conv1 = self.dconv_down1(x)
        conv1 = self.bn1(conv1) 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        conv2 = self.bn2(conv2)  
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        conv3 = self.bn3(conv3) 
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        x = self.bn4(x) 
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)   
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
    
class UNet_config1_dilated(nn.Module):
    def __init__(self, n_class, kernel_size=20):
        super().__init__()
        
        self.dconv_down1 = double_conv(1, 32, bias=False, 
                                       kernel=[kernel_size, 4], 
                                       padding=[int(kernel_size/2)-1, 5],
                                       dilation=[1, 3])
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.dconv_down2 = double_conv(32, 128)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.dconv_down3 = double_conv(128, 256)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        
        self.dconv_down4 = double_conv(256, 512) 
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, n_class, 1)
        
        
    def forward(self, x):

        conv1 = self.dconv_down1(x)
        conv1 = self.bn1(conv1) 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        conv2 = self.bn2(conv2)  
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        conv3 = self.bn3(conv3) 
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        x = self.bn4(x) 
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)   
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
    
class UNet_Dilated_config1(nn.Module):
    def __init__(self, n_class, kernel_size=20):
        super().__init__()
        
        self.model = UNet_Dilated(1, n_class)
        self.model.model.conv_down[0].conv_layers = nn.ModuleList(
                [nn.Conv2d(1, 32, kernel_size, 
                           padding=int(kernel_size/2)-1, 
                           dilation=1, bias=False),
                nn.Conv2d(32, 32, 4, padding=5, dilation=3)])
        
        
    def forward(self, x):
        out = self.model(x)
        
        return out
    