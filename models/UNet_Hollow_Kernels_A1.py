import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from UNet_PD import UNet_Dilated

def double_conv(in_channels, out_channels, bias=True, kernel=[3, 3], padding=[1, 1]):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel[0], padding=padding[0], bias=bias),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel[1], padding=padding[1]),
        nn.PReLU()
    )  

class Conv2d_with_GivenKernel(nn.Module):
    def __init__(self, stride=1, 
                 padding=0, dilation=1):
        
        super().__init__()
        
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        
    def forward(self, input, weight):
#         print("input: ", input.shape, weight.shape)
        if not self.padding:
            self.padding = int(weight.shape[-1] / 2) - 1
            
        out_for_batch = []
        for i in range(weight.shape[0]):
            input_ = input[i,:,:,:].unsqueeze(0)
            out_for_image = []
            for j in range(weight.shape[1]):

                weight_ = weight[i,j,:,:].unsqueeze(0).unsqueeze(0)

                out_for_image_channel =  F.conv2d(input_, weight_, 
                                                 padding=self.padding,
                                                 stride=self.stride,
                                                 dilation=self.dilation)


                out_for_image.append(out_for_image_channel)
            
            out_for_image = torch.cat(out_for_image, dim=1)
            out_for_batch.append(out_for_image)
                
        if weight.shape[1] > 1:
            return torch.cat(out_for_batch, dim=0)
        else:
            return out_for_batch[0]

class Dilated_UNet_config1(nn.Module):
    def __init__(self, n_class):
        super().__init__()
                
        
        self.model = UNet_Dilated(1, n_class)
        self.model.model.conv_down[0].conv_layers = nn.ModuleList(
                [Conv2d_with_GivenKernel(padding=None),
                 nn.Conv2d(32, 32, 4, padding=5, dilation=3)])
        
    def forward(self, x, w):
        
        encoding = []
        for i, conv_block in enumerate(self.model.model.conv_down):
            if i == 0:
                x = conv_block.conv_layers[0](x, w)
                x = conv_block.activation(x)
                x = conv_block.batchnorm_layers[0](x)
                x = conv_block.conv_layers[1](x)
                x = conv_block.activation(x)
                x = conv_block.batchnorm_layers[1](x)
            else:
                x = conv_block(x)
            
            if self.model.model.downsample_type == 'conv_stride':
                encoding.append(x)
            if self.model.model.downsample_type == 'maxpool':
                encoding.append(x)
                x = self.model.model.maxpool(x)
         
        if self.model.model.advanced_bottleneck:
            x = self.model.model.conv_middle_part1(x)
            x = self.model.model.conv_middle_part2(x)
        else:
            x = self.model.model.conv_middle(x)
        
        for i, conv_block in enumerate(self.model.model.conv_up):
            x = self.model.model.upsample(x)
            x = torch.cat([x, encoding[::-1][i]], dim=1)
            x = conv_block(x)
            
        if not self.model.model.is_block:
            x = self.model.model.conv_last(x)
   
        return x 
    
class UNet_config1(nn.Module):
    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = nn.Sequential(
            Conv2d_with_GivenKernel(padding=None),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, padding=2),
            nn.PReLU()
        )  
        
        
#         self.dconv_down1 = double_conv(1, 32, bias=False, kernel=[20, 4], padding=[9, 2])
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
        
        
    def forward(self, x, w):

        conv1 = self.dconv_down1[0](x, w)
        for i in range(1, 4):
            conv1 = self.dconv_down1[i](conv1)

        conv1 = self.bn1(conv1) ### was removed 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        conv2 = self.bn2(conv2) ### was removed 
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        conv3 = self.bn3(conv3) ### was removed 
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        x = self.bn4(x) ###
        
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

class UNet_config2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = nn.Sequential(
            Conv2d_with_GivenKernel(padding=None),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, padding=2),
            nn.PReLU()
        )  
        
        
#         self.dconv_down1 = double_conv(1, 32, bias=False, kernel=[20, 4], padding=[9, 2])
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
        
        
    def forward(self, x, w):
        
        conv1 = self.dconv_down1[0](x, w)
        for i in range(1, 4):
            conv1 = self.dconv_down1[i](conv1)

#         conv1 = self.bn1(conv1) ### was removed 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
#         conv2 = self.bn2(conv2) ### was removed 
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
#         conv3 = self.bn3(conv3) ### was removed 
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        x = self.bn4(x) ###
        
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

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        return x * self.tanh(self.softplus(x))
    
def double_conv_withMish(in_channels, out_channels, bias=True, kernel=[3, 3], padding=[1, 1]):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel[0], padding=padding[0], bias=bias),
        Mish(),
        nn.Conv2d(out_channels, out_channels, kernel[1], padding=padding[1]),
        Mish()
    )  

class UNet_config3_withMish(nn.Module):
    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = nn.Sequential(
            Conv2d_with_GivenKernel(padding=None),
            Mish(),
            nn.Conv2d(32, 32, 4, padding=2),
            Mish()
        )  
        
        
#         self.dconv_down1 = double_conv(1, 32, bias=False, kernel=[20, 4], padding=[9, 2])
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.dconv_down2 = double_conv_withMish(32, 128)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.dconv_down3 = double_conv_withMish(128, 256)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        
        self.dconv_down4 = double_conv_withMish(256, 512) 
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv_withMish(256 + 512, 256)
        self.dconv_up2 = double_conv_withMish(128 + 256, 128)
        self.dconv_up1 = double_conv_withMish(128 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, n_class, 1)
        
        
    def forward(self, x, w):
        
        conv1 = self.dconv_down1[0](x, w)
        for i in range(1, 4):
            conv1 = self.dconv_down1[i](conv1)

#         conv1 = self.bn1(conv1) ### was removed 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
#         conv2 = self.bn2(conv2) ### was removed 
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
#         conv3 = self.bn3(conv3) ### was removed 
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        x = self.bn4(x) ###
        
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


