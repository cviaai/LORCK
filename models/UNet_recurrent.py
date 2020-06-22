import sys
import torch
import torch.nn as nn

from ConvLSTM_pytorch.convlstm import ConvLSTMCell, ConvLSTM

from UNet_PD import UNet_Baseline, UNet_Dilated, UNet_Original, UNet_Original_with_BatchNorm, UNet_ProgressiveDilated
from UNet_Hollow_Kernels_A2_Congig2 import UNet_Dilated_2levels_config1


class RNN_UNet_Config1(nn.Module):
    def __init__(self, model="ProgressiveDilated", model_path=None, kernel_size=3, num_layers=1):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Baseline(in_channels=1, out_channels=3)
        elif model == "Original_with_BatchNorm":
            self.unet = UNet_Original_with_BatchNorm(in_channels=1, out_channels=3)
        else:
            self.unet = UNet_Original(in_channels=1, out_channels=3)
            
        if model_path:
            print("load model -- mode: {}".format(model))
            self.unet.load_state_dict(torch.load(model_path))
        
        self.convlstm_forward = ConvLSTM(input_size=(256, 256), input_dim=3, 
                                         hidden_dim=3, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.convlstm_backward = ConvLSTM(input_size=(256, 256), input_dim=3, 
                                         hidden_dim=3, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.last_conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)
        
    def forward(self, x):

        out_unet = self.unet(x)
        x = out_unet[None].permute(1, 0, 2, 3, 4)
#         print(x.shape)
        
        out_forward = self.convlstm_forward(x)
        out_backward = self.convlstm_backward(torch.flip(x, (0,)))
#         print(out_forward[0][0].shape, out_backward[0][0].shape)
        
        out = torch.cat([out_forward[0][0][0], out_backward[0][0][0]], dim=1)
        out = self.last_conv(out)
        
        return out_unet, out
    
    
class RNN_UNet_Config1_1(nn.Module):
    def __init__(self, model="ProgressiveDilated", model_path=None, kernel_size=3, num_layers=1,
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Baseline(in_channels=1, out_channels=3)
        elif model == "Original_with_BatchNorm":
            self.unet = UNet_Original_with_BatchNorm(in_channels=1, out_channels=3)
        else:
            self.unet = UNet_Original(in_channels=1, out_channels=3)
            
        if model_path:
            print("load model -- mode: {}".format(model))
            self.unet.load_state_dict(torch.load(model_path))
        
        self.train_unet_decoder = train_unet_decoder
        self.train_unet = train_unet
        
        self.convlstm_forward = ConvLSTM(input_size=(256, 256), input_dim=32, 
                                         hidden_dim=32, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.convlstm_backward = ConvLSTM(input_size=(256, 256), input_dim=32, 
                                         hidden_dim=32, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.last_conv = nn.Conv2d(in_channels=32*2, out_channels=3, kernel_size=3, padding=1)
        
    def forward(self, x):

        # 1
        with torch.set_grad_enabled(self.train_unet):
            encoding = []
            for conv_block in self.unet.model.conv_down:
                x = conv_block(x)

                if self.unet.model.downsample_type == 'conv_stride':
                    encoding.append(x)
                if self.unet.model.downsample_type == 'maxpool':
                    encoding.append(x)
                    x = self.unet.model.maxpool(x)
        
        # 2
        with torch.set_grad_enabled(self.train_unet_decoder or self.train_unet):
         
            if self.unet.model.advanced_bottleneck:
                x = self.unet.model.conv_middle_part1(x)
                x = self.unet.model.conv_middle_part2(x)
            else:
                x = self.unet.model.conv_middle(x)

            for i, conv_block in enumerate(self.unet.model.conv_up):
                x = self.unet.model.upsample(x)
                x = torch.cat([x, encoding[::-1][i]], dim=1)
                x = conv_block(x)

            out_unet = self.unet.model.conv_last(x)
        
        # 3
        x = x[None].permute(1, 0, 2, 3, 4)

        out_forward = self.convlstm_forward(x)
        out_backward = self.convlstm_backward(torch.flip(x, (0,)))

        out = torch.cat([out_forward[0][0][0], out_backward[0][0][0]], dim=1)
        out = self.last_conv(out)
        
        return out_unet, out
    
    
    
    
class RNN_UNet_Config2(nn.Module):
    def __init__(self, model="ProgressiveDilated", model_path=None, kernel_size=3, num_layers=1,
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Baseline(in_channels=1, out_channels=3)
        elif model == "Original_with_BatchNorm":
            self.unet = UNet_Original_with_BatchNorm(in_channels=1, out_channels=3)
        else:
            self.unet = UNet_Original(in_channels=1, out_channels=3)
            
        if model_path:
            print("load model -- mode: {}".format(model))
            self.unet.load_state_dict(torch.load(model_path))
            
        self.train_unet_decoder = train_unet_decoder
        self.train_unet = train_unet
        
        self.convlstm_forward = ConvLSTM(input_size=(16, 16), input_dim=1024, 
                                         hidden_dim=512, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.convlstm_backward = ConvLSTM(input_size=(16, 16), input_dim=1024, 
                                         hidden_dim=512, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
    def forward(self, x):

        # 1
        with torch.set_grad_enabled(self.train_unet):
            encoding = []
            for conv_block in self.unet.model.conv_down:
                x = conv_block(x)

                if self.unet.model.downsample_type == 'conv_stride':
                    encoding.append(x)
                if self.unet.model.downsample_type == 'maxpool':
                    encoding.append(x)
                    x = self.unet.model.maxpool(x)
        
        # 2
        with torch.set_grad_enabled(self.train_unet_decoder or self.train_unet):
            if self.unet.model.advanced_bottleneck:
                x = self.unet.model.conv_middle_part1(x)
                x = self.unet.model.conv_middle_part2(x)
            else:
                x = self.unet.model.conv_middle(x)

        # 3
        x = x[None].permute(1, 0, 2, 3, 4)

        out_forward = self.convlstm_forward(x)
        out_backward = self.convlstm_backward(torch.flip(x, (0,)))
        x = torch.cat([out_forward[0][0][0], out_backward[0][0][0]], dim=1)
        
        # 4
        with torch.set_grad_enabled(self.train_unet_decoder or self.train_unet):
            for i, conv_block in enumerate(self.unet.model.conv_up):
                x = self.unet.model.upsample(x)
                x = torch.cat([x, encoding[::-1][i]], dim=1)
                x = conv_block(x)

            out_unet = self.unet.model.conv_last(x)
        
        return out_unet, out_unet
    
    
    
class RNN_Dilated_UNet_with_hollow_kernels(nn.Module):
    def __init__(self, model_path=None, kernel_size=3, num_layers=1,
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        self.unet = UNet_Dilated_2levels_config1(3)
            
        if model_path:
#             print("load model -- mode: {}".format(model))
            self.unet.load_state_dict(torch.load(model_path))
        
        self.train_unet_decoder = train_unet_decoder
        self.train_unet = train_unet
        
        self.convlstm_forward = ConvLSTM(input_size=(256, 256), input_dim=32, 
                                         hidden_dim=32, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.convlstm_backward = ConvLSTM(input_size=(256, 256), input_dim=32, 
                                         hidden_dim=32, kernel_size=(kernel_size, kernel_size), 
                                         num_layers=num_layers, batch_first=False, 
                                         bias=True, return_all_layers=False)
        
        self.last_conv = nn.Conv2d(in_channels=32*2, out_channels=3, kernel_size=3, padding=1)
        
    def forward(self, x):

        # 1
        with torch.set_grad_enabled(self.train_unet):
            encoding = []
            for conv_block in self.unet.model.model.conv_down:
                x = conv_block(x)

                if self.unet.model.model.downsample_type == 'conv_stride':
                    encoding.append(x)
                if self.unet.model.model.downsample_type == 'maxpool':
                    encoding.append(x)
                    x = self.unet.model.model.maxpool(x)
        
        # 2
        with torch.set_grad_enabled(self.train_unet_decoder or self.train_unet):
         
            if self.unet.model.model.advanced_bottleneck:
                x = self.unet.model.model.conv_middle_part1(x)
                x = self.unet.model.model.conv_middle_part2(x)
            else:
                x = self.unet.model.model.conv_middle(x)

            for i, conv_block in enumerate(self.unet.model.model.conv_up):
                x = self.unet.model.model.upsample(x)
                x = torch.cat([x, encoding[::-1][i]], dim=1)
                x = conv_block(x)

            out_unet = self.unet.model.model.conv_last(x)
        
        # 3
        x = x[None].permute(1, 0, 2, 3, 4)

        out_forward = self.convlstm_forward(x)
        out_backward = self.convlstm_backward(torch.flip(x, (0,)))

        out = torch.cat([out_forward[0][0][0], out_backward[0][0][0]], dim=1)
        out = self.last_conv(out)
        
        return out_unet, out
    
    
    
    
