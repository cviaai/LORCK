import sys
import torch
import torch.nn as nn

from UNet_PD import UNet_Baseline, UNet_Dilated, UNet_Original, UNet_ProgressiveDilated, UNet_Original_with_BatchNorm


class Temporal_UNet_Config1_1(nn.Module):
    def __init__(self, t_shift, model="ProgressiveDilated", model_path=None, kernel_size=3, 
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Baseline":
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
            
        self.conv_temp_block = nn.Sequential(
            nn.Conv2d(in_channels=t_shift, out_channels=t_shift, kernel_size=kernel_size, padding=int((kernel_size-1)/2)),
            
            nn.PReLU(),
            nn.Conv2d(in_channels=t_shift, out_channels=t_shift, kernel_size=kernel_size, padding=int((kernel_size-1)/2) ))

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

        # 3
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x.permute(1, 0, 2, 3)
        
        # Output
        out_unet = self.unet.model.conv_last(x)
        
        return out_unet, out_unet
    
    
    
class Temporal_UNet_Config1_2(nn.Module):
    def __init__(self, t_shift, model="ProgressiveDilated", model_path=None, kernel_size=3, 
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Baseline":
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
            
        self.conv_temp_block = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) ))
        )

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

        # 3
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        x = x[None] # 1 x C x T x H x W
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x[0,:,:,:,:]
        x = x.permute(1, 0, 2, 3)
        
        # Output
        out_unet = self.unet.model.conv_last(x)
        
        return out_unet, out_unet

class Temporal_UNet_Config1_3(nn.Module):
    def __init__(self, t_shift, model="ProgressiveDilated", model_path=None, kernel_size=3, 
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Baseline":
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
            
        self.conv_temp_block = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) )),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) ))
        )

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

        # 3
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        x = x[None] # 1 x C x T x H x W
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x[0,:,:,:,:]
        x = x.permute(1, 0, 2, 3)
        
        # Output
        out_unet = self.unet.model.conv_last(x)
        
        return out_unet, out_unet


class Temporal_UNet_Config1_4(nn.Module):
    def __init__(self, t_shift, model="ProgressiveDilated", model_path=None, kernel_size=3, 
                 train_unet_decoder=False, train_unet=False, train_last_block=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Baseline":
            self.unet = UNet_Baseline(in_channels=1, out_channels=3)
        elif model == "Original_with_BatchNorm":
            self.unet = UNet_Original_with_BatchNorm(in_channels=1, out_channels=3)
        else:
            self.unet = UNet_Original(in_channels=1, out_channels=3)
            
        if model_path:
            print("load model -- mode: {}".format(model))
            self.unet.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        self.train_unet_decoder = train_unet_decoder
        self.train_unet = train_unet
        self.train_last_block = train_last_block
            
        self.conv_temp_block = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) )),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) ))
        )

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

            for i, conv_block in enumerate(self.unet.model.conv_up[:-1]):
                x = self.unet.model.upsample(x)
                x = torch.cat([x, encoding[::-1][i]], dim=1)
                x = conv_block(x)
        
        # 3: Last block
        with torch.set_grad_enabled(self.train_last_block):
            conv_block = self.unet.model.conv_up[-1]
            x = self.unet.model.upsample(x)
            x = torch.cat([x, encoding[::-1][i+1]], dim=1)
            x = conv_block[0](x)

            x = x.permute(1, 0, 2, 3)
            x = x[None] # 1 x C x T x H x W

            # Temporal convolutional block
            x = self.conv_temp_block(x)
            x = x[0,:,:,:,:]
            x = x.permute(1, 0, 2, 3)

            x = conv_block[1](x)

        # Output
        out = self.unet.model.conv_last(x)
        
        return out, out


class Temporal_UNet_Config2_2(nn.Module):
    def __init__(self, t_shift, model="ProgressiveDilated", model_path=None, kernel_size=3, 
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Baseline":
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
            
        self.conv_temp_block = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) ))
        )

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
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        x = x[None] # 1 x C x T x H x W
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x[0,:,:,:,:]
        x = x.permute(1, 0, 2, 3)
        
        # 4
        with torch.set_grad_enabled(self.train_unet_decoder or self.train_unet):
            for i, conv_block in enumerate(self.unet.model.conv_up):
                x = self.unet.model.upsample(x)
                x = torch.cat([x, encoding[::-1][i]], dim=1)
                x = conv_block(x)

            out_unet = self.unet.model.conv_last(x)
        
        return out_unet, out_unet
    
class Temporal_UNet_Config2_3(nn.Module):
    def __init__(self, t_shift, model="ProgressiveDilated", model_path=None, kernel_size=3, 
                 train_unet_decoder=False, train_unet=False):
        super().__init__()
        
        if model == "ProgressiveDilated":
            self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        elif model == "Dilated":
            self.unet = UNet_Dilated(in_channels=1, out_channels=3)
        elif model == "Baseline":
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
            
        self.conv_temp_block = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) )),
            nn.BatchNorm3d(1024),
            nn.PReLU(),
            
            nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(t_shift, kernel_size, kernel_size), 
                      padding=( int((t_shift - 1)/2), int( (kernel_size-1) /2), int( (kernel_size-1) /2) ))
        )

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
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        x = x[None] # 1 x C x T x H x W
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x[0,:,:,:,:]
        x = x.permute(1, 0, 2, 3)
        
        # 4
        with torch.set_grad_enabled(self.train_unet_decoder or self.train_unet):
            for i, conv_block in enumerate(self.unet.model.conv_up):
                x = self.unet.model.upsample(x)
                x = torch.cat([x, encoding[::-1][i]], dim=1)
                x = conv_block(x)

            out_unet = self.unet.model.conv_last(x)
        
        return out_unet, out_unet
    
    
class UNet_MiddleTemporalBlock(nn.Module):
    def __init__(self, t_shift, pretrained=True):
        super().__init__()
        
        self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        if pretrained:
            folder = "../unet_progressive/weights_folder_new/UNet_ProgressiveDilated_20.01.2020.00:50/"
            self.unet.load_state_dict(torch.load(folder + "weight.epoch_298_loss_train_0.8037294090116782.pth"))
        
        self.conv_temp_block = nn.Sequential(
            nn.Conv2d(in_channels=t_shift, out_channels=t_shift, kernel_size=2, padding=1,),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=t_shift, out_channels=t_shift, kernel_size=2, padding=0,))
        
    def forward(self, x):
        
        # UNet Encoding
        encoding = []
        for conv_block in self.unet.model.conv_down:
            x = conv_block(x)

            if self.unet.model.downsample_type == 'conv_stride':
                encoding.append(x)
            if self.unet.model.downsample_type == 'maxpool':
                encoding.append(x)
                x = self.unet.model.maxpool(x)
                
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x.permute(1, 0, 2, 3)
        
        # Middle block
        if self.unet.model.advanced_bottleneck:
            x = self.unet.model.conv_middle_part1(x)
            x = self.unet.model.conv_middle_part2(x)
        else:
            x = self.unet.model.conv_middle(x)
        
        # UNet Decoding
        for i, conv_block in enumerate(self.unet.model.conv_up):
            x = self.unet.model.upsample(x)
            x = torch.cat([x, encoding[::-1][i]], dim=1)
            x = conv_block(x)
            
        if not self.unet.model.is_block:
            x = self.unet.model.conv_last(x)
   
        return x


class UNet_FinalTemporalBlock(nn.Module):
    def __init__(self, t_shift, pretrained=True):
        super().__init__()
        
        self.unet = UNet_ProgressiveDilated(in_channels=1, out_channels=3)
        if pretrained:
            folder = "../unet_progressive/weights_folder_new/UNet_ProgressiveDilated_20.01.2020.00:50/"
            self.unet.load_state_dict(torch.load(folder + "weight.epoch_298_loss_train_0.8037294090116782.pth"))
        
        self.conv_temp_block = nn.Sequential(
            nn.Conv2d(in_channels=t_shift, out_channels=t_shift, kernel_size=2, padding=1,),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=t_shift, out_channels=t_shift, kernel_size=2, padding=0,))
        
    def forward(self, x):
        
        # UNet
        x = self.unet(x)
                
        # Permute batch_size and channel dims 
        x = x.permute(1, 0, 2, 3)
        
        # Temporal convolutional block
        x = self.conv_temp_block(x)
        x = x.permute(1, 0, 2, 3)
   
        return x