import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import randint
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

def encode_mask(mask):
    size = mask.shape[0], mask.shape[1]
    
    mask_new = np.zeros(( 3, mask.shape[0], mask.shape[1]))
    
    for i in range(1, 4):
        mask_new[i-1,:,:] = np.ones_like(mask)*(mask == i)

    return mask_new

def pad_if_needed(img):
    
    if img.shape[0] != img.shape[1]:
        max_shape = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]

        img_new = np.zeros((max_shape, max_shape))

        img_new[int((max_shape - img.shape[0]) / 2): max_shape - int((max_shape - img.shape[0]) / 2),
                int((max_shape - img.shape[1]) / 2): max_shape - int((max_shape - img.shape[1]) / 2)] = img
        return img_new
        
    else:
        return img


class MRI_Simple_Dataset(Dataset):
    def __init__(self, df, phase, img_size, num_classes=3, tr_dual=None, 
                 tr_img=None, HALF_CROP=None, num_no_mask=1., out_channels=None):
        self.df = df
        self.phase = phase
        
        self.img_size = img_size
        self.HALF_CROP = HALF_CROP
        
        self.df_no_mask = df[df.if_mask == False]
        self.df_with_mask = df[df.if_mask == True]
        
        self.tr_dual = tr_dual
        self.tr_img = tr_img
        
        assert num_classes in [1, 2, 3]
        self.num_classes = num_classes
        self.num_no_mask = max(min(num_no_mask, 1), 0)
        
        self.out_channels = 1 if self.num_classes > 1 else out_channels
        
    def __len__(self):
        return self.df_with_mask.shape[0] + int(self.num_no_mask * self.df_no_mask.shape[0])
    
    def __getitem__(self, ind):
        
        if ind < self.df_with_mask.shape[0]:
            ind = self.df_with_mask.index[ind]
        else:
            ind = self.df_no_mask.index[ind - self.df_with_mask.shape[0]]

        patient = self.df.loc[ind].patient
        seria = self.df.loc[ind].seria
        data_path = self.df.loc[ind].data_path
        
        folder_path = os.path.join(data_path, patient, seria)
        
        # Img
        img_path = os.path.join(folder_path, "img_npy/", self.df.loc[ind].img_name.split(".")[0] + ".npy")
        img = np.load(img_path)
        
        # Preprocessing 
        img = pad_if_needed(img)
#         img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        img = img - img.min()
        img = img / img.max()
        
        # Mask
        mask_path = os.path.join(folder_path, "mask", self.df.loc[ind].img_name.split(".")[0] + ".png")
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        mask = cv2.resize(mask, img.shape, interpolation=cv2.INTER_AREA)
        
        # Crop
        if self.HALF_CROP:
            X, Y = np.where(mask > 0)
            LIMs = []

            if len(X) != 0:
                for i, xs in enumerate([X, Y]):
                    x_len = (xs.max() - xs.min())

                    x_shift = max(self.HALF_CROP - int(x_len / 2), 25) 

                    if xs.min() - x_shift < 0:
                        x_lim1 = 0
                        x_lim2 = min(x_len + 2* x_shift , img.shape[i]-1)
                    elif xs.max() + x_shift > img.shape[i]-1:
                        x_lim2 = img.shape[i]-1
                        x_lim1 = max(img.shape[i]- x_len - 2* x_shift -1, 0)
                    else:
                        x_lim1 = xs.min() - x_shift
                        x_lim2 = xs.max() + x_shift
                    LIMs.append([x_lim1, x_lim2])

                x_lim1, x_lim2 = LIMs[0]
                y_lim1, y_lim2 = LIMs[1]

                mask = mask[x_lim1: x_lim2, y_lim1: y_lim2]
                img = img[x_lim1: x_lim2, y_lim1: y_lim2]

            else:
                x_lim1 = randint(-50, 100)
                x_lim2 = min(img.shape[0]-1, x_lim1+self.HALF_CROP*2)
                x_lim1 = max(x_lim1, 0)

                y_lim1 = randint(-50, 100)
                y_lim2 = min(img.shape[1]-1, y_lim1+self.HALF_CROP*2)
                y_lim1 = max(y_lim1, 0)

                mask = mask[x_lim1: x_lim2, y_lim1: y_lim2]
                img = img[x_lim1: x_lim2, y_lim1: y_lim2]
                LIMs = [[x_lim1, x_lim2], [y_lim1, y_lim2]]

        # Aug
        if self.tr_dual:
            aug = self.tr_dual(image=img, mask=mask) 
            img = aug["image"]
            mask = aug["mask"]
        
        if self.tr_img:
            aug = self.tr_img(image=img)
            img = aug["image"]
        
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.reshape(1, self.img_size, self.img_size)
        
        mask = encode_mask(mask)
        mask = torch.FloatTensor(mask)
        
        mask = mask[:self.num_classes,:,:]
        
        if self.out_channels > 1:
            mask = torch.cat([mask for _ in range(self.out_channels)], dim=0)
        
        return img, mask

    
def calc_iou(pred, target, t=0.5):
    pred = torch.sigmoid(pred)
    pred = pred.cpu().numpy() > 0.5
    
    intersection = np.logical_and(target.cpu().numpy(), pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    
def dice_loss(pred, target, smooth = 1.): 
    pred = pred.contiguous() 
    target = target.contiguous()

    intersection = (pred.float() * target.float()).sum(dim=2).sum(dim=2) 
    
    loss = (1 - ((2. * intersection + smooth) / (pred.float().sum(dim=2).sum(dim=2) + target.float().sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()



def calc_loss(pred, target, metrics, bce_weight, weight, pos_weight):
    bce_criterion = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
    bce = bce_criterion(pred.float(), target.float())
    
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    
    loss = bce * bce_weight + dice * (1 - bce_weight) 
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss.clone()

def calc_loss_nologs(pred, target, metrics, bce_weight, weight, pos_weight):
    bce_criterion = nn.BCELoss(weight=weight, reduction="mean")
    pred = torch.softmax(pred, 1)
    bce = bce_criterion(pred.float(), target.float())

    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight) 
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss.clone()


def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    
    
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, dataloaders, phase_range, optimizer, scheduler, num_epochs, model_dir, bce_weight = 0.1, weight=None, pos_weight=None, num_models_to_save=3, earlly_stopping_gap=None, print_per_iter=10, device=torch.device("cpu"), phase_to_save="val"):
    
    best_losses = {phase: [10000.] for phase in phase_to_save}
    num_saved_models = {phase: 0 for phase in phase_to_save}
    
    df = pd.DataFrame(columns=['epoch', 'phase', 'lr', 'itr', 'bce', 'dice', 'loss'])
    df_path = model_dir + "/df_logs.csv"
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        for phase in phase_range:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  
            else:
                model.eval()   

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for itr, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.float().to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    ######## Change to smth
                    loss = calc_loss(outputs, labels, metrics, bce_weight, weight, pos_weight)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                epoch_samples += inputs.size(0)
                
                if itr % print_per_iter == (print_per_iter-1):
                    print_metrics(metrics, epoch_samples, phase)
                    df.loc[df.shape[0]] = [epoch, phase, get_lr(optimizer), itr, 
                                           metrics['bce']/ epoch_samples, 
                                           metrics['dice']/ epoch_samples, 
                                           metrics['loss']/ epoch_samples]
                    df.to_csv(df_path, index=False)
                    
                del inputs, labels, outputs

            # Prints and df save
            print_metrics(metrics, epoch_samples, phase)
            df.loc[df.shape[0]] = [epoch, phase, get_lr(optimizer), itr, 
                                   metrics['bce']/ epoch_samples, 
                                   metrics['dice']/ epoch_samples, 
                                   metrics['loss']/ epoch_samples]
            df.to_csv(df_path, index=False)
            epoch_loss = metrics['loss'] / epoch_samples
                  
            if phase in phase_to_save and (epoch_loss < np.array(best_losses[phase])).any():
                
                if 10000. in best_losses[phase]:
                    best_losses[phase].remove(10000.)

                # Save new best model
                print("--saving best model--")
                best_losses[phase].append(epoch_loss)
                model_suffix = "weight.epoch_{}_loss_{}_{}.pth".format(epoch+1, phase, epoch_loss)
                model_path = os.path.join(model_dir, model_suffix)

                torch.save(model.state_dict(), model_path)
                num_saved_models[phase] += 1

                # If there are more then num_models_to_save saved models, remove the worst one
                if num_saved_models[phase] > num_models_to_save:
                    max_loss = 0.0
                    for file_name in os.listdir(model_dir):
                        if "loss" in file_name and phase in file_name:
                            loss = file_name.split("_")[-1][:-4]
                            loss = float(loss)
                            if loss > max_loss:
                                max_loss = loss
                                model_with_max_loss = os.path.join(model_dir, file_name)
                    os.remove(model_with_max_loss)
                    best_losses[phase].remove(max_loss)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    for phase in phase_to_save:
        print('Best {} loss: {:4f}'.format(phase, np.array(best_losses[phase]).min()))

    return model  
    
 