import cv2
import numpy as np
import pandas as pd
from random import randint
import os
import sys
import math
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
   
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn

from scipy import signal
from scipy import misc
from torchsummary import summary
from sklearn.model_selection import train_test_split 

from tqdm import tqdm
import time

import pydicom
from pydicom.data import get_testdata_files

from IPython.display import clear_output

from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations import Normalize


def get_rows(df, c, t_shift=9):
    
    half_shift = int((t_shift-1)/2)

    SET = df.loc[c].set
    patient = df.loc[c].patient
    seria = df.loc[c].seria
    img_name = df.loc[c].img_name

    df_ = df[(df.set == SET) & (df.patient == patient) & (df.seria == seria)]
    df_.sort_values(by="img_name", inplace=True, ignore_index=True)
#     ind_max = int(df_.loc[df_.index[-1]].img_name.split("-")[-1].split(".")[0])
    ind_max = df_.index.max()
    
    ind = df_[df_.img_name == img_name].index[0]
#     print(ind, ind_max)
    if ind - half_shift < 0:

        ind_1 = 0
        ind_2 = t_shift - 1
    elif ind + half_shift >= ind_max:
        
        ind_2 = ind_max
        ind_1 = ind_max - t_shift + 1
        print(ind, ind_max, ind_1, ind_2)
    else:
        ind_1 = ind - half_shift
        ind_2 = ind + half_shift
    return df_.loc[ind_1: ind_2], ind-ind_1



def get_and_process_rows(df_, ind, HALF_CROP=100, STATS=['0.29587', '0.079193'], img_size=256, t_shift=9):
    imgs, masks = [], []
    
    for t in range(t_shift):
        df_row = df_.loc[df_.index[t]]
        data_path = df_row.data_path
        img, mask = get_image_and_mask(df_row, data_path, img_size=img_size)
        imgs.append(img)
        masks.append(mask)
        
    # Concateate images ad masks
    imgs = np.concatenate(imgs) # (t_shift x H x W)
    imgs = np.transpose(imgs, (1, 2, 0)) # (H x W x t_shift)
    masks = np.concatenate(masks, axis=-1)  # (H x W x t_shift) 

    # Crop
    if HALF_CROP:
        X, Y = np.where(masks[:,:,ind] > 0)
        LIMs = []

        if len(X) != 0:
            for j, xs in enumerate([X, Y]):
                x_len = (xs.max() - xs.min())

                x_shift = max(HALF_CROP - int(x_len / 2), 25) 

                if xs.min() - x_shift < 0:
                    x_lim1 = 0
                    x_lim2 = min(x_len + 2* x_shift , imgs.shape[j]-1)
                elif xs.max() + x_shift > imgs.shape[j]-1:
                    x_lim2 = imgs.shape[j]-1
                    x_lim1 = max(imgs.shape[j]- x_len - 2* x_shift -1, 0)
                else:
                    x_lim1 = xs.min() - x_shift
                    x_lim2 = xs.max() + x_shift
                LIMs.append([x_lim1, x_lim2])

            x_lim1, x_lim2 = LIMs[0]
            y_lim1, y_lim2 = LIMs[1]

            masks = masks[x_lim1: x_lim2, y_lim1: y_lim2, :]
            imgs = imgs[x_lim1: x_lim2, y_lim1: y_lim2, :]

        else:
            x_lim1 = randint(-50, 100)
            x_lim2 = min(imgs.shape[0]-1, x_lim1+HALF_CROP*2)
            x_lim1 = max(x_lim1, 0)

            y_lim1 = randint(-50, 100)
            y_lim2 = min(imgs.shape[1]-1, y_lim1+HALF_CROP*2)
            y_lim1 = max(y_lim1, 0)

            masks = masks[x_lim1: x_lim2, y_lim1: y_lim2, :]
            imgs = imgs[x_lim1: x_lim2, y_lim1: y_lim2, :]
            LIMs = [[x_lim1, x_lim2], [y_lim1, y_lim2]]

    # Apply transforms
    tr_img = Normalize(mean=float(STATS[0]), std=float(STATS[1]), max_pixel_value=1, p=1.0)

    for j in range(imgs.shape[-1]):
        aug = tr_img(image=imgs[:,:,j])
        imgs[:,:,j] = aug["image"]

    # Image -> tensor with size (t_shift x 1 x H x W)
    imgs_ = np.zeros((t_shift, 1, img_size, img_size))
    for j in range(t_shift):
        imgs_[j,0,:,:] = cv2.resize(np.float32(imgs[:,:,j]), (img_size, img_size), interpolation=cv2.INTER_AREA)
    imgs = torch.FloatTensor(imgs_)

    # Masks -> tensor with size (t_shift x 3 x H x W)
    masks_ = np.zeros((t_shift, 3, img_size, img_size))
    masks = np.transpose(masks, (2, 0, 1))

    for j in range(t_shift):
        mask = cv2.resize(masks[j, :, :], (img_size, img_size), interpolation=cv2.INTER_AREA)
        mask = encode_mask(mask )
        masks_[j,:,:,:] = mask

    masks_ = torch.FloatTensor(masks_)

    return imgs, masks_

def remove_empty_masks(df, GAP):
    df["seq"] = df.img_name.apply(lambda x: int(x.split(".")[0].split("-")[-1]) )
    
    index_to_remove = []
    for patient in df.patient.unique():
        df_ = df[df.patient == patient]
        for seria in df_.seria.unique():
            
            df__ = df_[df_.seria == seria]
            df__ = df__.sort_values(by="seq")

            for i in range(df__.shape[0] - GAP):
                if (not df__.loc[df__.index[i + GAP]].if_mask) \
                and (not df__.loc[df__.index[i]].if_mask) \
                and (not df__.loc[df__.index[i - GAP]].if_mask):
                    index_to_remove.append(df__.index[i])
                    
            for i in range(df__.shape[0] - GAP, df__.shape[0]):
                if (not df__.loc[df__.index[i - GAP]].if_mask) \
                and (not df__.loc[df__.index[i]].if_mask):
                    index_to_remove.append(df__.index[i])
    
    return df[~df.index.isin(index_to_remove)]


def get_items(df, t_shift, set_name, step):
    df_ = df[df.set == set_name]
    return [(patient, seria, t, set_name) for patient in df_.patient.unique() 
            for seria in df_[df_.patient == patient].seria.unique()
            for t in range(0, df_[(df_.patient == patient) * (df_.seria == seria)].shape[0] - t_shift + 1, step) ]

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
    

def get_image_and_mask(df_row, data_path, img_size=256):
    IMG_SIZE = img_size
    
    patient = df_row.patient
    seria = df_row.seria

    folder_path = os.path.join(data_path, patient, seria)
    
    # Img
    img_path = os.path.join(folder_path, "img_npy/", df_row.img_name.split(".")[0] + ".npy")
    img = np.load(img_path)
#     print(img.shape)
#     img = img.astype('float32')
         
    # Preprocessing 
    img = pad_if_needed(img)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    img = img - img.min()
    img = img / img.max()
    
    # Mask
    mask_path = os.path.join(folder_path, "mask", df_row.img_name.split(".")[0] + ".png")
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    mask = cv2.resize(mask, img.shape, interpolation=cv2.INTER_AREA)
    
#     mask = encode_mask(mask)
#     mask = mask.transpose(1, 2, 0) ###
#     mask = torch.FloatTensor(mask)
    mask = mask.reshape(img.shape[0], img.shape[1], 1)
    img = img.reshape(1, img.shape[0], img.shape[1])
    
#     img = torch.FloatTensor(img)
    
    return img, mask



def generate_batch(df, t_shift, step, tr_dual=None, tr_img=None, img_size=256, HALF_CROP=None):
    items = get_items(df, t_shift, set_name="OLD", step=step)
    items += get_items(df, t_shift, set_name="NEW", step=step)
    
    idxs = np.arange(len(items))
    np.random.shuffle(idxs)
    i = 0
    
    while True:
        
        if i + 1 >= len(items):
            break

        idx = idxs[i]
        item = items[idx]
#         print(item)
        
        df_ = df[df.set == item[3]]
        df_ = df_[(df_.patient == item[0])*(df_.seria == item[1])]
        df_["seq"] = df_.img_name.apply(lambda x: int(x.split(".")[0].split("-")[-1]) )
        df_ = df_.sort_values(by="seq")

        imgs, masks = [], []
        for t in range(t_shift):
            df_row = df_.loc[df_.index[item[2]+t]]
            data_path = df_row.data_path
            img, mask = get_image_and_mask(df_row, data_path, img_size=img_size)
            imgs.append(img)
            masks.append(mask)
            
        # Concateate images ad masks
        imgs = np.concatenate(imgs) # (t_shift x H x W)
        imgs = np.transpose(imgs, (1, 2, 0)) # (H x W x t_shift)
        masks = np.concatenate(masks, axis=-1)  # (H x W x t_shift) 
        
        # Crop
        if HALF_CROP:
            X, Y, Z = np.where(masks > 0)
            LIMs = []

            if len(X) != 0:
                for j, xs in enumerate([X, Y]):
                    x_len = (xs.max() - xs.min())

                    x_shift = max(HALF_CROP - int(x_len / 2), 25) 

                    if xs.min() - x_shift < 0:
                        x_lim1 = 0
                        x_lim2 = min(x_len + 2* x_shift , imgs.shape[j]-1)
                    elif xs.max() + x_shift > imgs.shape[j]-1:
                        x_lim2 = imgs.shape[j]-1
                        x_lim1 = max(imgs.shape[j]- x_len - 2* x_shift -1, 0)
                    else:
                        x_lim1 = xs.min() - x_shift
                        x_lim2 = xs.max() + x_shift
                    LIMs.append([x_lim1, x_lim2])

                x_lim1, x_lim2 = LIMs[0]
                y_lim1, y_lim2 = LIMs[1]

                masks = masks[x_lim1: x_lim2, y_lim1: y_lim2, :]
                imgs = imgs[x_lim1: x_lim2, y_lim1: y_lim2, :]

            else:
                x_lim1 = randint(-50, 100)
                x_lim2 = min(imgs.shape[0]-1, x_lim1+HALF_CROP*2)
                x_lim1 = max(x_lim1, 0)

                y_lim1 = randint(-50, 100)
                y_lim2 = min(imgs.shape[1]-1, y_lim1+HALF_CROP*2)
                y_lim1 = max(y_lim1, 0)

                masks = masks[x_lim1: x_lim2, y_lim1: y_lim2, :]
                imgs = imgs[x_lim1: x_lim2, y_lim1: y_lim2, :]
                LIMs = [[x_lim1, x_lim2], [y_lim1, y_lim2]]
        
        # Apply transforms
        if tr_dual:
            aug = tr_dual(image=imgs, mask=masks) 
            imgs = aug["image"]
            masks = aug["mask"] 
        
#         print(imgs.shape, masks.shape)
        if tr_img:
            
            for j in range(imgs.shape[-1]):
                aug = tr_img(image=imgs[:,:,j])
                imgs[:,:,j] = aug["image"]

        # Image -> tensor with size (t_shift x 1 x H x W)
        imgs_ = np.zeros((t_shift, 1, img_size, img_size))
        for j in range(t_shift):
            imgs_[j,0,:,:] = cv2.resize(np.float32(imgs[:,:,j]), (img_size, img_size), interpolation=cv2.INTER_AREA)
        imgs = torch.FloatTensor(imgs_)

        # Masks -> tensor with size (t_shift x 3 x H x W)
        masks_ = np.zeros((t_shift, 3, img_size, img_size))
        masks = np.transpose(masks, (2, 0, 1))

        for j in range(t_shift):
            mask = cv2.resize(masks[j, :, :], (img_size, img_size), interpolation=cv2.INTER_AREA)
            mask = encode_mask(mask )
            masks_[j,:,:,:] = mask
            
#             masks_[j,:,:,:] = masks[j*3: (j+1)*3, :, :]
        
        masks_ = torch.FloatTensor(masks_)

        yield imgs, masks_
  
        i += 1
    
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

def calc_loss_new(pred_unet, pred, target, metrics, bce_weight, weight, pos_weight):
    bce_criterion = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
    
    # Model metrics
    bce = bce_criterion(pred.float(), target.float())
    
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight) 
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    # UNet metrics
    bce = bce_criterion(pred_unet.float(), target.float())
    
    pred_unet = torch.sigmoid(pred_unet)
    dice = dice_loss(pred_unet, target)
    
    loss_unet = bce * bce_weight + dice * (1 - bce_weight) 
    
    metrics['bce_unet'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_unet'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss_unet'] += loss_unet.data.cpu().numpy() * target.size(0)
    
    return loss.clone()


def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    
    
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train_model_rnn(model, dfs, t_shift, t_step, phase_range, optimizer, scheduler, num_epochs, model_dir, bce_weight = 0.1, weight=None, pos_weight=None, num_models_to_save=3, earlly_stopping_gap=None, print_per_iter=10, device=torch.device("cpu"), phase_to_save="val", tr_dual=None, tr_img=None, img_size=256, HALF_CROP=None):
    
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
                
            ### For batch norm
            for module in model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()

            metrics = defaultdict(float)
            epoch_samples = 0
            
            dataloader = generate_batch(dfs[phase], 
                                        t_shift=t_shift,
                                        step=t_step,
                                        tr_dual=tr_dual[phase], 
                                        tr_img=tr_img[phase], 
                                        img_size=img_size,
                                        HALF_CROP=HALF_CROP)
            
            for itr, (inputs, labels) in enumerate(dataloader):

                inputs = inputs.float().to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_unet, outputs = model(inputs)
                    
                    loss = calc_loss_new(outputs_unet, outputs, labels, metrics, bce_weight, weight, pos_weight)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                epoch_samples += inputs.size(0)
                
                # Prints and df save
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

