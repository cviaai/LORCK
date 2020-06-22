import numpy as np
import torch
import torch.nn as nn


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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']