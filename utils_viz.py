from random import randint
import cv2
import numpy as np
import pandas as pd
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

from scipy import signal
from scipy import misc
from skimage.io import imsave

from tqdm import tqdm
import time

import argparse
import logging
from datetime import datetime
from collections import defaultdict

import pydicom
from pydicom.data import get_testdata_files

from IPython.display import clear_output

from torch.utils.data import DataLoader, Dataset

from albumentations import (HorizontalFlip, VerticalFlip, Rotate, MotionBlur, MedianBlur, Blur,
                            Compose, OneOf, ElasticTransform, GridDistortion, OpticalDistortion,
                            CropNonEmptyMaskIfExists, Resize, Normalize)

from utils_temporal import (generate_batch, remove_empty_masks, encode_mask, get_and_process_rows,
                            calc_loss_new, print_metrics, get_image_and_mask, get_rows)

from utils import process

class_to_color = {
    0: [250, 240, 100], # OW - yellow
    1: [120, 220, 140], # IW - green
    2: [230, 150, 150], # tumor - orange
}


def viz_small_results(df, c, models, models_A2, device):
    t=0.8
    SIZE = 28
    
    ### Get images
    img, mask = get_image_and_mask(df.loc[c], df.loc[c].data_path, img_size=256)

    ### Preprocess
    X, mask_ = process(img[0].copy(), mask[:,:,0].copy())
    X = torch.tensor(X[None])
    
    X_for_kernel, mask_for_kernel = process(img[0].copy(), mask[:,:,0].copy(), HALF_CROP=50, img_size=20)
    X_for_kernel = torch.tensor(X_for_kernel[None])

    mask = mask_
    
    ### Viz
    num_models = 8
    fig, ax = plt.subplots(1, num_models, figsize=(num_models*5, 5))
    for i in range(num_models):
            ax[i].axis("off")

    # 0: Input image
    img = X[0, 0, :, :].numpy()
    img = (img - img.min())  * 255. / (img.max() - img.min())
    img = np.array(img, dtype=np.uint8)

    img_rgb = np.stack([img, img, img], axis=-1)

    ax[0].imshow(img_rgb / 255. , cmap="bone")
    ax[0].set_title("Input image", fontsize=SIZE)

    # 1: Ground truth
    for i in range(3):
        mask_ = np.array(mask[i, :, :], dtype=bool)
        color = class_to_color[i]

        for c_ in range(3):
            img_rgb[mask_,c_] = color[c_]

    ax[1].imshow(img_rgb/ 255.)
    ax[1].set_title("Ground truth", fontsize=SIZE)
    
    j = 2
    
    # 2-4: Baseline models
    for model_name in ["U-Net", "U-Net Dilated", "E-Net"]:
        with torch.no_grad():
            model = models[model_name].to(device)
            out_ = model(X.float().to(device))
            out = torch.sigmoid(out_)
            out_args = torch.argmax(out[0, :, :, :], dim=0).cpu().numpy()

        img_rgb = np.stack([img, img, img], axis=-1)

        for i in range(3):
            mask_ = (out[0, i, :, :].cpu().numpy() > t) & (out_args == i)
            mask_ = np.array(mask_, dtype=bool)
            color = class_to_color[i]

            for c_ in range(3):
                img_rgb[mask_,c_] = color[c_]

            ax[j].imshow(img_rgb)
            ax[j].set_title(model_name, fontsize=SIZE)
            
        j+=1
        
    # 5,7: Temporal and Bi-LSTM models
    try:
        for t, model_name in zip([0.5, 0.2], ["Bi-LSTM U-Net", "Bi-LSTM Hollow \nKernels U-Net"]):

            model = models[model_name].to(device)

            df_, ind_ = get_rows(df, c)

            imgs, masks = get_and_process_rows(df_, ind_)
            with torch.no_grad():
                out_ = model(imgs.to(device))
                out = torch.sigmoid(out_[1])
                out_args = torch.argmax(out[ind_, :, :, :], dim=0).cpu().numpy()

            img = imgs[ind_, 0, :, :].numpy()
            img = (img - img.min())  * 255. / (img.max() - img.min())
            img = np.array(img, dtype=np.uint8)

            img_rgb = np.stack([img, img, img], axis=-1)

            for i in range(3):
                mask_ = (out[ind_, i, :, :].cpu().numpy() > t)
                mask_ = np.array(mask_, dtype=bool)
                color = class_to_color[i]

                for c_ in range(3):
                    img_rgb[mask_,c_] = color[c_]

            ax[j].imshow(img_rgb)
            ax[j].set_title(model_name, fontsize=SIZE)

            j += 2
    except:
        pass
    
    ### 6: Hollow kernels
    model_name = "A2 Config. 2.1"
        
    model = models_A2[model_name]
    model = model.to(device)

    with torch.no_grad():
        out = model(X.to(device))
        out = torch.sigmoid(out)
        out_args = torch.argmax(out[0, :, :, :], dim=0).cpu().numpy()

    img_rgb = np.stack([img, img, img], axis=-1)

    for i in range(3):
        mask_ = (out[0, i, :, :].cpu().numpy() > t)# & (out_args == i)
        mask_ = np.array(mask_, dtype=bool)
        color = class_to_color[i]

        for c_ in range(3):
            img_rgb[mask_,c_] = color[c_]

    ax[6].imshow(img_rgb)
    title = "Hollow Kernel \nU-Net"
    ax[6].set_title(title, fontsize=SIZE)

    plt.subplots_adjust(wspace=0.1)
    plt.show()
    
    
    
def viz_big_results(df, c, models, models_and_kernels, models_A2, device):
    
    ### #1: ORIGINAL IMAGE, GROUND TRUTH, BASELINES
    
    SIZE = 28
    ### Get images
    img, mask = get_image_and_mask(df.loc[c], df.loc[c].data_path, img_size=256)

    ### Preprocess
    X, mask_ = process(img[0].copy(), mask[:,:,0].copy())
    X = torch.tensor(X[None])
    
    X_for_kernel, mask_for_kernel = process(img[0].copy(), mask[:,:,0].copy(), HALF_CROP=50, img_size=20)
    X_for_kernel = torch.tensor(X_for_kernel[None])

    mask = mask_

    num_models = 8
    fig, ax = plt.subplots(1, num_models, figsize=(num_models*5, 5))
    for i in range(num_models):
            ax[i].axis("off")

    # 0: Input image
    img = X[0, 0, :, :].numpy()
    img = (img - img.min())  * 255. / (img.max() - img.min())
    img = np.array(img, dtype=np.uint8)

    img_rgb = np.stack([img, img, img], axis=-1)

    ax[0].imshow(img_rgb / 255. , cmap="bone")
    ax[0].set_title("Input image", fontsize=SIZE)

    # 1: Ground truth

    for i in range(3):
        mask_ = np.array(mask[i, :, :], dtype=bool)
        color = class_to_color[i]

        for c_ in range(3):
            img_rgb[mask_,c_] = color[c_]

    ax[1].imshow(img_rgb/ 255.)
    ax[1].set_title("Ground truth", fontsize=SIZE)
    
    j = 2
    
    # 2-5: models
    t=0.8
    for model_name in ["U-Net", "U-Net Dilated", "U-Net PD", "E-Net"]:

        with torch.no_grad():
            model = models[model_name].to(device)
            out_ = model(X.float().to(device))
            out = torch.sigmoid(out_)
            out_args = torch.argmax(out[0, :, :, :], dim=0).cpu().numpy()

        img_rgb = np.stack([img, img, img], axis=-1)

        for i in range(3):
            mask_ = (out[0, i, :, :].cpu().numpy() > t) & (out_args == i)
            mask_ = np.array(mask_, dtype=bool)
            color = class_to_color[i]

            for c_ in range(3):
                img_rgb[mask_,c_] = color[c_]

            ax[j].imshow(img_rgb)
            ax[j].set_title(model_name, fontsize=SIZE)
            
        j+=1
        
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # 6,7: Temporal and Bi-LSTM models
    try:
        for model_name in ["Temporal U-Net", "Bi-LSTM U-Net"]:
            t = 0.2
            model = models[model_name].to(device)

            df_, ind_ = get_rows(df, c)

            imgs, masks = get_and_process_rows(df_, ind_)
            with torch.no_grad():
                out_ = model(imgs.to(device))
                out = torch.sigmoid(out_[1])
                out_args = torch.argmax(out[ind_, :, :, :], dim=0).cpu().numpy()

            img = imgs[ind_, 0, :, :].numpy()
            img = (img - img.min())  * 255. / (img.max() - img.min())
            img = np.array(img, dtype=np.uint8)

            img_rgb = np.stack([img, img, img], axis=-1)

            for i in range(3):
                mask_ = (out[ind_, i, :, :].cpu().numpy() > t)
                mask_ = np.array(mask_, dtype=bool)
                color = class_to_color[i]

                for c_ in range(3):
                    img_rgb[mask_,c_] = color[c_]

            ax[j].imshow(img_rgb)
            ax[j].set_title(model_name, fontsize=SIZE)

            j += 1
    except:
        pass
    
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    
    
    ### #2: HOLOW KERNELS
    
    # Parameters
    num_models = 8
    j = 0
    t = 0.5
    
    # Viz set-ups
    fig, ax = plt.subplots(1, num_models, figsize=(num_models*5, 5))
    for i in range(num_models):
        ax[i].axis("off")
    
    ### A1: Hollow kernels
    for model_name in models_and_kernels.keys():
        
        model = models_and_kernels[model_name]["model"]
        model = model.to(device)
        model_kernel = models_and_kernels[model_name]["kernel"]
        model_kernel = model_kernel.to(device)

        kernel = model_kernel(X_for_kernel.to(device))

        with torch.no_grad():
            out = model(X.to(device), kernel)
            out = torch.sigmoid(out)
            out_args = torch.argmax(out[0, :, :, :], dim=0).cpu().numpy()

        img_rgb = np.stack([img, img, img], axis=-1)

        for i in range(3):
            mask_ = (out[0, i, :, :].cpu().numpy() > t)# & (out_args == i)
            mask_ = np.array(mask_, dtype=bool)
            color = class_to_color[i]

            for c_ in range(3):
                img_rgb[mask_,c_] = color[c_]

        ax[j].imshow(img_rgb)
        ax[j].set_title(model_name, fontsize=SIZE)
        
        j += 1

    
    ### A2: Opt
    for model_name in models_A2.keys():
        with torch.no_grad():
            model = models_A2[model_name].to(device)
            model = model.to(device)
            
            out_ = model(X.float().to(device))
            out = torch.sigmoid(out_)
            out_args = torch.argmax(out[0, :, :, :], dim=0).cpu().numpy()

        img_rgb = np.stack([img, img, img], axis=-1)

        for i in range(3):
            mask_ = (out[0, i, :, :].cpu().numpy() > t) & (out_args == i)
            mask_ = np.array(mask_, dtype=bool)
            color = class_to_color[i]

            for c_ in range(3):
                img_rgb[mask_,c_] = color[c_]

            ax[j].imshow(img_rgb)
            ax[j].set_title(model_name, fontsize=SIZE)
            
        j+=1
        
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    


