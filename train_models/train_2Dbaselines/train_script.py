import os
import sys
import argparse
import logging

import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from albumentations import (HorizontalFlip, VerticalFlip, Rotate, MotionBlur, MedianBlur, Blur,
                            Compose, OneOf, ElasticTransform, GridDistortion, OpticalDistortion,
                            CropNonEmptyMaskIfExists, Resize, Normalize, CenterCrop)

from functional import MRI_Simple_Dataset, train_model

sys.path.insert(0, "../..")
from utils import preprocess_dataframe_unet

sys.path.insert(0, "../../models")
from ENet import ENet
from UNet import UNet
from UNet_PD import *

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', '--cuda', type=bool, default=True,
                        help='If cuda is available')
    parser.add_argument('-d', '--device', dest='device', type=int, default=0,
                        help='Cuda device number')

    parser.add_argument('-m', '--model', type=str, default="Original",
                        help='The model cofiguration, can be "UNet", "ENet", "Original", "Baseline", "Dilated", "ProgressiveDilated"')
    
    parser.add_argument('-f', '--load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--folder_to_save', dest='folder_to_save', type=str, default='weights_folder_new',
                        help='Save model to the folder as a .pth file')
    
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of target classes')
    parser.add_argument('-i', '--image-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('-a', '--aug', type=bool, default=True,
                        help='If apply augmentation')
    
    parser.add_argument('--crop-shift', action='store',
                        default=100, help='Shift for crops' )
    parser.add_argument('-pw', '--pos-weight', action='store',
                        default=(84., 32., 224.),
                        help='In BCE loss: pos-weight', nargs=3)
    parser.add_argument('-w', '--weight', default=(12.9, 5.0, 34.3),
                        help='In BCE loss: weight', nargs=3)
    parser.add_argument('--stats', metavar=('MEAN', 'STD'), default=(0.273035, 0.0668011),
                        help='Statistics for Normalization', nargs=2)
    
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=8,
                        help='Batch size')
    parser.add_argument('-lr', '--learning-rate', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-sh', '--sheduler', type=bool, default=True,
                        help='If apply sheduler', dest='if_sheduler')
    parser.add_argument('-shs', '--sheduler_step', type=int, default=50,
                        help='Sheduler step size', dest='sheduler_step_size')
    
    parser.add_argument('-bw', '--bce_weight', type=float, default=0.1,
                        help='BCE weight')
    
    parser.add_argument('-p', '--print-per-iter', type=int, default=10,
                        help='Print per the number of iterations')
    parser.add_argument('-N', '--num-models-to-save', type=int, default=2,
                        help='Number of models to save')


    return parser.parse_args()

def init_dir(model, weight_folder, args):

    # Init
#     arch_name = model.__module__
    arch_name = model.__class__.__name__
    model_full_name = "{}_{}".format(arch_name, datetime.now().strftime('%d.%m.%Y.%H:%M'))
    
    model_dir = os.path.join(weight_folder, model_full_name)
    
    # Make dirs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Details file
    details_file = model_dir + "/details.log"
    logging.basicConfig(filename=details_file, level=logging.INFO)
    logging.info("{}".format(args))

    return model_dir


if __name__ == '__main__':
    
    args = get_args()

    # Device
    if args.cuda:
        device = "cuda:{}".format(args.device)
    else:
        device = "cpu"
    device = torch.device(device)
    
    # Data
    df = pd.read_csv("../../../data/dataframes/patients_data.csv")
    df_train_test = pd.read_csv("../../../data/dataframes/train_test_split_unet.csv")

    df["if_mask"] = df.roi_name.apply(lambda x: True if isinstance(x, str) else False)
    df, dfs = preprocess_dataframe_unet(df, df_train_test)
    
    phase_range = ["train", "val"]

    # Transforms
    crop_coeff = max(int(args.image_size / 256), 1)
    if args.aug:
        tr_dual = {
            "train": Compose([ 

                VerticalFlip(),
                Rotate(20, p=0.4),

                OneOf([
                    ElasticTransform(p=0.2, alpha=120, sigma=120*0.1, alpha_affine=120 * 0.1),
                    GridDistortion(p=0.2, num_steps=10, distort_limit=0.3),
                    OpticalDistortion(p=0.2, distort_limit=0.1, shift_limit=0.05),
                ], p=0.2),  

                Resize(args.image_size, args.image_size)
            ]),
            "val": None
        }

        tr_img = {
            "train": Compose([ 
                OneOf([
                    MotionBlur(p=.2),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                
                Normalize(mean=args.stats[0], std=args.stats[1], max_pixel_value=1, p=1.0),
            ]),
            "val": Normalize(mean=args.stats[0], std=args.stats[1], max_pixel_value=1, p=1.0),
        }
    else:
        tr_dual, tr_img = {"train": None, "val": None}, {"train": None, "val": None}

    # Dataloader
    BATCH_SIZE = args.batch_size

    image_datasets = {x: MRI_Simple_Dataset(dfs[x], x, args.image_size, num_classes=args.num_classes,
                                            tr_dual=tr_dual[x], tr_img=tr_img[x], HALF_CROP=int(args.crop_shift))
                          for x in phase_range}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                     shuffle=True, num_workers=8)
                      for x in phase_range}
    dataset_sizes = {x: len(image_datasets[x]) for x in phase_range}
        
    # Model
    NAME2MODEL = {
        "ENet": ENet(in_channels=1, num_classes=args.num_classes),
        "UNet": UNet(in_channels=1, out_channels=args.num_classes),
        "Original": UNet_Original(in_channels=1, out_channels=args.num_classes),
        "Original_with_BatchNorm": UNet_Original_with_BatchNorm(in_channels=1, out_channels=args.num_classes),
        "Baseline": UNet_Baseline(in_channels=1, out_channels=args.num_classes),
        "Dilated": UNet_Dilated(in_channels=1, out_channels=args.num_classes),
        "ProgressiveDilated": UNet_ProgressiveDilated(in_channels=1, out_channels=args.num_classes),
    }
    model = NAME2MODEL[args.model]

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=torch.device("cpu")))
    
    # Train 

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    if args.if_sheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.sheduler_step_size, gamma=0.1)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    pos_weight=[float(args.pos_weight[0]), float(args.pos_weight[1]), float(args.pos_weight[2])]
    pos_weight=pos_weight[:args.num_classes]
    pos_weight=torch.tensor(pos_weight).reshape((1, args.num_classes, 1, 1)).to(device) / 3
    weight=[float(args.weight[0]), float(args.weight[1]), float(args.weight[2])]
    weight=weight[:args.num_classes]
    weight=torch.tensor(weight).reshape((1, args.num_classes, 1, 1)).to(device)
    
    # Init dir
    weight_folder = args.folder_to_save
    model_dir = init_dir(model, weight_folder, args)
    
    # Train
    start_train = time.time()
    try:
        train_model(model.to(device), dataloaders, 
                    phase_range, optimizer, scheduler, 
                    num_epochs=args.epochs, 
                    model_dir=model_dir, 
                    bce_weight=args.bce_weight, 
                    weight=weight, 
                    pos_weight=pos_weight, 
                    num_models_to_save=args.num_models_to_save, 
                    print_per_iter=args.print_per_iter, 
                    device=device, 
                    phase_to_save=["train", "val"])
    
    except Exception as e:
        print(e)
    finally:
        if time.time() - start_train < 2 * 60.:
            print("-> Remove folder!")
            for folder in os.listdir(model_dir):
                path_folder = os.path.join(model_dir, folder)
                os.remove(path_folder)
            os.removedirs(model_dir)

    
    
    
    
    
    
    
