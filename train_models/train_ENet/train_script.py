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

sys.path.insert(0, "../../models")
from ENet import ENet

sys.path.insert(0, "..")
from utils import preprocess_dataframe_unet

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', '--cuda', dest='cuda', type=bool, default=True,
                        help='If cuda is available')
    parser.add_argument('-d', '--device', dest='device', type=int, default=0,
                        help='Cuda device number')
    
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--folder_to_save', dest='folder_to_save', type=str, default='weights_folder_new',
                        help='Save model to the folder as a .pth file')
    
    parser.add_argument('-i', '--image-size', dest='image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('-a', '--aug', dest='aug', type=bool, default=True,
                        help='If apply augmentation')
    
    parser.add_argument('--crop-shift', action='store',
                        default=None, help='Shift for crops' )
    parser.add_argument('-pw', '--pos-weight', dest='pos_weight', action='store',
                        default=(84., 32., 224.),
                        help='In BCE loss: pos-weight', nargs=3)
    parser.add_argument('-w', '--weight', dest='weight', default=(12.9, 5.0, 34.3),
                        help='In BCE loss: weight', nargs=3)
    parser.add_argument('--stats', metavar=('MEAN', 'STD'), default=(0.273035, 0.0668011),
                        help='Statistics for Normalization', nargs=2)
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-sh', '--sheduler', metavar='sh', type=bool, default=True,
                        help='If apply sheduler', dest='if_sheduler')
    parser.add_argument('-shs', '--sheduler_step', metavar='sheduler_step_size', type=int, default=50,
                        help='Sheduler step size', dest='sheduler_step_size')
    
    parser.add_argument('-bw', '--bce_weight', metavar='bce_weight', type=float, default=0.1,
                        help='BCE weight', dest='bce_weight')
    
    parser.add_argument('-p', '--print_per_iter', metavar='print_per_iter', type=int, default=10,
                        help='Print per the number of iterations', dest='print_per_iter')
    parser.add_argument('-N', '--num_models_to_save', metavar='num_models_to_save', type=int, default=2,
                        help='Number of models to save', dest='num_models_to_save')
    parser.add_argument('-es', '--earlly_stopping', metavar='earlly_stopping', type=bool, default=False,
                        help='If apply earlly stopping', dest='if_earlly_stopping')
    parser.add_argument('-g', '--earlly_stopping_gap', metavar='earlly_stopping_gap', type=float, default=2.,
                        help='Earlly stopping gap', dest='earlly_stopping_gap')


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
    print(args)
    # Device
    if args.cuda:
        device = "cuda:{}".format(args.device)
    else:
        device = "cpu"
    device = torch.device(device)
    print(device, args.device)
    
    # Data
    df = pd.read_csv("../data/dataframes/patients_data.csv")
    df_train_test = pd.read_csv("../data/dataframes/train_test_split_unet.csv")

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
        tr_dual, tr_img = None, None

    # Dataloader
    print("-"*100)
    BATCH_SIZE = args.batchsize
    image_datasets = {x: MRI_Simple_Dataset(dfs[x], x, args.image_size, num_classes=3,
                                            tr_dual=tr_dual[x], tr_img=tr_img[x], HALF_CROP=int(args.crop_shift))
                          for x in phase_range}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                     shuffle=True, num_workers=4)
                      for x in phase_range}
    dataset_sizes = {x: len(image_datasets[x]) for x in phase_range}
    print(dataset_sizes)
    
    # Viz
    for X_batch, y_batch in dataloaders['train']:
        for i in range(X_batch.shape[0]):
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            ax[0].imshow(X_batch[i, 0, :, :].numpy() / 255. , cmap="bone")

            ax[1].imshow(y_batch[i, 0, :, :], cmap="bone")
            ax[2].imshow(y_batch[i, 1, :, :], cmap="bone")
            ax[3].imshow(y_batch[i, 2, :, :], cmap="bone")
            plt.savefig("ex.png")
            break
        break

        
    # Model
    model = ENet(in_channels=1, num_classes=3)
    
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=torch.device("cpu")))
    
    # Train 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    if args.if_sheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.sheduler_step_size, gamma=0.1)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # Weights 
    pos_weight=[float(args.pos_weight[0]), float(args.pos_weight[1]), float(args.pos_weight[2])]
    pos_weight=torch.tensor(pos_weight).reshape((1, 3, 1, 1)).to(device) / 3
    weight=[float(args.weight[0]), float(args.weight[1]), float(args.weight[2])]
    weight=torch.tensor(weight).reshape((1, 3, 1, 1)).to(device)
    
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

    
    
    
    
    
    
    
