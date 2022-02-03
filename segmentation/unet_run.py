import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import imageio
import cv2
from matplotlib import cm
import pickle as pkl
import glob
import os
from imutils import paths
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

def main(gpu, args):
    
    ############################################################
    world_size = args.gpus * args.nodes
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    ############################################################
    
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        
    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'unet_v1_duct.pt')
    
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
            
    best_valid_loss = float('inf')
    
    data_path = os.path.join("../data/", "train")
    image_path = os.path.join(data_path, "images")
    mask_path = os.path.join(data_path, "masks")

    imagePaths = sorted(list(paths.list_images(image_path)))
    maskPaths = sorted(list(paths.list_images(mask_path)))

    val_split = 0.15
    split = train_test_split(imagePaths, maskPaths,test_size=val_split, random_state=42)

    (trainImages, valImages) = split[:2]
    (trainMasks, valMasks) = split[2:]
    
    train_loader, val_loader = load_dataset(trainImages, valImages, batch_size=32, 
                                        resize_to=(256,256), cuda=True, gpu=gpu, world_size=world_size, rank=rank)
    
    model = UNet().to('cuda')
    
    print(cuda)
    if cuda==True:
        torch.cuda.set_device(args.gpu)
        
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr),])
    
    loss_fcn = smp.utils.losses.DiceLoss()
    
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    
    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss_fcn, metrics=metrics, optimizer=optimizer,device="cuda",
    verbose=True,)

    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss_fcn, metrics=metrics, device="cuda",
    verbose=True,)
    
    # Start training
    print('*** Start training ***')
    step = 0
    
    # DDP
    model = model.to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model.to(args.gpu), device_ids=[args.gpu])
    
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []
    
    for epoch in range(args.n_epochs):
        print("Epoch:", epoch)
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')