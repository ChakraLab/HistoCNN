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
import shutil
from imutils import paths
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import tarfile
import argparse

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
from google_drive_downloader import GoogleDriveDownloader as gdd
from unet_utils import SegmentationDataset, load_dataset, DoubleConv, DownBlock, UpBlock, UNet

os.system("pip install -q -U segmentation-models-pytorch albumentations > /dev/null")
import segmentation_models_pytorch as smp

# create directory to save train model performances
if os.path.isdir('unet') == True:
    shutil.rmtree("unet")
    os.makedirs("unet")
else:
    os.makedirs("unet")
    
# Download data
gdd.download_file_from_google_drive(file_id='1D-dWkJqaIieYPo8EOLkysLLkaFxLZZa3', dest_path='./data/trainUnet', unzip=False)
if os.path.isdir('data/segmentation_data') == False:
    os.mkdir("data/segmentation_data")
    
file = tarfile.open('data/trainUnet')
file.extractall('./data/segmentation_data')
file.close()
    
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
    
    image_path = 'data/segmentation_data/train_Unet/LRHR_Bracs20X_images_H/' 
    mask_path = 'data/segmentation_data/train_Unet/LRHR_ductMasks/' 

    imagePaths = sorted(list(paths.list_images(image_path)))
    maskPaths = sorted(list(paths.list_images(mask_path)))

    val_split = 0.25
    split = train_test_split(imagePaths, maskPaths,test_size=val_split, random_state=42)

    (trainImages, valImages) = split[:2]
    (trainMasks, valMasks) = split[2:]
    
    train_loader, val_loader = load_dataset(trainImages, valImages, trainMasks, valMasks, batch_size=32, 
                                        resize_to=(112,112), cuda=True, gpu=gpu, world_size=world_size, rank=rank)
    
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
        start_time = time.time()
        print("Epoch:", epoch)
        train_logs = train_epoch.run(train_loader)
        print(train_logs)
        valid_logs = valid_epoch.run(val_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        np.save('unet/train_log_list.npy', train_logs_list)
        np.save('unet/val_log_list.npy', valid_logs_list)
        
        train_time = time.time() - start_time
        with open('unet/time.txt', 'a') as training:
                    training.write('Step {:05d} | {}\n'.format(step,train_time))
                    training.close()
                    
        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, 'models/unet_duct_v1.pth')
            print('Model saved!')
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='UNet')
    
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--n-epochs",type=int,default=500,help="number of epochs")
    parser.add_argument("--batch-size",type=int,default=32,help="batch size")
    parser.add_argument("--eval-every",type=int,default=50,help="eval model every N steps")
    
    # Data distributed parallel training params
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--gpus", type=int, default=1, help="gpus")
    parser.add_argument("--nodes", type=int, default=1, help="nodes")
    parser.add_argument("--nr", type=int, default=0, help="nr")
    
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8820' 
    mp.spawn(main(0,args), nprocs= gpus, args = Args())