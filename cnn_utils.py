import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import sklearn
import pickle 
import numpy as np
import pickle as pkl
import os

from google_drive_downloader import GoogleDriveDownloader as gdd

import cv2
import argparse
import PIL
import time

# Make dataloader

def dloader(fp):
    """custom dataloader to workaround thresholding bug in PIL"""
    im = PIL.Image.open(fp)
    return im

def load_dataset(train_data_path, val_data_path, test_data_path, batch_size, resize_to, cuda, gpu, world_size, rank):

    ts = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ])
        
    train_dataset = ImageFolder(root=train_data_path, transform=ts, loader=dloader)
    test_dataset = ImageFolder(root=test_data_path, transform=ts, loader=dloader)
    val_dataset = ImageFolder(root=val_data_path, transform=ts, loader=dloader)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=world_size,rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,num_replicas=world_size,rank=rank)

    train_iterator = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=0,shuffle=False,
                                                pin_memory=True, sampler=train_sampler)
    val_iterator = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=0,shuffle=False,
                                                pin_memory=True, sampler=val_sampler)
    test_iterator = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=0,shuffle=False,
                                               pin_memory=True, sampler=test_sampler)
    
    return train_iterator, val_iterator, test_iterator