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
import shutil
import cv2
import argparse
import PIL
import time
import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd
from cnn_utils import dloader, load_dataset

# create directory to save train model performances
if os.path.isdir('resnext50') == True:
    shutil.rmtree("resnext50")
    os.makedirs("resnext50")
else:
    os.makedirs("resnext50")

# Download data
gdd.download_file_from_google_drive(file_id='1uVWstyM_-yfwrsFx9TAZSMBMhjL0ukis', dest_path='./data/LRHR_data_normalized', unzip=False)

if os.path.isdir('data/classifier_data') == False:
    os.mkdir("data/classifier_data")
os.system("unzip 'data/LRHR_data_normalized' -d './data/classifier_data'")

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
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnext50_lrhr_v1.pt')
    
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
            
    best_valid_loss = float('inf')
    
    
    # Load the dataset
    print('*** Create data loader ***')
    train_data_path = 'data/classifier_data/LRHR_data_normalized/train/'  
    val_data_path = 'data/classifier_data/LRHR_data_normalized/val/'
    test_data_path = 'data/classifier_data/LRHR_data_normalized/test/'
    train_loader, val_dataloader, test_dataloader = load_dataset(train_data_path, val_data_path, test_data_path, batch_size=args.batch_size, 
                                                                 resize_to=(256,256), cuda=True, gpu=args.gpu, world_size=world_size, rank=rank)
    
    print('*** Create model ***')
    model = models.resnext50_32x4d(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 2)
    
    print(cuda)
    if cuda==True:
        torch.cuda.set_device(args.gpu)
        
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # loss function
    loss_fcn = torch.nn.CrossEntropyLoss()
    
    # Start training
    print('*** Start training ***')
    step = 0
    model.train()
    
    # DDP
    model = model.to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model.to(args.gpu), device_ids=[args.gpu])
    
    losses = []
    for epoch in range(args.n_epochs):
        print("Epoch:", epoch)
        for iter, (data, labels) in enumerate(train_loader):
            
            
            optimizer.zero_grad()
            target = model(data)
            labels = labels.to('cuda')
            loss = loss_fcn(target, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # testing
            step += 1
            
            start_time = time.time()
            if step % args.eval_every == 0:
                
                val_loss, val_acc = test(val_dataloader, model, loss_fcn)
                print(val_loss, val_acc)
                print(
                    "Step {:05d} | Train loss {:.4f} | Over {} | Val loss {:.4f} |"
                    "Val acc {:.4f}".format(
                        step,
                        np.mean(losses),
                        len(losses),
                        val_loss,
                        val_acc,
                    ))
                with open('resnext50/training.txt', 'a') as training:
                    training.write('Step {:05d} | Train loss {:.4f} | Over {} | Val loss {:.4f} | Val acc {:.4f}\n'.format(step,np.mean(losses),len(losses),val_loss,val_acc))
                    training.close()
                    
                model.train()
                
                train_time = time.time() - start_time
                with open('resnext50/time.txt', 'a') as training:
                    training.write('Step {:05d} | {}\n'.format(step,train_time))
                    training.close()
                
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    
def test(data_loader, model, loss_fcn):
    """
    Testing
    :param data_loader: (data.Dataloader)
    :param model: (Model)
    :param loss_fcn: (torch.nn loss)
    :return: loss, accuracy
    """
    model.eval()
    losses = []
    accuracies = []
    for iter, (data, labels) in enumerate(data_loader):

        target = model(data)
        labels = labels.to('cuda')
        loss = loss_fcn(target, labels)
        losses.append(loss.item())

        _, indices = torch.max(target, dim=1)
        correct = torch.sum(indices == labels)
        accuracies.append(correct.item() * 1.0 / len(labels))

    return np.mean(losses), np.mean(accuracies)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ResNext50')
    
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay",type=float,default=5e-4,help="Weight for L2 loss")
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
    os.environ['MASTER_PORT'] = '8800' 
    mp.spawn(main(0,args), nprocs= gpus, args = Args())