# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch
from src.transform import ContrastiveTransformations#, MoCov1Transformation, MoCov2Transformation
from src.model import SimCLR, DBCLR
from src.utils import LitProgressBar, CheckpointOnEpochs
import os
from torch.utils.data import DataLoader
from torchvision import datasets

import argparse

parser = argparse.ArgumentParser(description="Training script for the unsupervised phase via self-supervision")
parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs")
parser.add_argument("--dataset", default="cifar100", help="Dataset: cifar10|100, tiny, stl10")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', help="Backbone: resnet|32|56|18|34|50")
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total.')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
parser.add_argument('--exp-dir', default="lightning_logs", type=str,help='experiment directory (default: lightning_logs)')
parser.add_argument('--exp-name', default="", type=str,help='experiment name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

#%% setup
workers = args.workers
data_folder = args.data

dataset = args.dataset
batch_size = args.batch_size
epochs = args.epochs

arch = args.arch
seed = args.seed

exp_dir = args.exp_dir
exp_name = args.exp_name if args.exp_name else "DBCLR_%s_%s_bs%d_round%d"%(dataset, arch, batch_size, seed)
checkpoint = args.checkpoint

#%%######################################################################
if __name__== "__main__":
    print("\n%s\n"%exp_name)
    pl.seed_everything(seed, workers=True)
    # cifar10_normalization()
    if dataset in ["cifar10", "cifar100"]:
        transform_train = ContrastiveTransformations(n_views=2, crop_size=32, gaussian_blur=False, jitter_strength=0.5)
    elif dataset in ["stl10", "tiny"]:
        transform_train = ContrastiveTransformations(n_views=2, crop_size=96, gaussian_blur=True, jitter_strength=1)
    
    if dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(data_folder,transform=transform_train)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(data_folder,transform=transform_train)
    elif dataset == 'stl10':
        dataset_train = datasets.STL10(data_folder, split='unlabeled', transform=transform_train)
    elif dataset == 'tiny-imagenet':
        dataset_train = datasets.ImageFolder(data_folder, transform=transform_train)
    
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=workers, drop_last=True, persistent_workers=not workers==0)
    # pin_memory=True, prefetch_factor=5, persistent_workers=True
    # loader_val = DataLoader(tinyimagenet_val,batch_size=batch_size,num_workers=num_workers)
    
    #%%
    model = DBCLR(
        proj_hidden_dim=256,
        proj_out_dim=128,
        lr = args.lr,
        temperature=0.1,
        weight_decay=1e-4,
        arch=arch,
        max_epochs=epochs,
        concentration=0.1,
        con_estimation = True,
        warmup_epoch=0,
        eps=[0.3,0.5],
        # eps=[10,20,30],
        use_mlp = True
        )
# model = DBCLR.load_from_checkpoint('models/ep100_DBCLR03_res50_stl10_bs256.ckpt')
#%%fit
    logger = pl.loggers.TensorBoardLogger(os.getcwd(), version=exp_name, name=exp_dir)
    bar = LitProgressBar()
    model_saver = CheckpointOnEpochs([200], "ep{epoch}_%s.ckpt"%(exp_name))
    
    # limit_train_batches=0.1
    trainer = pl.Trainer(max_epochs=epochs, precision=16, gpus=1, callbacks=[bar, model_saver],logger=logger)
    trainer.fit(model, train_dataloaders=loader_train,ckpt_path=checkpoint)
