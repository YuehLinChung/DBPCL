# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch
from src.transform import ContrastiveTransformations, MoCov1Transformation, MoCov2Transformation
from src.model import SimCLR, Supervised, DBCLR
from src.utils import LitProgressBar, CheckpointOnEpochs
import os
from torch.utils.data import DataLoader
from torchvision import datasets

# torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

#%% setup
workers = 0
data_folder = "C:\Dataset"
# data_folder = "C:/Dataset/tiny-imagenet-200/train/"

dataset = "cifar100"#cifar10, cifar100, stl10, tiny-imagenet
batch_size = 64
# crop_size = 32
epochs = 200

arch = "resnet32"
first_conv = False
maxpool1 = False
seed = 1

exp_dir = "lightning_logs/CIFAR100"
exp_name = "DBCLR_cifar100_res32_bs64_eps35_round1"
ckpt_path = ""

print("\n%s\n"%exp_name)
#%%######################################################################
if __name__ == '__main__':
    pl.seed_everything(seed, workers=True)
    if dataset in ["cifar10", "cifar100"]:
        transform_train = ContrastiveTransformations(n_views=2, crop_size=32, gaussian_blur=False, jitter_strength=0.5)
    elif dataset in ["tiny"]:
        transform_train = ContrastiveTransformations(n_views=2, crop_size=64, gaussian_blur=True, jitter_strength=1)
    elif dataset in ["stl10"]:
        transform_train = ContrastiveTransformations(n_views=2, crop_size=96, gaussian_blur=True, jitter_strength=1)
        
    if dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(data_folder,transform=transform_train)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(data_folder,transform=transform_train)
    elif dataset == 'stl10':
        dataset_train = datasets.STL10(data_folder, split='unlabeled', transform=transform_train)
    elif dataset == 'tiny-imagenet':
        dataset_train = datasets.ImageFolder(data_folder, transform=transform_train)
    
    # imbalance setup
    # =============================================================================
    # import numpy as np
    # indices = np.zeros(len(dataset_train), dtype=bool)
    # num_classes = len(dataset_train.classes)
    # ratio = 0.01
    # picked_number = ([int(np.exp(np.log(ratio)/(num_classes-1)*(c))*500) for c in range(num_classes)])
    # for c in range(len(dataset_train.classes)):
    #     indices[np.random.choice(np.arange(len(dataset_train))[np.array(dataset_train.targets)==c],picked_number[c], replace=False)] = True
    # dataset_train.data = dataset_train.data[indices]
    # dataset_train.targets = np.array(dataset_train.targets)[indices].tolist()
    # =============================================================================
    
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=workers, drop_last=True, persistent_workers=not workers==0)
    # pin_memory=True, prefetch_factor=5, persistent_workers=True
    # loader_val = DataLoader(tinyimagenet_val,batch_size=batch_size,num_workers=num_workers)
    
    #%%
    model = DBCLR(
        proj_hidden_dim=256,
        proj_out_dim=128,
        lr=0.01,
        temperature=0.1,
        weight_decay=1e-4,
        arch=arch,
        first_conv = first_conv,
        maxpool1 = maxpool1,
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
    trainer = pl.Trainer(max_epochs=epochs, deterministic=True, precision=16, gpus=1, callbacks=[bar, model_saver],logger=logger)
    trainer.fit(model, train_dataloaders=loader_train,ckpt_path=ckpt_path)#, val_dataloaders=loader_val)

