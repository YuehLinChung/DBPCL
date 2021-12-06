# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.transform import ContrastiveTransformations
from src.model import SimCLR
from src.utils import LitProgressBar
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False

#%% load data
# dm.setup('fit')
# cifar10_normalization()
batch_size = 256
num_workers = 0
transform_train = ContrastiveTransformations(n_views=2)

# mnist_train = datasets.MNIST('./',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), cifar10_normalization()]))
# mnist_test = datasets.MNIST('./',train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),cifar10_normalization()]))

cifar10_train = datasets.CIFAR10('../SwAV',transform=transform_train)
# cifar10_test = datasets.CIFAR10('../SwAV',train=False, transform=transform_test)

# cifar100_train = datasets.CIFAR100('./',transform=transform_train)
# cifar100_train = datasets.CIFAR100('./',transform=Compose([ToTensor(), cifar10_normalization()]))
# cifar100_test = datasets.CIFAR100('./',train=False, transform=Compose([ToTensor(), cifar10_normalization()]))

# datasets.INaturalist
# semi_iNat_unlabeled = ImageFolder('C:/Dataset/semi-inat-2021/u_train/', transform=transform_train, target_transform=lambda x:-1)

# svhn_train = datasets.SVHN('./', split='train',transform=Compose([ToTensor(), cifar10_normalization()]))
# svhn_test = datasets.SVHN('./', split='test',transform=Compose([ToTensor(), cifar10_normalization()]))

# stl10_unlabeled = datasets.STL10('C:\Dataset', split='unlabeled', transform=transform_train)
# stl10_train = datasets.STL10('C:\Dataset', split='train', transform=SwAVTrainDataTransform(size_crops=[32, 16], nmb_crops=[2,4],normalize=cifar10_normalization()))
# stl10_test = datasets.STL10('C:\Dataset', split='test', transform=SwAVEvalDataTransform(size_crops=[32, 16], nmb_crops=[2,4],normalize=cifar10_normalization()))

loader_train = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)# pin_memory=True, prefetch_factor=5, persistent_workers=True
# loader_val = DataLoader(cifar10_test,batch_size=batch_size,num_workers=num_workers, pin_memory=True)

#%%
model = SimCLR(
    proj_hidden_dim=2048,
    lr=0.01,
    temperature=0.1,
    weight_decay=1e-4,
    arch="resnet50",
    fusion=True
    )
# model = SimCLR.load_from_checkpoint('ep100_SimCLRFusion_res50_cifar10_bs256.ckpt')
# model = model.to(torch.float16)
#%%fit
logger = pl.loggers.TensorBoardLogger(os.getcwd(), version=None, name="lightning_logs")
bar = LitProgressBar()
# torch.manual_seed(0)
# pl.seed_everything(0, workers=True)
# limit_train_batches=0.1
trainer = pl.Trainer(max_epochs=100, precision=32, gpus=1, callbacks=[bar],logger=logger,limit_train_batches=1.0)
trainer.fit(model, train_dataloaders=loader_train)#, val_dataloaders=loader_val)
bar.close_logger()

# trainer.fit(model, train_dataloader=loader_train, val_dataloaders=loader_val)

# trainer.fit(model, datamodule=dm)

#%%
# trainer.save_checkpoint('ep200_res50_proto300_cifar10_bs256_soft_sharpen.ckpt')
# torch.save(model, 'ep200_res50_proto300_cifar10_bs256_soft_sharpen.pth')