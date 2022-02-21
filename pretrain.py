# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch
from src.transform import ContrastiveTransformations
from src.model import SimCLR, Supervised, DBCLR
from src.utils import LitProgressBar, CheckpointOnEpochs
import os
from torch.utils.data import DataLoader
from torchvision import datasets

# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#%% load data
# dm.setup('fit')
# cifar10_normalization()
batch_size = 256
num_workers = 8
transform_train = ContrastiveTransformations(n_views=2, crop_size=96)

# mnist_train = datasets.MNIST('./',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), cifar10_normalization()]))
# mnist_test = datasets.MNIST('./',train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),cifar10_normalization()]))

# cifar10_train = datasets.CIFAR10('../SwAV',transform=transform_train)
# cifar10_test = datasets.CIFAR10('../SwAV',train=False, transform=transform_test)

# cifar100_train = datasets.CIFAR100('./',transform=transform_train)
# cifar100_train = datasets.CIFAR100('./',transform=Compose([ToTensor(), cifar10_normalization()]))
# cifar100_test = datasets.CIFAR100('./',train=False, transform=Compose([ToTensor(), cifar10_normalization()]))

# datasets.INaturalist
# semi_iNat_unlabeled = ImageFolder('C:/Dataset/semi-inat-2021/u_train/', transform=transform_train, target_transform=lambda x:-1)

# svhn_train = datasets.SVHN('./', split='train',transform=Compose([ToTensor(), cifar10_normalization()]))
# svhn_test = datasets.SVHN('./', split='test',transform=Compose([ToTensor(), cifar10_normalization()]))

stl10_unlabeled = datasets.STL10('../SwAV', split='unlabeled', transform=transform_train)
# stl10_train = datasets.STL10('C:\Dataset', split='train', transform=transform_train)
# stl10_test = datasets.STL10('C:\Dataset', split='test', transform=transform_train)

loader_train = DataLoader(stl10_unlabeled, batch_size=batch_size, shuffle=True,num_workers=num_workers)# pin_memory=True, prefetch_factor=5, persistent_workers=True
# loader_val = DataLoader(cifar10_test,batch_size=batch_size,num_workers=num_workers, pin_memory=True)

#%%
max_epochs = 1000
model = DBCLR(
    proj_hidden_dim=2048,
    lr=0.01,
    temperature=0.1,
    weight_decay=1e-4,
    arch="resnet50",
    max_epochs=max_epochs,
    fusion=False,
    fusion_on = 'proj_norm',
    concentration=0.3
    )
# model = SimCLR.load_from_checkpoint('ep1000_SimCLRFusion_res50_stl10_bs256_feats.ckpt')
# model = model.to(torch.float16)
#%%fit
logger = pl.loggers.TensorBoardLogger(os.getcwd(), version='DBCLR03_aggr', name="lightning_logs",default_hp_metric=False)
bar = LitProgressBar()
model_saver = CheckpointOnEpochs([100, 500, 1000], "ep{epoch}_DBCLR03_res50_stl10_bs256_aggr.ckpt")
# torch.manual_seed(0)
# pl.seed_everything(0, workers=True)
# limit_train_batches=0.1
# trainer = pl.Trainer(max_epochs=100, precision=16, gpus=1, callbacks=[bar],logger=logger,limit_train_batches=1.0)
trainer = pl.Trainer(max_epochs=max_epochs, precision=16, gpus=1, callbacks=[bar, model_saver],logger=logger)
trainer.fit(model, train_dataloaders=loader_train)#, ckpt_path="ep500_DBCLR03_res50_stl10_bs256_aggr.ckpt")#, val_dataloaders=loader_val)
bar.close_logger()
