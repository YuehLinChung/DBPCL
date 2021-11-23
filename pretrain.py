# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import SimCLR
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.progress import ProgressBar, ProgressBarBase
class LitProgressBar(ProgressBarBase):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.train_loss = 0
        self.train_avg_loss = 0
        self.val_loss = 0
        self.epoch_idx = 0
        self.train_logger = None
    def disable(self):
        self.enable = False
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # self.lightning_module, outputs, batch, batch_idx, dataloader_idx
        # super().on_train_batch_end(trainer, pl_module, outputs)  # don't forget this :)
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.train_loss += outputs['loss'].detach().cpu().item()
        self.train_avg_loss = self.train_loss / (batch_idx + 1)
        self.train_logger.set_postfix({'loss': self.train_avg_loss})
        self.train_logger.update(1)
    
    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.val_loss = 0
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.val_loss += outputs.detach().item()
        if self.train_logger:
            self.train_logger.set_postfix({'loss':self.train_avg_loss, 'val_loss':self.val_loss/(batch_idx+1)})
            # self.train_logger.set_postfix_str(self.train_logger.postfix + ', val_loss=%.2f'%(self.val_loss/(batch_idx+1)))
        
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.train_logger:
            self.train_logger.close()
        self.train_logger = tqdm(total=trainer.num_training_batches, ascii=True, desc='epoch %2d'%self.epoch_idx, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        self.epoch_idx += 1
        self.train_logger.reset()
        # self.train_logger.set_description('training epoch %2d'%self.epoch_idx)
        self.train_loss = 0
    def close_logger(self):
        if self.train_logger:
            self.train_logger.close()
            self.train_logger = None
        
def sinkhorn(self, Q, nmb_iters=3):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        K, B = Q.shape

        if self.gpus > 0:
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
        else:
            u = torch.zeros(K)
            r = torch.ones(K) / K
            c = torch.ones(B) / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)

            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=32),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False

#%% load data
# dm.setup('fit')
# cifar10_normalization()
batch_size = 256
num_workers = 6
transform_train = ContrastiveTransformations(contrast_transforms, n_views=2)

# mnist_train = datasets.MNIST('./',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), cifar10_normalization()]))
# mnist_test = datasets.MNIST('./',train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),cifar10_normalization()]))

cifar10_train = datasets.CIFAR10('./',transform=transform_train)
# cifar10_test = datasets.CIFAR10('./',train=False, transform=transform_test)

# cifar100_train = datasets.CIFAR100('./',transform=transform_train)
# cifar100_train = datasets.CIFAR100('./',transform=Compose([ToTensor(), cifar10_normalization()]))
# cifar100_test = datasets.CIFAR100('./',train=False, transform=Compose([ToTensor(), cifar10_normalization()]))

# datasets.INaturalist
# semi_iNat_unlabeled = ImageFolder('C:/Dataset/semi-inat-2021/u_train/', transform=transform_train, target_transform=lambda x:-1)

# voc07_train = datasets.VOCDetection('./', year='2007', image_set='train',transform=Compose([Resize(size=(500,500)), ToTensor(), cifar10_normalization()]))
# voc07_test = datasets.VOCDetection('./', year='2007', image_set='val',transform=Compose([Resize(size=(500,500)),ToTensor(), cifar10_normalization()]))

# svhn_train = datasets.SVHN('./', split='train',transform=Compose([ToTensor(), cifar10_normalization()]))
# svhn_test = datasets.SVHN('./', split='test',transform=Compose([ToTensor(), cifar10_normalization()]))

# stl10_unlabeled = datasets.STL10('C:\Dataset', split='unlabeled', transform=transform_train)
# stl10_train = datasets.STL10('C:\Dataset', split='train', transform=SwAVTrainDataTransform(size_crops=[32, 16], nmb_crops=[2,4],normalize=cifar10_normalization()))
# stl10_test = datasets.STL10('C:\Dataset', split='test', transform=SwAVEvalDataTransform(size_crops=[32, 16], nmb_crops=[2,4],normalize=cifar10_normalization()))
# =============================================================================
# from torch.utils.data import Subset
# import torch
# indices = torch.randperm(len(cifar10_train))[:5120]
# cifar10_train_ = Subset(cifar10_train, indices)
# =============================================================================
loader_train = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True,num_workers=num_workers,persistent_workers=True,pin_memory=True)# pin_memory=True, prefetch_factor=5, persistent_workers=True
# loader_val = DataLoader(cifar10_test,batch_size=batch_size,num_workers=num_workers, pin_memory=True)

#%%
model = SimCLR(hidden_dim=128, lr=0.01, temperature=0.1, weight_decay=1e-4, arch="resnet50")
#%%fit
logger = pl.loggers.TensorBoardLogger(os.getcwd(), version=None, name="lightning_logs")
bar = LitProgressBar()
# torch.manual_seed(0)
# pl.seed_everything(0, workers=True)
# limit_train_batches=0.1
trainer = pl.Trainer(max_epochs=100, precision=16, gpus=1, callbacks=[bar],logger=logger)
trainer.fit(model, train_dataloaders=loader_train)#, val_dataloaders=loader_val)
bar.close_logger()

# trainer.fit(model, train_dataloader=loader_train, val_dataloaders=loader_val)

# trainer.fit(model, datamodule=dm)

#%%
trainer.save_checkpoint('ep100_res50_cifar10_bs256_SimCLR.ckpt')