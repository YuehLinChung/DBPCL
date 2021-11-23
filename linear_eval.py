# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch.nn.functional as F
from pl_bolts.models.self_supervised import SwAV
from src.model import SimCLR
from pl_bolts.datamodules import STL10DataModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVTrainDataTransform, SwAVEvalDataTransform
)

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

from pl_bolts.transforms.dataset_normalizations import stl10_normalization, cifar10_normalization
# import sys
from tqdm import tqdm
import torch
# import os
import numpy as np
import matplotlib.pyplot as plt

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13],dtype=np.int64)
    return coarse_labels[targets]

class LitProgressBar(pl.callbacks.progress.ProgressBarBase):

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
        self.train_logger = tqdm(total=trainer.num_training_batches, ascii=True, desc='epoch %2d'%self.epoch_idx)
        self.epoch_idx += 1
        # self.train_logger.reset()
        # self.train_logger.set_description('training epoch %2d'%self.epoch_idx)
        self.train_loss = 0
    def close_logger(self):
        if self.train_logger:
            self.train_logger.close()
        
class LinearNN(torch.nn.Module):
    def __init__(self, encoder, hidden_size, out_size, frozen=True):
        super().__init__()
        self.encoder = encoder
        
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.linear = torch.nn.Linear(hidden_size, out_size)
        self.frozen = frozen
    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                _x = self.encoder(x)
        else:
            _x = self.encoder.forward_backbone(x)
        return self.linear(_x)
#%%
# data
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# batch_size = 512
# dm = CIFAR10DataModule(data_dir=None, batch_size=batch_size,num_workers=0)
# dm.train_dataloader = dm.train_dataloader_mixed
# dm.train_dataloader = dm.train_dataloader
# dm.val_dataloader = dm.val_dataloader_mixed
# dm.val_dataloader = dm.val_dataloader
# dm.train_transforms = SwAVTrainDataTransform(size_crops=[32, 16], nmb_crops=[2,4])#, normalize=cifar10_normalization())

# dm.val_transforms = SwAVEvalDataTransform(size_crops=[32, 16], nmb_crops=[2,4])#,normalize=cifar10_normalization())

# train_loader = dm.train_dataloader()
# val_loader = dm.val_dataloader()
# test_loader = dm.test_dataloader()
# model
#%%
# =============================================================================
# model = SwAV(
#     arch='resnet18',
#     nmb_crops=[2,4],
#     nmb_prototypes=300,
#     # num_samples=dm.num_unlabeled_samples,
#     num_samples=dm.num_samples,
#     gpus=1,
#     dataset='cifar10',
#     batch_size=batch_size
# )
# =============================================================================

# =============================================================================
# model = model.load_from_checkpoint('ep100_res18_proto300_cifar10_bs1024.ckpt')
# =============================================================================

# fit
# bar = LitProgressBar()
# trainer = pl.Trainer(max_epochs=10, precision=16, gpus=1, callbacks=[bar])

# trainer.fit(model, train_loader, val_loader)

# =============================================================================
# trainer.fit(model, datamodule=dm)
# =============================================================================

# trainer.save_checkpoint('ch.ckpt')
#%% load data
# dm.setup('fit')
# cifar10_normalization()
batch_size = 256
num_workers = 0
# Normalize((0.5,), (0.5,))
cifar10_train = datasets.CIFAR10('../swav-main',transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
cifar10_test = datasets.CIFAR10('../swav-main',train=False, transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))

# cifar100_train = datasets.CIFAR100('./',transform=Compose([ToTensor(), cifar10_normalization()]))
# cifar100_test = datasets.CIFAR100('./',train=False, transform=Compose([ToTensor(), cifar10_normalization()]))
# stl10_train = datasets.STL10('./', split='train', transform=Compose([ToTensor()]))
# stl10_test = datasets.STL10('./', split='test', transform=Compose([ToTensor()]))
# stl10_unlabel = DataLoader(datasets.STL10('./', split='unlabeled', transform=Compose([SwAVTrainDataTransform(size_crops=[64, 32], nmb_crops=[2,4])])),batch_size=batch_size, shuffle=True)

loader_train = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)#,persistent_workers=True)
loader_val = DataLoader(cifar10_test,batch_size=batch_size,num_workers=num_workers, pin_memory=True)#,persistent_workers=True)

#%% train linear
# model2 = SwAV.load_from_checkpoint('ep1000_res18_proto300_cifar10_bs1024.ckpt')
model2 = SimCLR.load_from_checkpoint('ep100_SimCLR_res50_cifar10_bs256.ckpt')
# reset model
# model2.arch = "resnet18"
# model2.model = model2.init_model()
model = LinearNN(model2.convnet, 2048, 10, frozen=True)

# train from scratch
# model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=False)
# model.fc = torch.nn.Linear(512, 20)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device('cuda:0')
model = model.to(device)

for epoch in range(100):
    loss_avg = 0
    _loss_sum = 0
    loss_val_avg = 0
    batch_idx = 0
    correct = 0
    size = 0
    print('epoch: %d, '%(epoch +1), end='')
    # train
    for batch, label in loader_train:
        batch = batch.to(device)
        label = label.to(device)
        size += batch.shape[0]
        
        output = model(batch)
        # output = F.softmax(output, dim=1)
        # label_encoded = torch.zeros((label.shape[0],10)).scatter_(1,label.view(-1,1),1)
        loss = loss_fn(output, label)
        _loss_sum += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (output.argmax(1) == label).type(torch.float).sum().item()
        loss_avg = _loss_sum/(batch_idx+1)
        print('\repoch: %d, loss: %.3f, acc: %.3f'%((epoch +1), loss_avg, correct/size), end='')
        batch_idx += 1
    acc = correct/size
    # validation
    batch_idx = 0
    correct = 0
    size = 0
    with torch.no_grad():
        for batch, label in loader_val:
            batch = batch.to(device)
            label = label.to(device)
            output = model(batch)
            # output = F.softmax(output, dim=1)
            size += batch.shape[0]
            # label_encoded = torch.zeros((label.shape[0],10)).scatter_(1,label.view(-1,1),1)
            loss = loss_fn(output, label)
            loss_val_avg += loss.detach().item()
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
            print('\repoch: %d, loss: %.3f, acc: %.3f, loss_val: %.3f, acc_val: %.3f'%((epoch +1),loss_avg, acc, loss_val_avg/(batch_idx+1), correct/size), end='')
            batch_idx += 1
    print()
#%%visualize data
