# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""

import pytorch_lightning as pl
import torch.nn.functional as F
from pl_bolts.models.self_supervised import SwAV
from src.model import SimCLR

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

from pl_bolts.transforms.dataset_normalizations import stl10_normalization, cifar10_normalization
# import sys
import torch
# import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
#%% load data
# dm.setup('fit')

batch_size = 256
num_workers = 0
# Normalize((0.5,), (0.5,))
# cifar10_normalization()
normalization = cifar10_normalization()

# mnist_train = datasets.MNIST('../SwAV',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
# mnist_test = datasets.MNIST('../SwAV',train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),normalization]))

# 62 classes
# emnist_train = datasets.EMNIST('../SwAV',split='byclass',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]), download=True)
# emnist_test = datasets.EMNIST('../SwAV',split='byclass',train=False,transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]), download=True)

cifar10_train = datasets.CIFAR10('../SwAV',transform=Compose([ToTensor(), normalization]))
cifar10_test = datasets.CIFAR10('../SwAV',train=False, transform=Compose([ToTensor(), normalization]))

# cifar100_train = datasets.CIFAR100('../SwAV',transform=Compose([ToTensor(), normalization]))
# cifar100_test = datasets.CIFAR100('../SwAV',train=False, transform=Compose([ToTensor(), normalization]))

# stl10_train = datasets.STL10('C:/Dataset', split='train', transform=Compose([ToTensor(),normalization]))
# stl10_test = datasets.STL10('C:/Dataset', split='test', transform=Compose([ToTensor(),normalization]))
# stl10_unlabel = DataLoader(datasets.STL10('C:/Dataset', split='unlabeled', transform=Compose([SwAVTrainDataTransform(size_crops=[64, 32], nmb_crops=[2,4])])),batch_size=batch_size, shuffle=True)

# voc07_train = datasets.VOCDetection('./', year='2007', image_set='train',transform=Compose([Resize(size=(500,500)), ToTensor(), cifar10_normalization()]))
# voc07_test = datasets.VOCDetection('./', year='2007', image_set='val',transform=Compose([Resize(size=(500,500)),ToTensor(), cifar10_normalization()]))

# svhn_train = datasets.SVHN('../SwAV', split='train',transform=Compose([ToTensor(), normalization]))
# svhn_test = datasets.SVHN('../SwAV', split='test',transform=Compose([ToTensor(), normalization]))

loader_train = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True,num_workers=num_workers)#,persistent_workers=True)
loader_val = DataLoader(cifar10_test,batch_size=batch_size,num_workers=num_workers)#,persistent_workers=True)

#%% train linear
model = SwAV.load_from_checkpoint('../SwAV/ep100_res50_proto300_cifar10_bs256.ckpt')
# model = SimCLR.load_from_checkpoint('ep1000_SimCLRFusion_res50_cifar10_bs256.ckpt')
# model = SimCLR.load_from_checkpoint('ep100_SimCLRFusion_res18_cifar10_bs1024.ckpt')

device = torch.device('cuda:0')
model = model.to(device)
model.freeze()
#%%visualize (t-SNE)
# knn = KNN(n_neighbors=20)
feats_train = []
labels_train = []
feats_val = []
labels_val = []
torch.manual_seed(0)
# model2 = SimCLR.load_from_checkpoint('ep100_SimCLR_res50_cifar10_bs256.ckpt')

for batch, label in loader_train:
    batch = batch.to(device)
    labels_train.append(label.cpu().numpy())
    
    feat = model(batch).cpu().numpy()
    # feat = model2(batch)[0].cpu().numpy()
    feats_train.append(feat)

for batch, label in loader_val:
    batch = batch.to(device)
    labels_val.append(label.cpu().numpy())
    
    feat = model(batch).cpu().numpy()
    # feat = model2(batch)[0].cpu().numpy()
    feats_val.append(feat)
x_train = np.vstack(feats_train)
y_train = np.hstack(labels_train)
x_val = np.vstack(feats_val)
y_val = np.hstack(labels_val)

X_embedded = TSNE(n_components=2, init='pca', learning_rate='auto', n_jobs=-1,verbose=1).fit_transform(x_train)
df = pd.DataFrame(X_embedded,columns=['tsne-1','tsne-2'])
df['y'] = y_train
#%%
plt.figure()
sns.scatterplot(
    x='tsne-1', y='tsne-2',
    hue='y',
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.5
)
