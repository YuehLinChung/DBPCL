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
from sklearn.metrics import top_k_accuracy_score as topk

class LinearNN(torch.nn.Module):
    def __init__(self, encoder, hidden_size, out_size, frozen=True):
        super().__init__()
        self.encoder = encoder
        
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.freeze()
        self.linear = torch.nn.Linear(hidden_size, out_size)
        self.frozen = frozen
    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                _x = self.encoder(x)
        else:
            _x = self.encoder(x)
        return self.linear(_x)

#%% load data
# dm.setup('fit')

torch.manual_seed(0)
batch_size = 256
num_workers = 0
# Normalize((0.5,), (0.5,))
# cifar10_normalization()
normalization = cifar10_normalization()
# normalization = Normalize((0.5,), (0.5,))

# mnist_train = datasets.MNIST('../SwAV',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
# mnist_test = datasets.MNIST('../SwAV',train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),normalization]))

# 62 classes
# emnist_train = datasets.EMNIST('../SwAV',split='byclass',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]), download=True)
# emnist_test = datasets.EMNIST('../SwAV',split='byclass',train=False,transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]), download=True)

# cifar10_train = datasets.CIFAR10('../SwAV',transform=Compose([ToTensor(), normalization]))
# cifar10_test = datasets.CIFAR10('../SwAV',train=False, transform=Compose([ToTensor(), normalization]))

# cifar100_train = datasets.CIFAR100('../SwAV',transform=Compose([ToTensor(), normalization]))
# cifar100_test = datasets.CIFAR100('../SwAV',train=False, transform=Compose([ToTensor(), normalization]))

# stl10_train = datasets.STL10('C:/Dataset', split='train', transform=Compose([ToTensor(),normalization]))
# stl10_test = datasets.STL10('C:/Dataset', split='test', transform=Compose([ToTensor(),normalization]))
# stl10_unlabel = DataLoader(datasets.STL10('C:/Dataset', split='unlabeled', transform=Compose([SwAVTrainDataTransform(size_crops=[64, 32], nmb_crops=[2,4])])),batch_size=batch_size, shuffle=True)

# voc07_train = datasets.VOCDetection('./', year='2007', image_set='train',transform=Compose([Resize(size=(500,500)), ToTensor(), normalization]))
# voc07_test = datasets.VOCDetection('./', year='2007', image_set='val',transform=Compose([Resize(size=(500,500)),ToTensor(), normalization]))

svhn_train = datasets.SVHN('../SwAV', split='train',transform=Compose([ToTensor(), normalization]))
svhn_test = datasets.SVHN('../SwAV', split='test',transform=Compose([ToTensor(), normalization]))

loader_train = DataLoader(svhn_train, batch_size=batch_size, shuffle=True,num_workers=num_workers)#,persistent_workers=True)
loader_val = DataLoader(svhn_test,batch_size=batch_size,num_workers=num_workers)#,persistent_workers=True)

#%% load model
# encoder = SwAV.load_from_checkpoint('../SwAV/ep500_res50_proto3000_cifar10_bs256.ckpt')
encoder = SimCLR.load_from_checkpoint('ep500_SimCLRFusion_res50_cifar10_bs256.ckpt')
# encoder = SimCLR.load_from_checkpoint('ep100_SimCLRFusion_res18_cifar10_bs1024.ckpt')
out_dim = encoder(torch.empty(1,3,64,64, device=encoder.device)).shape[-1]
if hasattr(loader_train.dataset, 'classes'):
    classes = len(loader_train.dataset.classes)
elif hasattr(loader_train.dataset, 'labels'):
    classes = len(set(loader_train.dataset.labels))
else:
    raise AttributeError("number of classes unknown")
# classes = 10
model = LinearNN(encoder, out_dim, classes, frozen=True)
# model = LinearNN(encoder, 2048, 10, frozen=True)
model.eval()
#%% train linear
# train from scratch
# model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=False)
# model.fc = torch.nn.Linear(512, 20)
epochs = 100
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device('cuda:0')
model = model.to(device)

for epoch in range(epochs):
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
        # lr_scheduler.step()
        correct += (output.argmax(1) == label).type(torch.float).sum().item()
        loss_avg = _loss_sum/(batch_idx+1)
        print('\repoch: %d, loss: %6.3f, acc: %.3f'%((epoch +1), loss_avg, correct/size*100), end='')
        batch_idx += 1
    acc = correct/size*100
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
            print('\repoch: %d, loss: %6.3f, acc: %.3f, loss_val: %6.3f, acc_val: %.3f'%((epoch +1),loss_avg, acc, loss_val_avg/(batch_idx+1), correct/size*100), end='')
            batch_idx += 1
    print()
#%%validate data (KNN&SVM)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import classification_report
knn = KNN(n_neighbors=20)
# svc = SVC(kernel='linear')
feats_train = []
labels_train = []
feats_val = []
labels_val = []
torch.manual_seed(0)
model2 = SwAV.load_from_checkpoint('../SwAV/ep500_res50_proto3000_cifar10_bs256.ckpt')
# model2 = SimCLR.load_from_checkpoint('ep500_SimCLRFusion_res50_cifar10_bs256.ckpt')
device = torch.device('cuda:0')
model2 = model2.to(device)
model2.freeze()
for batch, label in loader_train:
    batch = batch.to(device)
    labels_train.append(label.cpu().numpy())
    
    feat = model2(batch).cpu().numpy()
    # feat = model2(batch)[0].cpu().numpy()
    feats_train.append(feat)

for batch, label in loader_val:
    batch = batch.to(device)
    labels_val.append(label.cpu().numpy())
    
    feat = model2(batch).cpu().numpy()
    # feat = model2(batch)[0].cpu().numpy()
    feats_val.append(feat)
x_train = np.vstack(feats_train)
y_train = np.hstack(labels_train)
x_val = np.vstack(feats_val)
y_val = np.hstack(labels_val)

knn.fit(x_train, y_train)
# pred = knn.predict(x_val)
pred_prob = knn.predict_proba(x_val)
pred = pred_prob.argmax(-1)
print(classification_report(y_val, pred, digits=5))
print('Top-1: %6.3f'%(topk(y_val, pred_prob, k=1)*100))
print('Top-5: %6.3f'%(topk(y_val, pred_prob, k=5)*100))