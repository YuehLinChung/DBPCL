# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""
from pl_bolts.models.self_supervised import SwAV
from src.model import SimCLR, LinearNN

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
# import sys
import torch
# import os
import numpy as np
import argparse
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score as topk
# from numba import cuda

def main(args):
    torch.manual_seed(0)
    batch_size = args.bs
    num_workers = args.workers

    if args.model == "swav":
        normalization = cifar10_normalization()
    elif args.model == "simclr":
        normalization = Normalize((0.5,), (0.5,))

    if args.dataset == "mnist":
        dataset_train = datasets.MNIST(args.data_folder,transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
        dataset_test = datasets.MNIST(args.data_folder,train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),normalization]))
    elif args.dataset == "emnist":
        dataset_train = datasets.EMNIST(args.data_folder,split='byclass',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
        dataset_test = datasets.EMNIST(args.data_folder,split='byclass',train=False,transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(args.data_folder,transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.CIFAR10(args.data_folder,train=False, transform=Compose([ToTensor(), normalization]))
    elif args.dataset == "cifar100":
        dataset_train = datasets.CIFAR100(args.data_folder,transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.CIFAR100(args.data_folder,train=False, transform=Compose([ToTensor(), normalization]))
    elif args.dataset == "stl10":
        dataset_train = datasets.STL10(args.data_folder, split='train', transform=Compose([ToTensor(),normalization]))
        dataset_test = datasets.STL10(args.data_folder, split='test', transform=Compose([ToTensor(),normalization]))
    elif args.dataset == "tiny-imagenet":
        dataset_train = datasets.ImageFolder(args.data_folder + '/tiny-imagenet-200/train/',transform=Compose([ToTensor(),normalization]))
        dataset_test = datasets.ImageFolder(args.data_folder + '/tiny-imagenet-200/val/',transform=Compose([ToTensor(),normalization]))

    # voc07_train = datasets.VOCDetection('./', year='2007', image_set='train',transform=Compose([Resize(size=(500,500)), ToTensor(), normalization]))
    # voc07_test = datasets.VOCDetection('./', year='2007', image_set='val',transform=Compose([Resize(size=(500,500)),ToTensor(), normalization]))
    elif args.dataset == "svhn":
        dataset_train = datasets.SVHN(args.data_folder, split='train',transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.SVHN(args.data_folder, split='test',transform=Compose([ToTensor(), normalization]))
    else:
        raise RuntimeError('unsupport dataset')
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=num_workers)#,persistent_workers=True)
    loader_val = DataLoader(dataset_test,batch_size=batch_size,num_workers=num_workers)#,persistent_workers=True)

    #%% load model
    if args.model == "swav":
        encoder = SwAV.load_from_checkpoint(args.model_path)
    elif args.model == "simclr":
        encoder = SimCLR.load_from_checkpoint(args.model_path)

    if args.cuda is not None:
        device = torch.device('cuda:%d'%args.cuda)
    else:
        device = torch.device('cpu')
    encoder = encoder.to(device)
    encoder.freeze()
    encoder.eval()

    #%% train linear
    if args.method == 'ann':
        out_dim = encoder(torch.empty(1,3,64,64, device=encoder.device)).shape[-1]
        if hasattr(dataset_train, 'classes'):
            classes = len(dataset_train.classes)
        elif hasattr(dataset_train, 'labels'):
            classes = len(set(dataset_train.labels))
        else:
            raise AttributeError("number of classes unknown")
        model = LinearNN(encoder, out_dim, classes, frozen=True)
        epochs = 100
        lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/50)
        loss_fn = torch.nn.CrossEntropyLoss()
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
        out_list = []
        label_list = []
        with torch.no_grad():
            for batch, label in loader_val:
                batch = batch.to(device)
                label = label.to(device)
                output = model(batch)
                output = torch.nn.functional.softmax(output, dim=1)
                out_list.append(output.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())
        outputs = np.vstack(out_list)
        y_true = np.hstack(label_list)

        top1 = topk(y_true, outputs, k=1)*100
        top5 = topk(y_true, outputs, k=5)*100
        # print('Top-1: %6.3f'%(top1))
        # print('Top-5: %6.3f'%(top5))

    elif args.method in ['knn', 'svm', 'lr']:
        if args.method == 'knn':
            model = KNN(n_neighbors=args.k, n_jobs=-1)
        elif args.method == 'svm':
            # model = SVC(kernel='linear', C=args.c, probability=True)
            model = LinearSVC(dual=False)
        elif args.method == 'lr':
            model = LogisticRegression(n_jobs=-1)
        feats_train = []
        labels_train = []
        feats_val = []
        labels_val = []
        torch.manual_seed(0)
        for batch, label in loader_train:
            batch = batch.to(device)
            labels_train.append(label.cpu().numpy())

            feat = encoder(batch).cpu().numpy()
            feats_train.append(feat)
            batch = batch.to('cpu')
            del batch

        for batch, label in loader_val:
            batch = batch.to(device)
            labels_val.append(label.detach().cpu().numpy())

            feat = encoder(batch).detach().cpu().numpy()
            feats_val.append(feat)

        # cuda.close()
        # print('close cuda')
        x_train = np.vstack(feats_train)
        y_train = np.hstack(labels_train)
        x_val = np.vstack(feats_val)
        y_val = np.hstack(labels_val)

        model.fit(x_train, y_train)
        if args.method in ['knn', 'lr']:
            pred_prob = model.predict_proba(x_val)
        elif args.method == 'svm':
            pred_prob = model.decision_function(x_val)
        pred = pred_prob.argmax(-1)
        # print(classification_report(y_val, pred, digits=5))
        top1 = topk(y_val, pred_prob, k=1)*100
        top5 = topk(y_val, pred_prob, k=5)*100
        # print('Top-1: %6.3f'%(top1))
        # print('Top-5: %6.3f'%(top5))
    return {"top1":top1,"top5":top5}

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["knn", "ann", "svm", "lr"], default="knn", type=str, help="choose evaluation method from one of: {K Nearest Neighbor, Artificial Neural Network, Support Vector Machine, Logistic Regression}")
    parser.add_argument("--model", choices=["swav", "simclr"], type=str, required=True, help="self-supervised pre-trained model to be used")
    parser.add_argument("--model-path", type=str, required=True, help="path to your model")
    parser.add_argument("--data-folder", type=str, required=True, help="path to your dataset root folder")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist", "emnist", "stl10", "svhn"], type=str, required=True, help="dataset you want to evaluate on")
    # parser.add_argument("--out-dim", type=int, required=True, help="output dimension of your model")
    parser.add_argument("--cuda", type=int, default=None, help="cuda device to be used, use cpu as default")
    parser.add_argument("--bs", type=int, default=256, help="batch size used in evaluation")
    parser.add_argument("--workers", type=int, default=0, help="number of workers used in dataloader")
    parser.add_argument("--k", type=int, default=20, help="value of k used in KNN")
    parser.add_argument("--c", type=int, default=1, help="value of C used in SVM")

    args = parser.parse_args()
    main(args)
