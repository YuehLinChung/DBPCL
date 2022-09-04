# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:45:20 2021

@author: Colin
"""
from src.model import SimCLR, LinearNN, DBPCL
from src.utils import sparse2coarse

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

import torch
import random
import numpy as np
import argparse
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score as topk
from sklearn.model_selection import StratifiedShuffleSplit

torch.backends.cudnn.determinstic = True

def main(args):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    
    batch_size = args.bs
    num_workers = args.workers
    
    normalization = Normalize((0.5,), (0.5,))
        
    if args.dataset == "mnist":
        dataset_train = datasets.MNIST(args.data_folder,transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
        dataset_test = datasets.MNIST(args.data_folder,train=False, transform=Compose([ToTensor(),lambda x:x.expand(3,28,28),normalization]))
        classes = 10
    elif args.dataset == "emnist":
        dataset_train = datasets.EMNIST(args.data_folder,split='byclass',transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
        dataset_test = datasets.EMNIST(args.data_folder,split='byclass',train=False,transform=Compose([ToTensor(),lambda x:x.expand(3,28,28), normalization]))
        classes = 62
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(args.data_folder,transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.CIFAR10(args.data_folder,train=False, transform=Compose([ToTensor(), normalization]))
        classes = 10
    elif args.dataset == "cifar20":
        dataset_train = datasets.CIFAR100(args.data_folder,transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.CIFAR100(args.data_folder,train=False, transform=Compose([ToTensor(), normalization]))
        dataset_train.targets = sparse2coarse(dataset_train.targets).tolist()
        dataset_test.targets = sparse2coarse(dataset_test.targets).tolist()
        classes = 20
    elif args.dataset == "cifar100":
        dataset_train = datasets.CIFAR100(args.data_folder,transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.CIFAR100(args.data_folder,train=False, transform=Compose([ToTensor(), normalization]))
        classes = 100
    elif args.dataset == "stl10":
        dataset_train = datasets.STL10(args.data_folder, split='train', transform=Compose([ToTensor(),normalization]))
        dataset_test = datasets.STL10(args.data_folder, split='test', transform=Compose([ToTensor(),normalization]))
        classes = 10
    elif args.dataset == "tiny-imagenet":
        dataset_train = datasets.ImageFolder(args.data_folder + '/tiny-imagenet-200/train/',transform=Compose([ToTensor(),normalization]))
        dataset_test = datasets.ImageFolder(args.data_folder + '/tiny-imagenet-200/val/',transform=Compose([ToTensor(),normalization]))
        classes = 200
    elif args.dataset == "svhn":
        dataset_train = datasets.SVHN(args.data_folder, split='train',transform=Compose([ToTensor(), normalization]))
        dataset_test = datasets.SVHN(args.data_folder, split='test',transform=Compose([ToTensor(), normalization]))
        classes = 10
    else:
        raise RuntimeError('unsupport dataset')
    if args.fraction != 1 or isinstance(args.fraction, int):
        if hasattr(dataset_train, 'targets'):
            targets = dataset_train.targets
        elif hasattr(dataset_train, 'labels'):
            targets = dataset_train.labels
        else:
            raise AttributeError("targets unknown")
        fraction = args.fraction
        if isinstance(args.fraction, int):
            fraction *= classes
        sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=0)
        indices = next(sss.split(np.zeros(len(targets)), targets))[0]
        srs = torch.utils.data.SubsetRandomSampler(indices)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False,num_workers=num_workers, sampler=srs)
    else:    
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=num_workers)#,persistent_workers=True)
    loader_val = DataLoader(dataset_test,batch_size=batch_size,num_workers=num_workers)#,persistent_workers=True)

    #%% load model
    encoder = DBPCL.load_from_checkpoint(args.model_path)
    
    if args.cuda is not None:
        device = torch.device('cuda:%d'%args.cuda)
    else:
        device = torch.device('cpu')
    encoder = encoder.to(device)
    encoder.eval()

    #%% train linear
    if args.method == 'ann':
        out_dim = encoder(torch.empty(1,3,64,64, device=encoder.device)).shape[-1]
        model = LinearNN(encoder, out_dim, classes, frozen=not args.finetune)
        epochs = 100
        lr = 0.001
        if args.finetune:
            optimizer = torch.optim.SGD([{"params":model.encoder.parameters(), "lr":0.0001},
                                         {"params":model.linear.parameters(), "lr":0.001}],momentum=0.9)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
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
            model.train()
            for batch, label in loader_train:
                batch = batch.to(device)
                label = label.to(device)
                size += batch.shape[0]

                output = model(batch)
                loss = loss_fn(output, label)
                _loss_sum += loss.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct += (output.argmax(1) == label).type(torch.float).sum().item()
                loss_avg = _loss_sum/(batch_idx+1)
                print('\repoch: %d, loss: %6.3f, acc: %.3f'%((epoch +1), loss_avg, correct/size*100), end='')
                batch_idx += 1
            acc = correct/size*100
            # validation
            batch_idx = 0
            correct = 0
            size = 0
            model.eval()
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
        for param in encoder.parameters():
            param.requires_grad = False
        
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

            feat = encoder(batch).cpu().numpy()
            
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
        top1 = topk(y_val, pred_prob, k=1)*100
        top5 = topk(y_val, pred_prob, k=5)*100
        print('Top-1: %6.3f'%(top1))
        print('Top-5: %6.3f'%(top5))
    return {"top1":top1,"top5":top5}

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["knn", "ann", "svm", "lr"], default="lr", type=str, help="choose evaluation method from one of: {K Nearest Neighbor, Artificial Neural Network, Support Vector Machine, Logistic Regression}")
    parser.add_argument("--model-path", type=str, required=True, help="path to your model")
    parser.add_argument("--data-folder", type=str, required=True, help="path to your dataset root folder")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "cifar20", "mnist", "emnist", "stl10", "svhn"], type=str, required=True, help="dataset you want to evaluate on")
    parser.add_argument("--finetune", action="store_true", help="whether to finetune trained backbone")
    parser.add_argument("--cuda", type=int, default=None, help="cuda device to be used, use cpu as default")
    parser.add_argument("--bs", type=int, default=32, help="batch size used in evaluation")
    parser.add_argument("-j", "--workers", type=int, default=0, help="number of workers used in dataloader")
    parser.add_argument("--k", type=int, default=20, help="value of k used in KNN")
    parser.add_argument("--c", type=int, default=1, help="value of C used in SVM")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of training data for fine-tuning")
    args = parser.parse_args()
    main(args)
