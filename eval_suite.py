# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 01:55:15 2021

@author: Colin
"""

from eval_cli import main
import argparse
args = argparse.Namespace()

args.method="svm"

args.model="swav"
args.model_path="../SwAV/ep500_res50_proto3000_cifar10_bs256.ckpt"

# args.model="simclr"
# args.model_path="ep500_SimCLRFusion_res50_cifar10_bs256.ckpt"

args.data_folder="../SwAV"
# args.data_folder="C:Dataset"
args.dataset="cifar100"

args.cuda=0
args.bs=256
args.workers=0
args.k=20
args.c=1
print()
print('Model:', args.model)
print('Model ckpt:', args.model_path)
print('Dataset:', args.dataset)
print('Method:', args.method)
print()
main(args)
