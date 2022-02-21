# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 01:55:15 2021

@author: Colin
"""

from eval_cli import main
import argparse
import pandas as pd

args = argparse.Namespace()

# args.model="swav"
# args.model_path="../SwAV/ep500_res50_proto3000_cifar10_bs256.ckpt"

args.model="simclr"
args.model_path="./models/ep100_DBCLR03_res50_stl10_bs256_aggr.ckpt"

# args.data_folder="../SwAV"
args.data_folder="C:\Dataset"

methods = ["knn", "lr"]
# datasets = ["mnist", "cifar10", "cifar100", "stl10"]
datasets = ["tiny-imagenet"]
# methods = ["ann", ]
# datasets = ["cifar100"]

# =============================================================================
args.cuda = 0
args.bs = 256
args.workers = 0
args.k = 20
args.c = 1

print()
print('Model:', args.model)
print('Model ckpt:', args.model_path)
print()

metrics = [m+'_top%d'%k for m in methods for k in [1, 5]]
summary = pd.DataFrame(index=datasets, columns=metrics)
print('',*metrics, sep='\t')

for dataset in datasets:
    args.dataset = dataset
    print(args.dataset, end='\t', flush=True)
    for method in methods:
        args.method = method
        result = main(args)
        print('%.3f'%result['top1'], '%.3f'%result['top5'], sep='\t', end='\t', flush=True)
        summary.loc[dataset, method+"_top1"] = '%.3f'%result['top1']
        summary.loc[dataset, method+"_top5"] = '%.3f'%result['top5']
        summary.to_csv('eval_result.tsv', sep='\t')
    print()