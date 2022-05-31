# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 01:55:15 2021

@author: Colin
"""

from eval_cli import main
import argparse
import pandas as pd

args = argparse.Namespace()

args.model="swav"
args.model_path="../SwAV/ep100_res50_proto300_cifar10_bs256.ckpt"

# args.model="simclr"
# args.model_path="./ep100_DBCLR01_eps123_res18_cifar10_bs256_aggr_warmup_proto_fixed2.ckpt"
args.fraction = 1.0

# args.data_folder="../SwAV"
args.data_folder="C:\Dataset"
methods = ["knn", "lr"]
# datasets = ["mnist", "cifar10", "cifar100", "stl10", "tiny-imagenet"]
mode = 'row'
datasets = ["cifar100"]
# methods = ["lr", ]

# =============================================================================
args.cuda = 0
args.bs = 128
args.workers = 0
args.k = 20
args.c = 1

print()
print('Model:', args.model)
print('Model ckpt:', args.model_path)
print('Fraction: %.2f'%args.fraction)
print()

if mode == 'column':
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
elif mode == 'row':
    metrics = [d+'_'+m+'_top%d'%k for m in methods for d in datasets for k in [1, 5]]
    summary = pd.Series(index=metrics, dtype=float)
    
    for method in methods:
        args.method = method
        for dataset in datasets:
            args.dataset = dataset
            print(method + '_' + dataset, end='\t', flush=True)
            result = main(args)
            summary[dataset+"_"+method+"_top1"] = '%.3f'%result['top1']
            summary[dataset+"_"+method+"_top5"] = '%.3f'%result['top5']
            summary.to_frame(args.model_path).transpose().to_csv('eval_result.tsv', sep='\t')
    print()
