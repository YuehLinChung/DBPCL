# Density-Based Prototypical Contrastive Learning on Visual Representations
<img src="./figures/DBPCL_framework.png">

<!-- A Self-Supervised Learning method on Representation Learning. -->
## Requirements
- PyTorch
- Pytorch-Lightning
- scikit-learn

## Create Environment
- ```conda env create -n <env_name> environment.yml```
- ```conda activate <env_name>```

## Self-supervised Training
Our code only supports single-gpu and single-machine training currently.

To perform self-supervised training of a ResNet-32 model on CIFAR-100, run:
```bash!
python main.py --epochs 200 --dataset cifar100 -a resnet32 --mlp-dim 256 --out-dim 128 -b 64 -t 0.1 --con 0.1 --eps 0.3 0.5 --exp-dir logs --exp-name DBPCL_cifar100_res32_bs64_eps35_round1 --seed 1 --warmup-epoch 0 [dataset folder]
```


## Evaluation
run ```python eval_cli.py --help``` for details.


### Linear Evaluation
```bash!
python eval_cli.py --method lr --bs 32 --cuda 0 --j 0 --model-path [model checkpoint file] --data-folder [dataset folder] --dataset [dataset]
```

### Finetune
```bash!
python eval_cli.py --method ann --bs 32 --cuda 0 --j 4 --model-path [model checkpoint file] --data-folder [dataset folder] --dataset [dataset] --finetune
```

## Download Pretrained Models
| pretrain dataset | epochs | backbone| link |
| :--------------: | :----: | :-----: | :--: |
| CIFAR-10         | 200    | ResNet-32 | [download](https://github.com/YuehLinChung/DBPCL/files/9485109/ep200_DBCLR_cifar10_res32_bs64_eps35_round1.zip) |
| CIFAR-100        | 200    | ResNet-32 | [download](https://github.com/YuehLinChung/DBPCL/files/9485125/ep200_DBCLR_cifar100_res32_bs64_eps35_round1.zip)|
| STL-10           | 200    | ResNet-32 | [download](https://github.com/YuehLinChung/DBPCL/files/9485135/ep200_DBCLR_stl10_res32_bs64_eps35_round1.zip)   |
| Tiny-ImageNet    | 200    | ResNet-32 | [download](https://github.com/YuehLinChung/DBPCL/files/9485136/ep200_DBCLR_tiny_res32_bs64_eps35_round1.zip)    |
