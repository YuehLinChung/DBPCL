import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.callbacks.progress import ProgressBarBase
import pytorch_lightning as pl

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
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # self.lightning_module, outputs, batch, batch_idx, dataloader_idx
        # super().on_train_batch_end(trainer, pl_module, outputs)  # don't forget this :)
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.train_loss += outputs['loss'].detach().cpu().item()
        self.train_avg_loss = self.train_loss / (batch_idx + 1)
        self.train_logger.set_postfix({'loss': '%.4f'%self.train_avg_loss})
        self.train_logger.update(1)
    
    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.val_loss = 0
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.val_loss += outputs.detach().item()
        if self.train_logger:
            self.train_logger.set_postfix({'loss':'%.4f'%self.train_avg_loss, 'val_loss':self.val_loss/(batch_idx+1)})
            # self.train_logger.set_postfix_str(self.train_logger.postfix + ', val_loss=%.2f'%(self.val_loss/(batch_idx+1)))
        
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.train_logger:
            self.train_logger.close()
        self.train_logger = tqdm(total=trainer.num_training_batches, ascii=True, desc='epoch %3d'%(self.trainer.current_epoch+1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        # self.epoch_idx += 1
        self.train_logger.reset()
        # self.train_logger.set_description('training epoch %2d'%self.epoch_idx)
        self.train_loss = 0
    def close_logger(self):
        if self.train_logger:
            self.train_logger.close()
            self.train_logger = None

class CheckpointOnEpochs(pl.Callback):
    def __init__(self, epochs: list, path_fmt,):
        self.epochs = epochs
        self.path_fmt = path_fmt

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch + 1
        if epoch in self.epochs:
            ckpt_path = self.path_fmt.format(epoch=epoch)
            trainer.save_checkpoint(ckpt_path)

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
