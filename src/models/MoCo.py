from typing import Union

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
from torch.nn import functional as F
from ..ResNet import resnet18, resnet50
from ..ResNetCifar import resnet20, resnet32, resnet56

from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.utils import _PL_GREATER_EQUAL_1_4, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

class Moco_v2(LightningModule):
    """PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_

    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.

    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:

        - `William Falcon <https://github.com/williamFalcon>`_

    Example::
        from pl_bolts.models.self_supervised import Moco_v2
        model = Moco_v2()
        trainer = Trainer()
        trainer.fit(model)

    """

    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = "resnet18",
        emb_dim: int = 256,
        proj_out_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        lr: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        use_mlp: bool = False,
        max_epochs: int = 1000,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            lr: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            batch_size: batch size
            use_mlp: add an mlp to the encoders
        """

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Linear(emb_dim, proj_out_dim)
            self.encoder_k.fc = nn.Linear(emb_dim, proj_out_dim)
            
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, emb_dim), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, emb_dim), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(proj_out_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation queue
        self.register_buffer("val_queue", torch.randn(proj_out_dim, num_negatives))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=0)

        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        """Override to add your own encoders."""
        if base_encoder == "resnet18":
            backbone = resnet18
        elif base_encoder == "resnet50":
            backbone = resnet50
        elif base_encoder == "resnet20":
            backbone = resnet20
        elif base_encoder == "resnet32":
            backbone = resnet32
        elif base_encoder == "resnet56":
            backbone = resnet56

        encoder_q = backbone()
        encoder_q.fc = nn.Linear(encoder_q.inplanes, self.hparams.proj_out_dim)
        encoder_k = backbone()
        encoder_k.fc = nn.Linear(encoder_k.inplanes, self.hparams.proj_out_dim)
        
        # template_model = getattr(torchvision.models, base_encoder)
        # encoder_q = template_model(num_classes=self.hparams.emb_dim)
        # encoder_k = template_model(num_classes=self.hparams.emb_dim)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        # gather keys before updating queue
        if self._use_ddp_or_ddp2(self.trainer):
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward(self, x):
        feats = self.encoder_q(x)
        return feats
        
    def shared_step(self, img_q, img_k, queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = self.encoder_q.fc(q)  # queries: NxC

        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            if self._use_ddp_or_ddp2(self.trainer):
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self._use_ddp_or_ddp2(self.trainer):
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k

    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self.shared_step(img_q=img_1, img_k=img_2, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), labels = batch

        output, target, keys = self.shared_step(img_q=img_1, img_k=img_2, queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     self.hparams.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.hparams.weight_decay,
        # )
        # return optimizer

    @staticmethod
    def _use_ddp_or_ddp2(trainer: Trainer) -> bool:
        # for backwards compatibility
        if _PL_GREATER_EQUAL_1_4:
            return trainer._accelerator_connector.use_ddp or trainer._accelerator_connector.use_ddp2
        return trainer.use_ddp or trainer.use_ddp2


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
