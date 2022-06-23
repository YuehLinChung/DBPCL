import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from ..ResNet import resnet18, resnet50
from ..ResNetCifar import resnet20, resnet32, resnet56
from torch import nn, optim
from sklearn.cluster import DBSCAN, KMeans
import numpy as np

class SimCLR(pl.LightningModule):
    def __init__(
            self, 
            proj_hidden_dim=2048,
            proj_out_dim=128,
            lr=0.01, 
            temperature=0.1, 
            weight_decay=1e-4, 
            max_epochs=500, 
            arch: str = "resnet50",
            fusion: bool = False,
            fusion_on: str = 'proj'):
        super().__init__()
        self.arch = arch
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_out_dim = proj_out_dim
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.fusion = fusion
        self.fusion_on = fusion_on
        assert self.fusion_on in ['proj', 'proj_norm', 'feat'], "The fusion must take place on either proj, proj_norm or feat"
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = self.init_model()
        # The MLP for g(.) consists of Linear->ReLU->Linear
# =============================================================================
#         self.mlp = nn.Sequential(
#             nn.Linear(self.convnet.num_out_filters, 4 * hidden_dim),  # Linear(ResNet output, 4*hidden_dim)
#             nn.ReLU(inplace=True),
#             nn.Linear(4 * hidden_dim, hidden_dim),
#         )
# =============================================================================
        self.projection_head = nn.Sequential(
            nn.Linear(self.convnet.inplanes, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.proj_out_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
    
    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50
        elif self.arch == "resnet20":
            backbone = resnet20
        elif self.arch == "resnet32":
            backbone = resnet32
        elif self.arch == "resnet56":
            backbone = resnet56
        return backbone()
    
    def forward(self, x, proj=False):
        feats = self.convnet(x)
        if proj:
            proj = self.projection_head(feats)
            return feats, proj
        else:
            return feats
        
    def info_nce_loss(self, batch, mode="train"):
        if self.trainer.current_epoch >= 100:
            self.trainer.should_stop = True
        loss = 0
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        bs = imgs.size(0)
        # Encode all images
        feats = self.convnet(imgs)
        proj = self.projection_head(feats)
        # Calculate cosine similarity
        # cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        proj_normed = F.normalize(proj, p=2)
        cos_sim = torch.mm(proj_normed, proj_normed.t())
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # cos_sim.masked_fill_(self_mask, torch.finfo(cos_sim.dtype).min) #-9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
# =============================================================================
#         cos_sim.masked_fill_(self_mask, torch.finfo(cos_sim.dtype).min)
#         nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
#         loss = nll.mean()
# =============================================================================
        
        neg_pairs = cos_sim.masked_select(~(pos_mask | self_mask)).reshape(cos_sim.shape[0], cos_sim.shape[1]-2)
        non_diag = torch.cat((cos_sim[pos_mask].unsqueeze(1), neg_pairs), dim=1)
        prob = F.softmax(non_diag, dim=1)
        # prob = F.softmax(non_diag / 0.1, dim=1)
        label = torch.zeros_like(prob)
        label[:,0] = 1
        # entropy = -(label * prob.log()).sum(1).mean()
        entropy = -prob[:, 0].log().mean()
        loss += entropy
        self.log("hp/infoNCE", entropy)
# =============================================================================
#         corr = torch.mm(embedding, embedding_d.t())
#         positive_pairs_mask = torch.eye(corr.shape[0], dtype=bool, device=self.device).roll(bs, dims=1)
#         diag = torch.eye(corr.shape[0], dtype=bool, device=self.device)
#         neg_pairs = corr.masked_select(~(positive_pairs_mask | diag)).reshape(corr.shape[0], corr.shape[1]-2)
# 
#         non_diag = torch.cat((corr[positive_pairs_mask].unsqueeze(1), neg_pairs), dim=1)
#         prob = self.softmax(non_diag / 0.1)
#         label = torch.zeros_like(prob)
#         label[:,0] = 1
#         entropy = -(label * prob.log()).sum(1).mean()
#         loss += entropy
# =============================================================================

        if self.fusion:
            # pick a random sample in the mini-batch
            choice = torch.randint(0,bs,(bs,))
            # draw a random number from [0, 1] as a ratio of the fusion
            ratio = torch.rand(bs,1, device=proj.device)
            if self.fusion_on == 'proj':
                fusion = (proj * ratio + proj[choice] * (1-ratio))
            elif self.fusion_on == 'proj_norm':
                fusion = (proj_normed * ratio + proj_normed[choice] * (1-ratio))
            elif self.fusion_on == 'feat':
                fusion = (feats * ratio + feats[choice] * (1-ratio))
                fusion = self.projection_head(fusion)
            
            target = torch.zeros(bs,bs, device=proj.device)
            target[self_mask] = ratio.squeeze()
            target[range(bs),choice] = (1-ratio.squeeze())
            corr = torch.mm(F.normalize(fusion), proj_normed.t())#proj_normed.detach()
            
# =============================================================================
#             target = torch.zeros(bs,2, device=proj.device)
#             target[:,0] = ratio.squeeze()
#             target[:,1] = (1-ratio.squeeze())
#             corr = torch.stack(((F.normalize(fusion) * proj_normed).sum(dim=1), (F.normalize(fusion) * proj_normed[choice]).sum(dim=1)),dim=1)
# =============================================================================
            # loss = 0
            corr = (corr / self.temperature).exp() / (corr / self.temperature).exp().sum(1,keepdim=True)
            fusion_loss = (target * corr.log()).sum(1).mean()
            loss -= fusion_loss
            self.log("hp/fusion_loss", -fusion_loss)
        
# =============================================================================
#         corr
# =============================================================================
# =============================================================================
#         proj_a = proj_normed[:bs//2]
#         proj_b = proj_normed[bs//2:]
#         corr_loss = 0.1*((torch.mm(proj_a, proj_a.t()) - torch.mm(proj_b, proj_b.t()))**2).sum()
#         loss += corr_loss
#         self.log("hp/corr_loss", corr_loss)
# =============================================================================
# =============================================================================
#         corr
# =============================================================================
        # Logging loss
        # self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        # comb_sim = torch.cat(
        #     [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, torch.finfo(cos_sim.dtype).min)],  # First position positive example
        #     dim=-1,
        # )
        # sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        # self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), on_step=False, on_epoch=True)
        # self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        # self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return loss

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="val")
    
    def training_epoch_end(self, outputs):
        epoch_loss = (sum([out['loss'] for out in outputs])/len(outputs)).item()
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)
