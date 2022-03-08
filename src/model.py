import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .ResNet import resnet18, resnet50
from torch import nn, optim
from sklearn.cluster import DBSCAN
import numpy as np

class LinearNN(torch.nn.Module):
    def __init__(self, encoder, hidden_size, out_size, frozen=True):
        super().__init__()
        self.encoder = encoder
        
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.freeze()
        self.linear = torch.nn.Linear(hidden_size, out_size, device=self.encoder.device)
        self.frozen = frozen
    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                _x = self.encoder(x)
        else:
            _x = self.encoder(x)
        return self.linear(_x)

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
        return backbone()
    
    def forward(self, x, proj=False):
        feats = self.convnet(x)
        if proj:
            proj = self.projection_head(feats)
            return feats, proj
        else:
            return feats
        
    def info_nce_loss(self, batch, mode="train"):
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

class DBCLR(SimCLR):
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
            fusion_on: str = 'proj',
            concentration: float = 0.1,
            warmup_epoch: int = 20,
            eps:float = 0.3):
        self.concentration = concentration
        self.warmup_epoch = warmup_epoch
        self.eps = eps
        super().__init__(
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        lr=lr, 
        temperature=temperature, 
        weight_decay=weight_decay, 
        max_epochs=max_epochs, 
        arch = arch,
        fusion = fusion,
        fusion_on = fusion_on)
        
    def info_nce_loss(self, batch, mode="train"):
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
        if self.current_epoch >= self.warmup_epoch:
            inter_cluster_loss = 0
            cross_cluster_loss = 0
            for eps in [0.1,0.3,0.5]:
                db = DBSCAN(eps=eps, min_samples=1, metric='cosine', n_jobs=None)
                assignments = db.fit_predict(proj_normed.detach().cpu().numpy())
                clusters = assignments.max()+1
                # concentrations = (np.bincount(assignments)/(np.bincount(assignments)).mean())*self.temperature
                _inter_cluster_loss = 0
                _inter_clusters = 0
                centroids = []
                for group_index in np.unique(assignments):
                    group = proj_normed[assignments==group_index]
                    centroids.append(F.normalize(group.mean(dim=0, keepdim=True)))
                for group_index in np.unique(assignments):
                    group = proj_normed[assignments==group_index]
                    _inter_cluster_loss -= F.softmax(torch.mm(group, torch.vstack(centroids).t()) / self.concentration, dim=1)[:,group_index].log().mean()
                    _inter_clusters += 1
                    # if group.shape[0] == 1:
                    #     continue
                    # # inter_cluster_loss -= F.softmax(torch.mm(group, group.t()) / self.concentration, dim=1).diag().log().mean()
                    # _inter_cluster_loss -= F.softmax(torch.mm(group, torch.vstack(centroids).t()) / self.concentration, dim=1)[:,-1].log().mean()
                    # _inter_clusters += 1
                
                if _inter_clusters != 0:
                    _inter_cluster_loss /= _inter_clusters
                _inter_cluster_loss /= 3
                inter_cluster_loss += _inter_cluster_loss
                
                centroids = torch.vstack(centroids)
                _cross_cluster_loss = -F.softmax(torch.mm(centroids, centroids.t()) / self.concentration, dim=1).diag().log().mean()
                _cross_cluster_loss /= clusters
            
                _cross_cluster_loss /= 3
                cross_cluster_loss += _cross_cluster_loss
                
            loss += inter_cluster_loss
            loss += cross_cluster_loss
            self.log("hp/inter_cluster_loss", inter_cluster_loss, on_step=True)
            self.log("hp/cross_cluster_loss", cross_cluster_loss, on_step=True)
            self.log("hp/clusters", float(clusters), on_step=True)
        return loss

class MutiClustering(pl.LightningModule):
    def __init__(
            self, 
            proj_hidden_dim=2048,
            proj_out_dim=128,
            lr=0.01, 
            temperature=0.1, 
            weight_decay=1e-4, 
            max_epochs=500, 
            arch: str = "resnet50",
            nmb_cluster: list = [100, 10]):
        super().__init__()
        self.arch = arch
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_out_dim = proj_out_dim
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.nmb_cluster = nmb_cluster
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = self.init_model()
        # The MLP for g(.) consists of Linear->ReLU->Linear
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
        return backbone()
    def build_centroids(self):
        self.centroids = []
        for nmb_cluster in self.nmb_cluster:
            self.centroids.append(torch.nn.init.orthogonal(torch.rand(nmb_cluster, self.proj_out_dim)))
    def forward(self, x):
        feats = self.convnet(x)
        proj = self.projection_head(feats)
        
        return feats, proj
        
    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        x_a, x_b = imgs

        # Encode all images
        embedded_a = self.convnet(x_a)
        embedded_b = self.convnet(x_b)
        
        proj_a = self.projection_head(embedded_a)
        proj_b = self.projection_head(embedded_b)
        
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
        cos_sim.masked_fill_(self_mask, torch.finfo(cos_sim.dtype).min)
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

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

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")
    
    def training_epoch_end(self, outputs):
        epoch_loss = (sum([out['loss'] for out in outputs])/len(outputs)).item()
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)
        
class Supervised(pl.LightningModule):
    def __init__(
            self, 
            lr=0.01, 
            weight_decay=1e-4, 
            max_epochs=500, 
            arch: str = "resnet50",
            out_dim:int = 10):
        super().__init__()
        self.arch = arch
        self.lr = lr
        self.weight_decay = weight_decay
        self.out_dim = out_dim
        self.save_hyperparameters()
        # Base model f(.)
        self.convnet = self.init_model()
        self.fc = torch.nn.Linear(self.convnet.inplanes, self.out_dim)
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]
    
    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50
        return backbone()
    
    def forward(self, x):
        feats = self.convnet(x)
        return feats
        
    def cross_entropy_loss(self, batch, mode="train"):
        imgs, targets = batch
        imgs = torch.cat(imgs, dim=0)
        bs = imgs.size(0)
        # Encode all images
        feats = self.convnet(imgs)
        out = self.fc(feats)
        loss = self.criterion(out, targets)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.cross_entropy_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.cross_entropy_loss(batch, mode="val")
    
    def training_epoch_end(self, outputs):
        epoch_loss = (sum([out['loss'] for out in outputs])/len(outputs)).item()
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)

#%%
# =============================================================================
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(assignments, bins=range(256))
# fig = plt.figure()
# idx = 11
# for i in range((assignments==idx).sum()):
#     ax = fig.add_subplot(3,4,i+1)
#     ax.imshow((imgs.cpu()[assignments==idx]*0.5+0.5)[i].permute(1,2,0))
#     ax.set_title(np.arange(256)[assignments==idx][i])
#     ax.axis('off')
# =============================================================================

