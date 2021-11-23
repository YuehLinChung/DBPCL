import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .ResNet import resnet18, resnet50
from torch import nn, optim

class SimCLR(pl.LightningModule):
    def __init__(
            self, 
            proj_hidden_dim=2048,
            proj_out_dim=128,
            lr=0.01, 
            temperature=0.1, 
            weight_decay=1e-4, 
            max_epochs=500, 
            arch: str = "resnet50"):
        super().__init__()
        self.arch = arch
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_out_dim = proj_out_dim
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        
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
    
    def forward(self, x):
        feats = self.convnet(x)
        proj = self.projection_head(feats)
        
        return feats, proj
        
    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

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
        