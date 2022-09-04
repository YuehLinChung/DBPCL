import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .ResNet import resnet18, resnet34, resnet50
from .ResNetCifar import resnet8, resnet20, resnet32, resnet56, resnet110, Conv4
from torch import nn, optim
from sklearn.cluster import DBSCAN

class LinearNN(torch.nn.Module):
    def __init__(self, encoder, hidden_size, out_size, frozen=True):
        super().__init__()
        self.encoder = encoder

        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
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
            proj_hidden_dim=256,
            proj_out_dim=128,
            lr=0.01,
            temperature=0.1,
            weight_decay=1e-4,
            max_epochs=200,
            arch: str = "resnet32",
            first_conv = False,
            maxpool1 = False):
        super().__init__()
        self.arch = arch
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_out_dim = proj_out_dim
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = self.init_model()

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.projection_head = nn.Sequential(
            nn.Linear(self.convnet.inplanes, self.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.proj_out_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*self.max_epochs),int(0.8*self.max_epochs)], gamma=0.1)

        return [optimizer], [lr_scheduler]

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18(first_conv=self.first_conv, maxpool1=self.maxpool1)
        elif self.arch == "resnet34":
            backbone = resnet34(first_conv=self.first_conv, maxpool1=self.maxpool1)
        elif self.arch == "resnet50":
            backbone = resnet50(first_conv=self.first_conv, maxpool1=self.maxpool1)
        elif self.arch == "resnet8":
            backbone = resnet8()
        elif self.arch == "resnet20":
            backbone = resnet20()
        elif self.arch == "resnet32":
            backbone = resnet32()
        elif self.arch == "resnet56":
            backbone = resnet56()
        elif self.arch == "resnet110":
            backbone = resnet110()
        elif self.arch == "conv4":
            backbone = Conv4()
        return backbone

    def forward(self, x, proj=False):
        feats = self.convnet(x)
        if proj:
            proj = self.projection_head(feats)
            return feats, proj
        else:
            return feats

    def nt_xent_loss(self, batch, mode="train"):
        loss = 0
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        bs = imgs.size(0)

        # Encode all images
        feats = self.convnet(imgs)
        proj = self.projection_head(feats)

        # Calculate cosine similarity
        proj_normed = F.normalize(proj, p=2)
        cos_sim = torch.mm(proj_normed, proj_normed.t())

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # NT-Xent loss
        neg_pairs = cos_sim.masked_select(~(pos_mask | self_mask)).reshape(cos_sim.shape[0], cos_sim.shape[1]-2)
        non_diag = torch.cat((cos_sim[pos_mask].unsqueeze(1), neg_pairs), dim=1)
        prob = F.softmax(non_diag / self.hparams.temperature, dim=1)
        entropy = -prob[:, 0].log().mean()
        loss += entropy
        self.log("hp/infoNCE", entropy)
        top1_acc = (non_diag.argmax(axis=1)==0).float().mean()
        self.log("inst_acc_top1", top1_acc)#, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.nt_xent_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.nt_xent_loss(batch, mode="val")

    def training_epoch_end(self, outputs):
        epoch_loss = (sum([out['loss'] for out in outputs])/len(outputs)).item()
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)

class DBPCL(SimCLR):
    def __init__(
            self,
            proj_hidden_dim=256,
            proj_out_dim=128,
            lr=0.01,
            temperature=0.1,
            weight_decay=1e-4,
            max_epochs=200,
            arch: str = "resnet32",
            first_conv = False,
            maxpool1 = False,
            concentration: float = 0.1,
            con_estimation: bool = True,
            warmup_epoch: int = 0,
            eps:list = [0.3, 0.5],
            use_mlp: bool = True):
        self.concentration = concentration
        self.con_estimation = con_estimation
        self.warmup_epoch = warmup_epoch
        self.eps = eps
        self.use_mlp = use_mlp
        super().__init__(
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
            lr=lr,
            temperature=temperature,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            arch = arch,
            first_conv = first_conv,
            maxpool1 = maxpool1)

        if not use_mlp:
            # hack: brute-force replacement
            self.projection_head = nn.Linear(self.convnet.inplanes, proj_out_dim)

    def protoXent_loss(self, batch, mode="train"):
        loss = 0
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        bs = imgs.size(0)

        # Encode all images
        feats = self.convnet(imgs)
        proj = self.projection_head(feats)
        # Calculate cosine similarity
        proj_normed = F.normalize(proj, p=2)

        cos_sim = torch.mm(proj_normed, proj_normed.t())
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # NT-Xent loss
        neg_pairs = cos_sim.masked_select(~(pos_mask | self_mask)).reshape(cos_sim.shape[0], cos_sim.shape[1]-2)
        non_diag = torch.cat((cos_sim[pos_mask].unsqueeze(1), neg_pairs), dim=1)
        prob = F.softmax(non_diag / self.hparams.temperature, dim=1)

        entropy = -prob[:, 0].log().mean()
        loss += entropy
        self.log("hp/infoNCE", entropy)
        top1_acc = (non_diag.argmax(axis=1)==0).float().mean()
        self.log("inst_acc_top1", top1_acc)#, on_step=False, on_epoch=True)

        # Proto loss
        if self.current_epoch >= self.warmup_epoch:
            proto_loss = 0
            num_cluster = 0
            eps_list = self.eps
            for eps in eps_list:
                db = DBSCAN(eps=eps, min_samples=1, metric='precomputed', n_jobs=0)
                assignments = db.fit_predict(1-cos_sim.detach().cpu().numpy().clip(-1,1))

                self.log("clusters/eps: %.2f"%eps, float(assignments.max()+1), on_step=True)
                num_cluster += assignments.max() + 1
                _proto_loss = 0

                centroids = torch.LongTensor(assignments)
                centroids = centroids.view(centroids.size(0), 1).expand(-1, proj_normed.size(1))

                unique_labels, labels_count = centroids.unique(dim=0, return_counts=True)
                res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, centroids, proj_normed.cpu())
                res = res / labels_count.float().unsqueeze(1)
                res = F.normalize(res, p=2)
                centroids = res.type_as(proj_normed)

                if self.con_estimation and not (labels_count!=1).sum().item()==0:
                    concentrations = torch.zeros(centroids.shape[0]).scatter_add_(0,torch.LongTensor(assignments),1 - (proj_normed.detach().cpu()*centroids[assignments].detach().cpu()).sum(1).clip(-1,1))
                    concentrations /= eps * labels_count * (labels_count+10).log()

                    dmin = concentrations[labels_count!=1].min()
                    concentrations.clip_(max(dmin,1e-3))
                    concentrations /= (concentrations.mean() / self.concentration)
                    concentrations = concentrations.type_as(proj_normed)

                    concentrations = concentrations[assignments]
                    concentrations = concentrations.unsqueeze(1)
                else:
                    concentrations = self.concentration

                _proto_loss -= F.softmax(torch.mm(proj_normed,centroids.t()) / concentrations, dim=1)[range(proj_normed.shape[0]), assignments].log().mean()
                _proto_loss /= len(eps_list)
                proto_loss += _proto_loss

            loss += proto_loss

            self.log("hp/proto_loss", proto_loss, on_step=True)
            self.log("hp/clusters", float(num_cluster/len(eps_list)), on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.protoXent_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.protoXent_loss(batch, mode="val")


