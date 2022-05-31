import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .ResNet import resnet18, resnet34, resnet50
from .ResNetCifar import resnet20, resnet32, resnet56
from torch import nn, optim
from sklearn.cluster import DBSCAN, KMeans
import numpy as np

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
            proj_hidden_dim=2048,
            proj_out_dim=128,
            lr=0.01,
            temperature=0.1,
            weight_decay=1e-4,
            max_epochs=500,
            arch: str = "resnet50",
            fusion: bool = False,
            fusion_on: str = 'proj',
            first_conv = True,
            maxpool1 = True):
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
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
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
            nn.Linear(self.convnet.inplanes, self.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.proj_out_dim),
        )

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,160], gamma=0.1)
        # return optimizer
        return [optimizer], [lr_scheduler]

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18(first_conv=self.first_conv, maxpool1=self.maxpool1)
        elif self.arch == "resnet34":
            backbone = resnet34(first_conv=self.first_conv, maxpool1=self.maxpool1)
        elif self.arch == "resnet50":
            backbone = resnet50(first_conv=self.first_conv, maxpool1=self.maxpool1)
        elif self.arch == "resnet20":
            backbone = resnet20()
        elif self.arch == "resnet32":
            backbone = resnet32()
        elif self.arch == "resnet56":
            backbone = resnet56()
        return backbone

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
            lr=0.001,
            temperature=0.1,
            weight_decay=1e-4,
            max_epochs=500,
            arch: str = "resnet50",
            fusion: bool = False,
            fusion_on: str = 'proj',
            first_conv = True,
            maxpool1 = True,
            concentration: float = 0.1,
            con_estimation: bool = True,
            warmup_epoch: int = 20,
            eps:list = [0.1, 0.3, 0.5],
            use_mlp: bool = True):
            # num_views:int = 2
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
            fusion = fusion,
            fusion_on = fusion_on,
            first_conv = first_conv,
            maxpool1 = maxpool1)

        if not use_mlp:
            # hack: brute-force replacement
            self.projection_head = nn.Linear(self.convnet.inplanes, proj_out_dim)
        # for semi-supervised training
        # self.classification_head = torch.nn.Linear(self.convnet.inplanes, 100)
        # num_negatives = 65536
        # self.register_buffer("queue", torch.empty(0, proj_out_dim))

# =============================================================================
#     def _dequeue_and_enqueue(self, feats):
#         num_negatives = 4096
#         step = feats.shape[0]
#         if self.queue.shape[0] < num_negatives:
#             self.queue = torch.cat((feats, self.queue))
#         else:
#             self.queue[step:] = self.queue[:-step].clone()
#             self.queue[:step] = feats.clone()
# 
#         if self.queue.shape[0] > num_negatives:
#             self.queue = self.queue[:num_negatives]
#         return
# =============================================================================

# =============================================================================
#     def configure_optimizers(self):
#         # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
#         # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.lr / 50)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,160], gamma=0.1)
# 
#         # return optimizer
#         return [optimizer], [lr_scheduler]
# =============================================================================

    def info_nce_loss(self, batch, mode="train"):
        loss = 0
        # imgs, labels = batch
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        # labels = labels.repeat(2)
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
        # cos_sim.masked_fill_(self_mask, torch.finfo(cos_sim.dtype).min)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        # cos_sim = cos_sim / self.hparams.temperature

        neg_pairs = cos_sim.masked_select(~(pos_mask | self_mask)).reshape(cos_sim.shape[0], cos_sim.shape[1]-2)
        non_diag = torch.cat((cos_sim[pos_mask].unsqueeze(1), neg_pairs), dim=1)
        prob = F.softmax(non_diag / self.hparams.temperature, dim=1)
        # prob = F.softmax(non_diag / 0.1, dim=1)
        # label = torch.zeros_like(prob)
        # label[:,0] = 1
        # entropy = -(label * prob.log()).sum(1).mean()
        entropy = -prob[:, 0].log().mean()
        loss += entropy
        self.log("hp/infoNCE", entropy)

        # semi_fraction = 0.1
        # if semi_fraction:
        #     idx_picked = torch.randperm(bs)[:int(semi_fraction*bs)]
        #     labels_picks = labels[idx_picked]
        #     pred = self.classification_head(proj[idx_picked])
        #     semi_entropy = F.cross_entropy(pred, labels_picks)
        #     loss += semi_entropy
        #     self.log("hp/semi_entropy", semi_entropy)
            # torch.bincount(labels_picks)
            # available_labels = torch.arange(labels_picks.max()+1)[torch.bincount(labels_picks)>1]
            # cos_sim_picked = []
            # cos_sim_normed = F.softmax(cos_sim, dim=1)
            # import itertools
            # if len(available_labels)>0:
            #     for l in available_labels:
            #         same_class = idx_picked[labels_picks==l]
            #         for pair in list(itertools.permutations(same_class.numpy(),2)):
            #             cos_sim_picked.append(cos_sim_normed[pair[0], pair[1]])
            #     semi_entropy = -torch.hstack(cos_sim_picked).log().mean()
            #     loss += semi_entropy
            #     self.log("hp/semi_entropy", semi_entropy)

        if self.current_epoch >= self.warmup_epoch:
            inter_cluster_loss = 0
            cross_cluster_loss = 0
            num_cluster = 0
            eps_list = self.eps
            for eps in eps_list:
                db = DBSCAN(eps=eps, min_samples=1, metric='precomputed', n_jobs=-1)
                assignments = db.fit_predict(1-cos_sim.detach().cpu().numpy().clip(-1,1))

                # db = KMeans(n_clusters=eps, init='random', n_init=5, max_iter=100)
                # assignments = db.fit_predict(proj_normed.detach().cpu().numpy())
                self.log("clusters/eps: %.2f"%eps, assignments.max() + 1, on_step=True)
                num_cluster += assignments.max() + 1
                _inter_cluster_loss = 0
                _cross_cluster_loss = 0
                _inter_clusters = 0
# =============================================================================
#
# =============================================================================
                centroids = torch.LongTensor(assignments)
                centroids = centroids.view(centroids.size(0), 1).expand(-1, proj_normed.size(1))

                unique_labels, labels_count = centroids.unique(dim=0, return_counts=True)
                res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, centroids, proj_normed.cpu())
                res = res / labels_count.float().unsqueeze(1)
                res = F.normalize(res, p=2)
                centroids = res.type_as(proj_normed)
# =============================================================================
#
# =============================================================================
                
                if self.con_estimation and not (labels_count!=1).sum().item()==0:
                    ##### conRev
                    # concentrations = labels_count.float()
                    #####
                    # concentrations = labels_count / len(labels_count)
                    # concentrations = 1 / (labels_count.log()+10)
                    ##### conDisClip
                    concentrations = (torch.zeros(centroids.shape[0]).scatter_add_(0,torch.LongTensor(assignments),1 - (proj_normed.detach().cpu()*centroids[assignments].detach().cpu()).sum(1).clip(-1,1))/labels_count) / ((labels_count+10).log())
                    concentrations /= eps
                    dmin = concentrations[labels_count!=1].min()
                    concentrations.clip_(max(dmin,1e-3))
                    # concentrations[concentrations == 1e-3] = concentrations.mean()# concentrations.max()
                    # concentrations.clip_(np.percentile(concentrations,10), np.percentile(concentrations,90))
                    #####
                    ##### common
                    concentrations /= (concentrations.mean() / self.concentration)
                    concentrations = concentrations.type_as(proj_normed)
                    concentrations = concentrations[assignments]
                    concentrations = concentrations.unsqueeze(1)
                    # concentrations = concentrations.unsqueeze(0)
                else:
                    concentrations = self.concentration
                weight = bs/2/(assignments.max() + 1)
                _inter_cluster_loss -= weight*F.softmax(torch.mm(proj_normed,centroids.t()) / concentrations, dim=1)[range(proj_normed.shape[0]), assignments].log().mean()
                # _inter_cluster_lossv2 = ((1-torch.mm(proj_normed,centroids.t())[range(proj_normed.shape[0]), assignments])**2).mean()
                # centroids.mm(centroids.t()).sort(dim=1)[1][:,:int(0.9*centroids.shape[0])][assignments]
                _inter_cluster_loss /= len(eps_list)
                inter_cluster_loss += _inter_cluster_loss
                # _cross_cluster_loss = -F.softmax(torch.mm(centroids, centroids.t()) / self.concentration, dim=1).diag().log().mean()
                # _cross_cluster_loss /= len(eps_list)
                # cross_cluster_loss += _cross_cluster_loss
            loss += inter_cluster_loss
            # loss += cross_cluster_loss
            self.log("hp/inter_cluster_loss", inter_cluster_loss, on_step=True)
            # self.log("hp/cross_cluster_loss", cross_cluster_loss, on_step=True)
            self.log("hp/clusters", float(num_cluster/len(eps_list)), on_step=True)
        return loss

class Supervised(pl.LightningModule):
    def __init__(
            self,
            lr=0.01,
            weight_decay=1e-4,
            max_epochs=500,
            arch: str = "resnet50",
            out_dim:int = 10):
        super().__init__()
        self.max_epochs = max_epochs
        self.arch = arch
        self.lr = lr
        self.weight_decay = weight_decay
        self.out_dim = out_dim
        self.save_hyperparameters()
        # Base model f(.)
        self.convnet = self.init_model()
        self.fc = torch.nn.Linear(self.convnet.inplanes, self.out_dim)
        # self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.lr / 50)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,160], gamma=0.1)

        # return optimizer
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
        loss = F.cross_entropy(out, targets) # self.criterion(out, targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self.cross_entropy_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.cross_entropy_loss(batch, mode="val")

    def training_epoch_end(self, outputs):
        epoch_loss = (sum([out['loss'] for out in outputs])/len(outputs)).item()
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)

class FocalLoss(torch.nn.Module):
    """Sigmoid focal cross entropy loss.
    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
          gamma: exponent of the modulating factor (1 - p_t)^gamma.
          alpha: optional alpha weighting factor to balance positives vs negatives,
               with alpha in [0, 1] for class 1 and 1-alpha for class 0.
               In practice alpha may be set by inverse class frequency,
               so that for a low number of positives, its weight is high.
        """
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self.BCEWithLogits = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, prediction_tensor, target_tensor):
        """Computepute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets.
        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = self.BCEWithLogits(prediction_tensor, target_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) + #positives probs
              ((1 - target_tensor) * (1 - prediction_probabilities))) #negatives probs
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma) #the lowest the probability the highest the weight
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return torch.mean(focal_cross_entropy_loss)

class RelationNet(pl.LightningModule):
    def __init__(
            self,
            arch: str = "resnet50",
            aggregation="cat",
            lr = 0.01,
            weight_decay=1e-4,
            max_epochs=500):
        super().__init__()
        self.arch = arch
        self.aggregation=aggregation
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.save_hyperparameters()
        self.convnet = self.init_model()

        if(self.aggregation=="cat"): resizer=2
        elif(self.aggregation=="sum"): resizer=1
        elif(self.aggregation=="mean"): resizer=1
        elif(self.aggregation=="max"): resizer=1
        else: RuntimeError("[ERROR] aggregation type " + str(self.aggregation) +  " not supported, must be: cat, sum, mean.")
        import collections
        self.relation_module = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(self.convnet.inplanes*resizer, 256)),
          ("bn1",      nn.BatchNorm1d(256)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(256, 1)),
        ]))

        self.fl = FocalLoss(gamma=2.0, alpha=0.5) #Using reccommended value for gamma: 2.0
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
    def aggregate(self, features, tot_augmentations, type="cat"):
        """Aggregation function.
        Args:
          features: The features returned by the backbone, it is a tensor
            of shape [batch_size*K, feature_size].
            num_classes] representing the predicted logits for each class
          tot_augmentations: The total number of augmentations, corresponds
            to the parameter K in the paper.
        Returns:
          relation_pairs: a tensor with the aggregated pairs that can be
            given as input to the relation head.
          target: the values (zeros and ones) for each pair, that
            represents the target used to train the relation head.
          tot_positive: Counter for the total number of positives.
          tot_negative: Counter for the total number of negatives.
        """
        relation_pairs_list = list()
        target_list = list()
        size = int(features.shape[0] / tot_augmentations)
        tot_positive = 0.0
        tot_negative = 0.0
        shifts_counter=1
        for index_1 in range(0, size*tot_augmentations, size):
            for index_2 in range(index_1+size, size*tot_augmentations, size):
                if(type=="cat"):
                    positive_pair = torch.cat([features[index_1:index_1+size], features[index_2:index_2+size]], 1)
                    negative_pair = torch.cat([features[index_1:index_1+size],
                                               torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)], 1)
                elif(type=="sum"):
                    positive_pair = features[index_1:index_1+size] + features[index_2:index_2+size]
                    negative_pair = features[index_1:index_1+size] + torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)
                elif(type=="mean"):
                    positive_pair = (features[index_1:index_1+size] + features[index_2:index_2+size]) / 2.0
                    negative_pair = (features[index_1:index_1+size] + torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)) / 2.0
                elif(type=="max"):
                    positive_pair, _ = torch.max(torch.stack([features[index_1:index_1+size], features[index_2:index_2+size]], 2), 2)
                    negative_pair, _ = torch.max(torch.stack([features[index_1:index_1+size],
                                                              torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)], 2), 2)
                relation_pairs_list.append(positive_pair)
                relation_pairs_list.append(negative_pair)
                target_list.append(torch.ones(size, dtype=torch.float32))
                target_list.append(torch.zeros(size, dtype=torch.float32))
                tot_positive += size
                tot_negative += size
                shifts_counter+=1
                if(shifts_counter>=size): shifts_counter=1 # reset to avoid neutralizing the roll
        relation_pairs = torch.cat(relation_pairs_list, 0)
        target = torch.cat(target_list, 0)
        return relation_pairs, target, tot_positive, tot_negative
    def forward(self, x):
        feats = self.convnet(x)

        return feats
    def focalloss(self, batch, mode="train"):
        imgs, _ = batch
        tot_augmentations = len(imgs)
        batch_size = imgs[0].shape[0]

        train_x = torch.cat(imgs, 0)

        # forward pass in the backbone
        features = self.convnet(train_x)
        # aggregation over the representations returned by the backbone
        relation_pairs, train_y, tot_positive, tot_negative = self.aggregate(features, tot_augmentations, type=self.aggregation)
        train_y = train_y.type_as(train_x)
        tot_pairs = int(relation_pairs.shape[0])
        # forward of the pairs through the relation head
        predictions = self.relation_module(relation_pairs).squeeze()
        # estimate the focal loss (also standard BCE can be used here)
        loss = self.fl(predictions, train_y)
        # backward step and weights update
        return loss
    def training_step(self, batch, batch_idx):
        return self.focalloss(batch, mode="train")

    def training_epoch_end(self, outputs):
        epoch_loss = (sum([out['loss'] for out in outputs])/len(outputs)).item()
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)
#%%
# =============================================================================
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(assignments, bins=range(assignments.max()+1))
# fig = plt.figure()
# idx = 14
# for i in range((assignments==idx).sum()):
#     ax = fig.add_subplot(3,4,i+1)
#     ax.imshow((imgs.cpu()[assignments==idx]*0.5+0.5)[i].permute(1,2,0))
#     ax.set_title(np.arange(128)[assignments==idx][i])
#     ax.axis('off')
# 
# =============================================================================
