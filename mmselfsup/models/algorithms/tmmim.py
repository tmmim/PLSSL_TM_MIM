# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class TMMIM(BaseModel):
    """Tower mask MIM.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        base_momentum (float): Update param for momentum training.
        use_distillation (bool): Whether to use knowledge distillation.
        distillation_indices (tuple): Indices of feature level for distillation.
        distillation_lambda (float): Loss weight of distillation.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 dual_reconstruction=True,
                 base_momentum=0.996,
                 use_distillation=True,
                 kd_loss='mse', # spport mse, kl, cos, ce
                 distillation_indices=(3,),
                 distillation_lambda=1.,
                 init_cfg: Optional[dict] = None) -> None:
        super(TMMIM, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        if dual_reconstruction or use_distillation:
            self.teacher = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

        self.use_distillation = use_distillation
        self.dual_reconstruction = dual_reconstruction
        self.momentum = base_momentum
        self.distillation_indices = distillation_indices
        self.distillation_lambda = distillation_lambda

        self.similarity_loss = torch.nn.CosineSimilarity()

        if kd_loss == 'mse':
            self.kd_loss = self.distillation_mse_loss
        elif kd_loss == 'kl':
            self.kd_loss = self.distillation_kl_loss
        elif kd_loss == 'ce':
            self.kd_loss = self.distillation_cross_entropy_loss
        elif kd_loss == 'cos':
            self.kd_loss = self.distillation_cos_loss

    def init_weights(self) -> None:
        super().init_weights()
        if self.use_distillation or self.dual_reconstruction:
            self._init_teacher()

    def _init_teacher(self) -> None:
        # init the weights of teacher with those of backbone
        for param_backbone, param_teacher in zip(self.backbone.parameters(),
                                                 self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_backbone.data)
            param_teacher.requires_grad = False

    def momentum_update(self) -> None:
        """Momentum update of the teacher network."""
        for param_bacbone, param_teacher in zip(self.backbone.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + \
                param_bacbone.data * (1. - self.momentum)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def distillation_cross_entropy_loss(self, fs_list, ft_list):
        loss_total = 0
        N = len(self.distillation_indices)
        for i in self.distillation_indices:
            s_logits = F.softmax(fs_list[i], dim=1)
            t_logits = F.softmax(ft_list[i], dim=1)
            loss = torch.sum(t_logits * s_logits, dim=1)
            loss = loss.mean()

            loss_total += loss
        loss_total /= N

        return loss_total

    def distillation_cos_loss(self, fs_list, ft_list):
        loss_total = 0
        N = len(self.distillation_indices)
        B = fs_list[0].shape[0]
        for i in self.distillation_indices:
            loss = torch.mean(1 - self.similarity_loss(fs_list[i].view(B, -1), 
                ft_list[i].view(B, -1)))

            loss_total += loss
        loss_total /= N

        return loss_total


    def distillation_kl_loss(self, fs_list, ft_list):
        loss_total = 0
        N = len(self.distillation_indices)
        for i in self.distillation_indices:
            fs = F.log_softmax(fs_list[i], dim=1)
            ts = F.softmax(ft_list[i], dim=1)
            loss = F.kl_div(fs, ts, reduction="batchmean")

            loss_total += loss
        loss_total /= N

        return loss_total


    def distillation_mse_loss(self, fs_list, ft_list):
        loss_total = 0
        N = len(self.distillation_indices)
        for i in self.distillation_indices:
            loss = F.mse_loss(fs_list[i], ft_list[i], reduction='mean')

            loss_total += loss
        loss_total /= N

        return loss_total

    def extract_feat(self, img: torch.Tensor) -> tuple:
        """Function to extract features from backbone.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Latent representations of images.
        """
        _, feats = self.backbone(img)

        return feats

    def mixup_batch(self, img, mask):
        B = img.shape[0]
        mix_len = int(B*self.mixup_ratio)

        w = mask.repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1).contiguous()
        subidx = torch.randperm(mix_len)
        idx = torch.arange(0, B, 1)
        idx[:mix_len] = subidx
        img_suffle = img[idx, ...].view(img.shape)

        img_mask = img_suffle * (1-w)
        img_fg = img * w

        img_mix = img_mask + img_fg

        return img_mix

    def forward_train(self, x: List[torch.Tensor], **kwargs) -> dict:
        """Forward the masked image and get the reconstruction loss.

        Args:
            x (List[tensor, tensor]): Images and masks.

        Returns:
            dict: Reconstructed loss.
        """
        img, mask = x

        outs_fg, outs_all = self.backbone(img, mask, visible_mode='foreground')

        if self.use_distillation or self.dual_reconstruction:
            with torch.no_grad():
                outs_bg, outs_latent_all = self.teacher(img, mask, visible_mode='background')
                self.momentum_update()
        else:
            outs_bg = None
            outs_latent_all = None

        img_recs_f2b, img_recs_b2f = self.neck(outs_fg, outs_bg)
        losses = self.head(img, img_recs_f2b, img_recs_b2f, mask)

        if self.use_distillation:
            loss_distillation = self.kd_loss(outs_all, outs_latent_all)
            loss_distillation = loss_distillation * self.distillation_lambda
            losses['loss'] += loss_distillation
            losses['distillation'] = loss_distillation

        return losses
