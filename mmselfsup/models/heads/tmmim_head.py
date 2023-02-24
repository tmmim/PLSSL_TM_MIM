# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule
from torch.nn import functional as F
import torch.nn as nn

from ..builder import HEADS

def mask_Nxdownsample(mask, nx):
    B, H, W = mask.shape
    h, w = H // nx, W // nx
    iw = torch.linspace(0, W-1, w).long()
    ih = torch.linspace(0, H-1, h).long()

    return mask[..., ih[:, None], iw]

@HEADS.register_module()
class TMMIMHead(BaseModule):
    """Pretrain Head for TMMIM.

    Args:
        patch_size (int): Patch size of each token.
        encoder_in_channels (int): Number of input channels for encoder.
        dual_reconstruction (float): If use dual reconstruction.
        rec_loss (str): Loss type for reconstruction. Support 'l1', 'mse', 'kl'.
        out_indices (tuple): Indices of feature level to compute.
        fpn_weight_f2b (list): weight of each feature level.
        fpn_weight_b2f (list): weight of each feature level.
        lambda_f2b (flaot): weight of foreground to background reconstruction.
        lambda_b2f (flaot): weight of background to foreground reconstruction.

    """

    def __init__(self, patch_size: int, encoder_in_channels: int,
                 dual_reconstruction=True,
                 rec_loss='l1',
                 out_indices=(3,),
                 fpn_weight_f2b=[1.,1.,1.,1.], fpn_weight_b2f=[1.,1.,1.,1.],
                 lambda_f2b=1., lambda_b2f=1.,) -> None:
        super(TMMIMHead, self).__init__()
        self.patch_size = patch_size
        self.encoder_in_channels = encoder_in_channels
        self.dual_reconstruction = dual_reconstruction
        self.out_indices = out_indices
        self.fpn_weight_f2b = fpn_weight_f2b
        self.fpn_weight_b2f = fpn_weight_b2f
        self.lambda_f2b = lambda_f2b
        self.lambda_b2f = lambda_b2f

        if rec_loss == 'l1':
            self.rec_loss = F.l1_loss
        elif rec_loss == 'mse':
            self.rec_loss = F.mse_loss
        elif rec_loss == 'kl':
            self.rec_loss = self.rec_kl_loss

    def rec_kl_loss(self, y, x, reduction='none'):
        x = F.log_softmax(x, dim=1)
        y = F.softmax(y, dim=1)
        loss = F.kl_div(x, y, reduction=reduction)

        return loss

    def forward(self, x, x_recs_f2b, x_recs_b2f, mask) -> dict:
        """
        
        Arguments
        ---------
            x (tensor): input images of (B,3,H,W). 
            x_recs_f2b (list[tensors]): reconstruction images from different feature
                levels, [(B,3,H,W),...]. 
            x_recs_b2f (list[tensors]): reconstruction images from different feature
                levels, [(B,3,H,W),...]. 
            mask (tensor): 1 for foreground, (B,H//4,W//4)
        
        """
        losses = dict()

        # resize mask from stem layer stride to input image.
        w_fg = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
                self.patch_size, 2).unsqueeze(1).contiguous()
        w_bg = 1 - w_fg

        loss_f2b_total = 0
        N = len(self.out_indices)
        for i in range(N):
            loss_f2b = self.rec_loss(x, x_recs_f2b[i], reduction='none')
            loss_f2b = (loss_f2b * w_bg).sum() / (w_bg.sum() +
                                            1e-5) / self.encoder_in_channels
            losses[f'f2b_stage{self.out_indices[i]}'] = loss_f2b
            loss_f2b_total += loss_f2b * self.fpn_weight_f2b[self.out_indices[i]]
        loss_f2b_total /= N

        if self.dual_reconstruction:
            assert x_recs_b2f is not None
            loss_b2f_total = 0
            for i in range(N):
                loss_b2f = self.rec_loss(x, x_recs_b2f[i], reduction='none')
                loss_b2f = (loss_b2f * w_fg).sum() / (w_fg.sum() +
                                                1e-5) / self.encoder_in_channels
                losses[f'b2f_stage{self.out_indices[i]}'] = loss_b2f
                loss_b2f_total += loss_b2f * self.fpn_weight_b2f[self.out_indices[i]]
            loss_b2f_total /= N
        else: 
            loss_b2f_total = 0

        loss = self.lambda_f2b * loss_f2b_total + self.lambda_b2f * loss_b2f_total
        losses['loss'] = loss
        losses['f2b'] = loss_f2b_total
        if self.dual_reconstruction:
            losses['b2f'] = loss_b2f_total

        return losses
