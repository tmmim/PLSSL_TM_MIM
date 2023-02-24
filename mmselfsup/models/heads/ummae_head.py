import torch
from mmcls.models import LabelSmoothLoss
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class UMMAEHead(BaseModule):
    """Pre-training head for UM-MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        mask_patch_size (int): Patch size for image masking. Defaults to 16.
    """

    def __init__(self, norm_pix=False, mask_patch_size=16):
        super(UMMAEHead, self).__init__()
        self.norm_pix = norm_pix
        self.mask_patch_size = mask_patch_size

    def patchify(self, imgs):
        """
        Args
            imgs: (N, 3, H, W)
        
        Return 
            x: (N, L, patch_size**2 *3)
        """
        p = self.mask_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, x, pred, mask):
        """
        x: [N, 3, H, W]
        pred: [N, L, p*p*3], p is mask_patch_size
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        losses = dict()
        target = self.patchify(x) # (1,256,16x16x3)
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D) # (1,192,16x16x3)

        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = loss.sum() / mask.sum()
        losses['loss'] = loss
        return losses