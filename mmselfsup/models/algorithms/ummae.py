# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class UMMAE(BaseModel):
    """UM-MAE.

    Implementation of `Uniform Masking: Enabling MAE Pre-training for 
    Pyramid-based Vision Transformers with Locality`

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self, backbone=None, neck=None, head=None, init_cfg=None):
        super(UMMAE, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super(UMMAE, self).init_weights()

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        return self.backbone(img)

    def forward_train(self, x: List[torch.Tensor], **kwargs):
        """Forward computation during training.

        Args:
            x (List[torch.Tensor, torch.Tensor]): Images and masks.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img, mask = x

        latent = self.backbone(img, mask)
        pred, mask_len = self.neck(latent[0], mask)
        losses = self.head(img, pred[:, -mask_len:, :], mask)

        return losses
