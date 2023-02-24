# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.backbones import ResNet as _ResNet
from mmcls.models.backbones.resnet import BasicBlock, Bottleneck

from mmcv.cnn.bricks import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_

import torch
import torch.nn as nn
from ..builder import BACKBONES


@BACKBONES.register_module()
class TMMIMResNet(_ResNet):
    """ResNet backbone for Tower mask MIM.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int): Output channels of the stem layer. Defaults to 64.
        base_channels (int): Middle channels of the first stage.
            Defaults to 64.
        num_stages (int): Stages of the network. Defaults to 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to ``(0, 1, 2, 3)``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Defaults to False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        conv_cfg (dict | None): The config dict for conv layers.
            Defaults to None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Defaults to False.
        Probability of the path to be zeroed. Defaults to 0.1
    Example:
        >>> from mmselfsup.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 use_distillation=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0,
                 **kwargs):
        super(TMMIMResNet, self).__init__(
            depth=depth,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            expansion=expansion,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            init_cfg=init_cfg,
            drop_path_rate=drop_path_rate,
            **kwargs)
        self.use_distillation = use_distillation
        self.mask_token = nn.Parameter(torch.zeros(1, stem_channels, 1, 1)) # (B,C,H,W)

    def init_weights(self) -> None:
        """Initialize weights."""
        super(_ResNet, self).init_weights()

        trunc_normal_(self.mask_token, mean=0, std=.02)

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, visible_mode='foreground'):
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        them.

        Args:
            x (torch.Tensor): Input images. (B,3,H,W)
            mask (torch.Tensor): Masks used to construct masked images. (B, h, w)
                h = H//4 and w = W//4, refer to the stride of stem layer.
            visible_mode (str): decide which type of patches to mask.
                default to 'foreground', can be 'background'.

        Returns:
            tuple, tuple: The tuple containing features from multi-stages. One is 
                masked, another is non-masked.
        """
        # stem layer
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)  # r50: 64x128x128
        x = self.maxpool(x)  # r50: 64x56x56
        x_all = x

        # mask the output of stem layer
        B, _, H, W = x.shape
        mask_token = self.mask_token.expand(B, -1, H, W)
        w_fg = mask.unsqueeze(1).type_as(mask_token) # 1 for foreground
        w_bg = 1. - w_fg

        if visible_mode == 'foreground':
            x = x * w_fg + mask_token * w_bg # visible patches are foreground
        elif visible_mode == 'background':
            x = x * w_bg + mask_token * w_fg # visible patches are background

        outs= []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
            else:
                outs.append(None)

        if self.use_distillation:
            outs_all = []
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                x_all = res_layer(x_all)
                if i in self.out_indices:
                    outs_all.append(x_all)
                else:
                    outs_all.append(None)
        else:
            outs_all = [None, None, None, None]

        # r50: 0-256x56x56; 1-512x28x28; 2-1024x14x14; 3-2048x7x7
        return tuple(outs), tuple(outs_all)