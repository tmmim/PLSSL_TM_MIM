from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcls.models import ConvNeXt
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import BACKBONES

def mask_Nxdownsample(mask, nx):
    B, H, W = mask.shape
    h, w = H // nx, W // nx
    iw = torch.linspace(0, W-1, w).long()
    ih = torch.linspace(0, H-1, h).long()

    return mask[..., ih[:, None], iw]

@BACKBONES.register_module()
class TMMIMConvnext(ConvNeXt):
    """ConvNext for Tower mask MIM.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): Initialization config dict
    """
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 use_distillation=True,
                 init_cfg=None):
        super().__init__(
                 arch=arch,
                 in_channels=in_channels,
                 stem_patch_size=stem_patch_size,
                 norm_cfg=norm_cfg,
                 act_cfg=act_cfg,
                 linear_pw_conv=linear_pw_conv,
                 drop_path_rate=drop_path_rate,
                 layer_scale_init_value=layer_scale_init_value,
                 out_indices=out_indices,
                 frozen_stages=frozen_stages,
                 gap_before_final_norm=gap_before_final_norm,
                 init_cfg=init_cfg)
        self.use_distillation = use_distillation
        self.mask_token = nn.Parameter(torch.zeros(1, self.channels[0], 1, 1)) # (B,C,H,W)

    def init_weights(self) -> None:
        """Initialize weights."""
        super(ConvNeXt, self).init_weights()

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

    def forward(self, x, mask=None, visible_mode='foreground') -> Sequence[torch.Tensor]:
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

        x_all = x

        outs = []
        # 0-stem(4x)-stage0, 1-down(2x)-stage1, 2-down(2x)-stage2, 3-down(2x)-stage3
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x) # include stem layer
            
            # mask the output of stem layer
            if i == 0 and isinstance(mask, torch.Tensor):
                B, _, H, W = x.shape
                mask_token = self.mask_token.expand(B, -1, H, W)
                w_fg = mask.unsqueeze(1).type_as(mask_token) # 1 for foreground
                w_bg = 1. - w_fg

                if visible_mode == 'foreground':
                    x = x * w_fg + mask_token * w_bg # visible patches are foreground
                elif visible_mode == 'background':
                    x = x * w_bg + mask_token * w_fg # visible patches are background        

            x = stage(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x).contiguous())
            else:
                outs.append(None)
        
        if self.use_distillation:
            outs_all = []
            # 0-stem(4x)-stage0, 1-down(2x)-stage1, 2-down(2x)-stage2, 3-down(2x)-stage3
            for i, stage in enumerate(self.stages):
                x_all = self.downsample_layers[i](x_all) # include stem layer
                x_all = stage(x_all)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    outs_all.append(norm_layer(x_all).contiguous())
                else:
                    outs_all.append(None)
        else:
            outs_all = [None, None, None, None]

        return tuple(outs), tuple(outs_all)