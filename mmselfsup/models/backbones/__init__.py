# Copyright (c) OpenMMLab. All rights reserved.
from .cae_vit import CAEViT
from .mae_vit import MAEViT
from .mim_cls_vit import MIMVisionTransformer
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .simmim_swin import SimMIMSwinTransformer
from .vision_transformer import VisionTransformer


__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'MAEViT', 'MIMVisionTransformer',
    'VisionTransformer', 'SimMIMSwinTransformer', 'CAEViT'
]


from .tmmim_resnet import TMMIMResNet
from .tmmim_convnext import TMMIMConvnext
__all__ += ['TMMIMResNet', 'TMMIMConvnext']

