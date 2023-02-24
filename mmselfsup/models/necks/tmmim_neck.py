# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class TMMIMNeck(BaseModule):
    """Pre-train Neck For SimMIM.

    This neck reconstructs the original image from the shrunk feature map.

    Args:
        out_indices (tuple): Indices of feature level to compute.
        in_channels (list): Channel dimension of the feature map list.
        stem_stride (int): The stride of stem layer. Default to 4.
    """

    def __init__(self, 
                 dual_reconstruction=True,
                 out_indices=(3,),
                 in_channels=[256, 512, 1024, 2048], # resnet-50
                 stem_stride=4) -> None:
        super(TMMIMNeck, self).__init__()
        self.dual_reconstruction = dual_reconstruction
        self.out_indices = out_indices

        decoders_f2b = []
        for i in out_indices:
            encoder_stride = stem_stride*2**i
            decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=(encoder_stride)**2 * 3,
                    kernel_size=1),
                nn.PixelShuffle(encoder_stride),
            )
            decoders_f2b.append(decoder)
        self.decoders_f2b = nn.ModuleList(decoders_f2b)

        if dual_reconstruction:
            decoders_b2f = []
            for i in out_indices:
                encoder_stride = stem_stride*2**i
                decoder = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels[i],
                        out_channels=(encoder_stride)**2 * 3,
                        kernel_size=1),
                    nn.PixelShuffle(encoder_stride),
                )
                decoders_b2f.append(decoder)
            self.decoders_b2f = nn.ModuleList(decoders_b2f)

    def forward(self, xs_fg, xs_bg):
        """
        
        Arguments
        ---------
        xs_fg: the feature map list of backbone (foreground visible), tuple(s0,s1,s2,s3)
            s_k maybe None if backbone do not ouput k-th level.
        xs_bg: the feature map list of backbone (background visible), tuple(s0,s1,s2,s3)
        
        """

        xs_f2b = []
        for out_indice, decoder_f2b in zip(self.out_indices, self.decoders_f2b):
            x_f2b = decoder_f2b(xs_fg[out_indice])
            xs_f2b.append(x_f2b)

        if self.dual_reconstruction:
            assert len(xs_fg) == len(xs_bg)
            assert xs_bg is not None
            xs_b2f = []
            for out_indice, decoder_b2f in zip(self.out_indices, self.decoders_b2f):
                x_b2f = decoder_b2f(xs_bg[out_indice])
                xs_b2f.append(x_b2f)
        else:
            xs_b2f = None

        return xs_f2b, xs_b2f