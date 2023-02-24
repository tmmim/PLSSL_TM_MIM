import torch
import torch.nn as nn
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import build_2d_sincos_position_embedding


@NECKS.register_module()
class UMMAENeck(BaseModule):
    """Decoder for UM-MAE Pre-training.

    Args:
        token_num (int): The number of total tokens which need to reconstruct. 
            Defaults to (input_size\mask_patch_size)^2=256. 
        mask_patch_size (int): Image patch size for masking. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        encoder_out_dim (int): Encoder's output dimension. Defaults to 1024 (Swin base).
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.

    Return:
        x (tensor): The decoder output to reconstruct input image. 
            Shape: (B, token_num**2, mask_patch_size**2*3).
        mask_len (int): The number of masked patches.

    Some of the code is borrowed from
    `https://github.com/facebookresearch/mae`.
    `https://github.com/implus/UM-MAE`

    Example:
        >>> from mmselfsup.models import UMMAENeck
        >>> import torch
        >>> self = UMMAENeck()
        >>> self.eval()
        >>> inputs = torch.rand(1, 16, 1024)
        >>> mask = torch.ones(1, 256).to(torch.bool)
        >>> mask[:,:64]=0 # vis patches after upsample. 4x4 to 8x8.
        >>> level_outputs = self.forward(inputs, mask)
        >>> print(tuple(level_outputs[0].shape),level_outputs[1])
        (1, 256, 768) 192
    """

    def __init__(self,
                 token_num=256,
                 mask_patch_size=16,
                 in_chans=3,
                 encoder_out_dim=1024,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(UMMAENeck, self).__init__()
        self.token_num = token_num

        # self.decoder_embed = nn.Linear(encoder_out_dim, decoder_embed_dim, bias=True) # MAE
        self.decoder_embed = nn.Linear(encoder_out_dim, 4 * decoder_embed_dim, bias=True) # UM-MAE
        self.decoder_expand = nn.PixelShuffle(2) # upsample 2x

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # (1,1,512)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.token_num, decoder_embed_dim),
            requires_grad=False) # (1,16x16,512)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, mask_patch_size**2 * in_chans, bias=True) # 512 -> 16x16x3


    def init_weights(self):
        super(UMMAENeck, self).init_weights()

        # initialize position embedding of MAE decoder
        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.token_num**.5),
            self.decoder_pos_embed.shape[-1],
            cls_token=False)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def decoder_norm(self):
        return getattr(self, self.decoder_norm_name)

    def forward(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x) # (1,16,2048)

        # upsample the x from Swin
        x_vis = x
        B, L, nD = x_vis.shape
        M = int(L**0.5) # 4
        x_vis = self.decoder_expand(
            x_vis.permute(0, 2, 1).reshape(-1, nD, M, M)).flatten(2) 
        x_vis = x_vis.permute(0, 2, 1)
        _, _, D = x_vis.shape # (1,64,512)

        # append mask tokens to sequence
        expand_pos_embed = self.decoder_pos_embed.expand(B, -1, -1) # (1,16x16,512)
        pos_vis = expand_pos_embed[~mask].reshape(B, -1, D) # (1,1x64,512), 0.25 visible
        pos_mask = expand_pos_embed[mask].reshape(B, -1, D) # (1,3x64,512)
        mask_len = pos_mask.shape[1]
        
        x = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1) # (1,4x64,512)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x) # (1,16x16,16x16x3)

        return x, mask_len