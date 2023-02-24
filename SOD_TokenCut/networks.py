"""
Loads model. 
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits
from mmcls.models import ResNet

#import moco.vits as vits_moco

def get_model(arch, patch_size, device, load_local=None):

    # Initialize model with pretraining
    url = None
    if ("moco" in arch) and ('vit' in arch):
        if arch == "moco_vit_small" and patch_size == 16:
            url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        elif arch == "moco_vit_base" and patch_size == 16:
            url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        model = vits.__dict__[arch](num_classes=0)
    elif "mae" in arch:
        if arch == "mae_vit_base" and patch_size == 16:
            url = "mae/visualize/mae_visualize_vit_base.pth"
        model = vits.__dict__[arch](num_classes=0)
    elif "vit" in arch:
        if arch == "vit_small" and patch_size == 16:
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" 
        elif arch == "vit_base" and patch_size == 16:
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "resnet50":
            url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    elif "moco_resnet50" in arch:
        url = r'https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth'
        model = ResNet(depth=50, out_indices=(0,1,2,3))
        state_dict = torch.hub.load_state_dict_from_url(url)
        state_dict = state_dict['state_dict']
        msg = model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError 

    for p in model.parameters():
        p.requires_grad = False

    if load_local:
        url = None
        state_dict = torch.load(load_local)
        state_dict = state_dict['state_dict']

        if 'moco' in arch:
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('backbone'):
                    if 'patch_embed' in k:
                        new_key = k[len("backbone."):].replace('projection','proj').replace('layers', 'blocks')
                    elif k == 'backbone.ln1.weight':
                        new_key = 'norm.weight'
                    elif k == 'backbone.ln1.bias':
                        new_key = 'norm.bias'
                    elif 'ln' in k:
                        new_key = k[len("backbone."):].replace('ln','norm').replace('layers', 'blocks')
                    elif 'attn' in k:
                        new_key = k[len("backbone."):].replace('layers', 'blocks')
                    elif 'ffn' in k and ('0.0' in k):
                        new_key = k[len("backbone."):].replace('ffn', 'mlp').replace('layers.0.0','fc1').replace('layers', 'blocks')
                    elif 'ffn' in k and ('.1.w' in k):
                        new_key = k[len("backbone."):].replace('ffn', 'mlp').replace('layers.1.w','fc2.w').replace('layers', 'blocks')
                    elif 'ffn' in k and ('.1.b' in k):
                        new_key = k[len("backbone."):].replace('ffn', 'mlp').replace('layers.1.b','fc2.b').replace('layers', 'blocks')
                    else:
                        new_key = k[len("backbone."):]
                    state_dict[new_key] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                load_local, msg
            )
        )


    if url is not None and ('vit' in arch):
        print(
            "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        if "moco" in arch:
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "mae" in arch:
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('decoder') or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )

    model.eval()
    model.to(device)
    return model
