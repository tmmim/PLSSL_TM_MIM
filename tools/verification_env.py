# -*- encoding: utf-8 -*-
'''
Filename         :verification_env.py
Description      :To verify whether MMSelfSup is installed correctly, 
                 we can run the following sample code to initialize 
                 a model and inference a demo image.
Time             :2022/05/05 20:29:50
Author           :***
Version          :1.0
'''

import torch

from mmselfsup.models import build_algorithm

model_config = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=-1),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=1000))

model = build_algorithm(model_config).cuda()

image = torch.randn((1, 3, 224, 224)).cuda()
label = torch.tensor([1]).cuda()

loss = model.forward_train(image, label)

print('loss:', loss)