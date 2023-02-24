_base_ = [
    '../_base_/datasets/pl_tm_mim.py',
    '../_base_/default_runtime.py',
]

use_pretrain = True
dual_reconstruction = True
out_indices = (0,1,2,3) # backbone, neck, head
use_distillation = True
distillation_indices = (0,1,2,3)

if use_pretrain:
    ckpt = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth'
    init_cfg=dict(type='Pretrained', checkpoint=ckpt, prefix='backbone.')
else:
    init_cfg = None
    
# model settings
model = dict(
    type='TMMIM',
    dual_reconstruction=dual_reconstruction,
    base_momentum=1.,
    use_distillation=use_distillation,
    kd_loss='mse', # spport mse, kl, cos, ce. default: mse
    distillation_indices=distillation_indices,
    distillation_lambda=1, # defualt: 1
    backbone=dict(
        type='TMMIMConvnext',
        arch='tiny',
        out_indices=out_indices,
        gap_before_final_norm=False,
        use_distillation=use_distillation,
        init_cfg=init_cfg),
    neck=dict(
        type='TMMIMNeck', 
        dual_reconstruction=dual_reconstruction,
        out_indices=out_indices,
        in_channels=[96, 192, 384, 768], 
        stem_stride=4),
    head=dict(
        type='TMMIMHead', 
        patch_size=4, 
        encoder_in_channels=3,
        rec_loss='l1', # Support 'l1', 'mse', 'kl'. default: l1
        dual_reconstruction=dual_reconstruction,
        out_indices=out_indices,
        fpn_weight_f2b=[1., 1., 1., 1.],
        fpn_weight_b2f=[1., 1., 1., 1.],
        lambda_f2b=1, 
        lambda_b2f=1))

# img size 224: 128 per gpu: 26G
# img size 384: 64 per gpu: 34G
# img size 448: 32 per gpu: 23G
# img size 896: 8 per gpu: 23G
# default: 128*3*1
samples_per_gpu = 128
gpus = 3
update_interval = 1
data = dict(samples_per_gpu=samples_per_gpu)
total_bs = samples_per_gpu*gpus*update_interval

# optimizer: AdamW
optimizer = dict(
    type='AdamW', 
    lr=0.001, # 8e-4 * total_bs / 2048
    weight_decay=0.05,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
    })


# 梯度积累
optimizer_config = dict(grad_clip=dict(max_norm=3.0),
                        update_interval=update_interval)


lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5 * total_bs / 2048,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=1e-6 / 2e-4,
    warmup_by_epoch=True,
    # by_epoch=False
    )

# mixed precision
fp16 = dict(loss_scale='dynamic')

# schedule
runner = dict(type='EpochBasedRunner', max_epochs=30)

# runtime
checkpoint_config = dict(interval=50, max_keep_ckpts=1, out_dir='')
persistent_workers = True
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),])
# find_unused_parameters=True