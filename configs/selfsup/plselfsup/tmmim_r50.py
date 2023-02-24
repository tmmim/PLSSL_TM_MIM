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
    ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth' # rsb from timm
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
        type='TMMIMResNet',
        depth=50,
        out_indices=out_indices,
        use_distillation=use_distillation,
        init_cfg=init_cfg),
    neck=dict(
        type='TMMIMNeck', 
        dual_reconstruction=dual_reconstruction,
        out_indices=out_indices,
        in_channels=[256, 512, 1024, 2048], 
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

# 2048 batchsize, 8e-4 lr
# 128 per gpu: 20G, with FPN is 26G, when fpn_out_channels=256 is 9.6G
# img size 384: 64 per gpu: 31G, when apply gn: 32 per gpu: 34G
# img size 512: 32 per gpu: 29G
# img size 1280: 16 per gpu: 30G
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