# dataset settings
data_source = 'PLSelfSup'
dataset_type = 'TowerMaskDataset'
img_norm_cfg = dict(mean=[0.5126, 0.5567, 0.5268], std=[0.2819, 0.2415, 0.2519]) # pl_norm
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # in1k_norm

image_size = 224
train_pipeline = [
    dict(
        type='RandomResizedCropWithBox',
        size=image_size,
        scale=(0.67, 1.0),
        ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomHorizontalFlipWithBox'),
    dict(type='RandomAppliedTranswithBox',
         transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
         ],
         p=0.8),
    dict(type='RandomGrayscalewithBox', p=0.2),
    dict(type='GaussianBlurwithBox', sigma_min=0.1, sigma_max=2.0, p=0.5),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='TMToTensor'),
         dict(type='TMNormalize', **img_norm_cfg)])

train_pipeline.append(
    dict(
        type='TowerMaskGenerator',
        input_size=image_size,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.2,
        tower_mask=True,
        tm_rand_mask=False,
        tm_rand_mask_ratio=.6, # mask ratio in tower mask region, only effect when tm_rand_mask=True
        use_rle_mask=False,
        reverse=False,
        hollow=False,
        hollow_size=4,
        stem_fea_size=True))

# dataset summary
data = dict(
    samples_per_gpu=256, # 8gpu * 256 = 2048
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/PLSelfSup/train',
            ann_file='data/PLSelfSup/meta/train.txt',
            pkl_file='data/PLSelfSup/tower_info.pkl',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch))
