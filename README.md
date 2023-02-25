# PLSSL_TM_MIM
This repo is the implementation of the under review paper.

# Installation

## Pretraining 
Please refer to the [Installation](https://mmselfsup.readthedocs.io/en/stable/get_started.html) of mmselfsup. The version of mmselfsup is 0.9.0.

## Finetuning 
Please refer to the [Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) of mmdetection. The version of mmdetection is 2.24.1.


# Usages

## Pretraining using TM-MIM

- Generate tower conductor region, output `tower_info.pkl`
```
cd PLSSL_TM_MIM/SOD_TokenCut
python main_tokencut.py --dataset {your_dataset}
```

- Verify the generated `tower_info.pkl`
```
python tcr_evalutation.py
```

- Self-supervised learning for ConvNeXt-t
```
cd ..
CUDA_VISIBLE_DEVICES=0 PORT=29501 bash tools/dist_train.sh \
    configs/selfsup/plselfsup/tmmim_convnext-t.py \
    1 \
    --work-dir work_dirs/selfsup/tmmim/convnext-t \
```

After pretaining, we obtain the pretrained model in: `./work_dirs/selfsup/tmmim/convnext-t/last.pth`

## Finetuning for PLCD

```
cd {your_mmdet_dir}
```

### Using `ImageNet-1K` pretrained model

1. Modify the backbone initialization in the Faster RCNN config file: `configs/faster_rcnn_convnext-t_fpn.py`
```
init_cfg = dict(type='Pretrained', checkpoint={in1k_pretrained_model_path}, prefix='backbone.')
```

2. Start finetuning.
```
CUDA_VISIBLE_DEVICES=0 PORT=29501 bash ./tools/dist_train.sh \
    configs/faster_rcnn_convnext-t_fpn.py \
    1 \
    logs/faster_rcnn_convnext-t_fpn_in1k
```

3. Test
```
bash tools/dist_test.sh \
    logs/faster_rcnn_convnext-t_fpn.py \
    logs/faster_rcnn_convnext-t_fpn_in1k/latest.pth \
    1 \
    --eval bbox \
```

### Using `TM-MIM` pretrained model
1. Modify the backbone initialization in the Faster RCNN config file: `configs/faster_rcnn_convnext-t_fpn.py`
```
init_cfg = dict(type='Pretrained', checkpoint={tmmim_pretrained_model_path}, prefix='backbone.')
```

2. Start finetuning.

```
CUDA_VISIBLE_DEVICES=0 PORT=29501 bash ./tools/dist_train.sh \
    configs/faster_rcnn_convnext-t_fpn.py \
    1 \
    logs/faster_rcnn_convnext-t_fpn_tmmim
```

3. Test
```
bash tools/dist_test.sh \
    logs/faster_rcnn_convnext-t_fpn.py \
    logs/faster_rcnn_convnext-t_fpn_tmmim/latest.pth \
    1 \
    --eval bbox \
```

# Acknowledgement
Our implementation is mainly based on the codebase of [mmselfsup](https://github.com/open-mmlab/mmselfsup), [mmdetection](https://github.com/open-mmlab/mmdetection), and [TokenCut](https://github.com/YangtaoWANG95/TokenCut). We gratefully thank the authors for their wonderful works.
