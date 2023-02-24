# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.utils import print_log

from pycocotools.mask import decode
from scipy.signal import savgol_filter
from torchvision.transforms import Compose
from mmcv.utils import build_from_cfg

from .tmbase import TMBaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy

import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def visualize(img, masks, box,
              im_name, scales=(16,16), alpha=0.5) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray or PIL.Image
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, h, w)`, h and w are size of feature map. 
    pred: numpy.ndarray (4,)
        Prediction box of the major object.
    ct_point: (int,int).
        center point with hihgest value in attn_map.
    scales: (int,int).
        Stride or patch size of the backbone model.
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    img = np.copy(img)

    cols = masks.shape[-1]
    y_proj = np.sum(masks, axis=0)
    _y_proj = y_proj.copy()
    y_proj_mean = np.mean(y_proj)
    y_proj = savgol_filter(y_proj, 11, 3)


    # colors = np.array([[255,0,0],[0,255,0],[0,255,255]])
    colors = np.array([[255,0,0]]).reshape(-1,3)

    masks = np.expand_dims(masks,axis=0)
    masks = masks.repeat(scales[0], axis = 1).repeat(scales[1], axis = 2)

    H, W, _ = img.shape
    masks = masks[:, :H, :W]

    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)

    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)

    vis_folder = 'work_dirs/debug'
    pltname = f"{vis_folder}/{im_name}_TokenCut_maskimg.jpg"
    Image.fromarray(img.astype(np.uint8)).save(pltname)

    x = np.arange(0, cols, 1)
    plt.figure()
    plt.plot(x, y_proj)
    plt.hlines(y_proj_mean, min(x), max(x), color="red") 
    pltname = f"{vis_folder}/{im_name}_projection_x.jpg"
    plt.savefig(pltname, bbox_inches='tight')
    plt.close()

def prepare_tr_cr(box_t, mask, img_w, patch_size=16, lamda=1.):
    """
    Prepare tower region box and conductor region box, this function does not 
    consider resizing.

    Arguments
    ---------
        box_t: raw tower region (on raw image).
        mask: tower mask (on downsample feature map).
        img_w: width of raw image.
        patch_size: raw_image / mask_image.
        lamda: thredshold for refining tower box.
    
    Returns
    -------

    """

    # horizental projection
    cols = mask.shape[1]
    y_proj = np.sum(mask, axis=0)
    _y_proj = y_proj.copy()
    y_proj_mean = np.mean(y_proj)

    y_proj = savgol_filter(y_proj, 11, 3)
    ind = np.where(y_proj > (y_proj_mean * lamda))[0]

    xmin, xmax = ind[0], ind[-1]
    box_t[0] = xmin * patch_size
    box_t[2] = xmax * patch_size

    # vertical projection
    rows = mask.shape[0]
    y = np.arange(0, rows, 1)

    # conductor region (cr) generation
    tower_h = box_t[3] - box_t[1]
    tower_cx = int((box_t[0] + (box_t[2] - box_t[0]) / 2) / patch_size)
    ind = np.where(_y_proj > 0)[0]
    _lenth = (tower_cx - ind[0], ind[-1]-tower_cx)
    if _lenth[0] > _lenth[1]:
        cr_x = ind[0]
    else:
        cr_x = ind[-1]
    cr_y = np.where(mask[:, cr_x]>0)[0][0]

    cr_y1 = int(cr_y * patch_size - tower_h / 3 / 2)
    cr_y2 = int(cr_y * patch_size + tower_h / 3 / 2)
    cr_box = np.array([1, cr_y1, img_w, cr_y2])

    return box_t, cr_box

@DATASETS.register_module()
class TowerMaskDataset(TMBaseDataset):
    """The dataset outputs one view of an image, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(TowerMaskDataset, self).__init__(data_source, pipeline,
                                                prefetch)
        self.gt_labels = self.data_source.get_gt_labels()
        self.tower_infos = self.data_source.load_tower_info()
        self.prefetch = prefetch

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        img = self.data_source.get_img(idx)
        img_w, img_h = img.size # PIL Image (W, H)
        file_name = self.data_source.data_infos[idx]['img_info']['filename']

        # tower_info: {'box': array(4,), 'rle': coco rle mask, ct_point: tuple(2),
        #               'raw_img_hw_size':(H, W),'resize_img_hw_size':(h, w)}
        # logit: raw_img -> resized_img (optimal) -> mask_img
        tower_info = self.tower_infos[file_name] 
        box_t = tower_info['box'] # tower region on raw_img or resized_img
        rle_mask = decode(tower_info['rles']) # tower mask on mask_img. (h/16, w/16)

        # refine tower box and generate conductor box
        try:
            box_t, box_c = prepare_tr_cr(box_t, rle_mask, img_w)
        except:
            # visualize(img, mask, box_t, file_name)
            box_c = box_t

        # map to real raw image if the tower region is generate in a resized image
        rle_mask = rle_mask.repeat(16, axis=0).repeat(16, axis=1) # tokencut use 16 patch_size, map to raw image size
        if 'resize_img_hw_size' in list(tower_info.keys()):
            raw_h, raw_w = tower_info['raw_img_hw_size']
            resize_h, resize_w = tower_info['resize_img_hw_size']
            rle_mask = rle_mask[:resize_h, :resize_w]
            h_scale = raw_h / resize_h
            w_scale = raw_w / resize_w

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

            box_t = box_t * scale_factor
            box_c = box_c * scale_factor

            rle_mask = cv2.resize(rle_mask, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        bbox = np.vstack((box_t, box_c))    

        # np to pil
        rle_mask = np.array(rle_mask, dtype=np.uint8)
        rle_mask = rle_mask[:img_h, :img_w]
        rle_mask = Image.fromarray(rle_mask)

        img = self.pipeline((img, bbox, rle_mask))

        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        eval_res = {}
        for name, val in results.items():
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.data_source.get_gt_labels())
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')

            num = val.size(0)
            _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                acc = correct_k * 100.0 / num
                eval_res[f'{name}_top{k}'] = acc
                if logger is not None and logger != 'silent':
                    print_log(f'{name}_top{k}: {acc:.03f}', logger=logger)
        return eval_res
