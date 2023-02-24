import pickle
from object_discovery import rle_decode
from PIL import Image
from datasets import Dataset, bbox_iou
from tqdm import tqdm
import numpy as np
import torch
from scipy.signal import savgol_filter
from visualizations import visualize_all_np

import matplotlib.pyplot as plt


def prepare_tr_cr(bbox, mask, img_w=None, patch_size=16, lamda=1.4, 
                  vis_dir=None, im_name=None):
    """
    Prepare tower region box and conductor region box, this function does not 
    consider resizing.

    Arguments
    ---------
        bbox: raw tower region (on raw image).
        mask: tower mask (on downsample feature map).
        img_w: width of raw image.
        patch_size: raw_image / mask_image.
        lamda: thredshold for refining tower box.
    
    Returns
    -------

    """
    tr_bbox = bbox.copy()

    # refine tower region (tr)
    y_proj = np.sum(mask, axis=0)
    _y_proj = y_proj.copy()
    y_proj_mean = np.mean(y_proj)

    y_proj = savgol_filter(y_proj, 11, 3)
    ind = np.where(y_proj > (y_proj_mean * lamda))[0]

    xmin, xmax = ind[0], ind[-1]
    tr_bbox[0] = xmin * patch_size
    tr_bbox[2] = xmax * patch_size

    # visualization
    if vis_dir is not None:
        cols = mask.shape[1]
        x = np.arange(0, cols, 1)
        plt.figure()
        plt.plot(x, y_proj)
        plt.hlines(y_proj_mean, min(x), max(x), color="red") 
        pltname = f"{OUTPUT_DIR}/{im_name}_proj_x.jpg"
        plt.savefig(pltname, bbox_inches='tight')
        plt.close()

    # conductor region (cr) generation
    if img_w is not None:
        tower_h = tr_bbox[3] - tr_bbox[1]
        tower_cx = int((tr_bbox[0] + (tr_bbox[2] - tr_bbox[0]) / 2) / patch_size)
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

        return tr_bbox, cr_box
    else:
        return tr_bbox, None

OUTPUT_DIR = 'outputs/debug/'
pkl_file = 'outputs/PLSelfSup_train/TokenCut-vit_small16_k/tower_mask.pkl'
with open(pkl_file, 'rb') as f:
    tower_info = pickle.load(f, encoding='bytes')

dataset = Dataset('PLSelfSup', '', '')

lamda = 1.0
refine_tower_region = True
MAX_VIS_NUM = 0
patch_size=16

draw_cnt = 0
cnt = 0
corloc = np.zeros(len(dataset.dataloader))
_corloc = np.zeros(len(dataset.dataloader))
iou_avg = []
pbar = tqdm(dataset.dataloader)
for im_id, inp in enumerate(pbar):

    im_name = dataset.get_image_name(inp[1])

    results = tower_info[im_name]
    _box_t = results['box']
    ct_point = results['ct_point']
    mask = rle_decode(results['rles'])

    gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

    # refine tower box and generate conductor box
    if refine_tower_region:
        try:
            box_t, _ = prepare_tr_cr(_box_t, mask, lamda=lamda)
            # box_t, _ = prepare_tr_cr(_box_t, mask, lamda=lamda, vis_dir=OUTPUT_DIR, im_name=im_name)
        except:
            print('Refine tower region failed!')
    else:
        box_t = _box_t

    _ious = bbox_iou(torch.from_numpy(_box_t), torch.from_numpy(gt_bbxs)) # before refine
    ious = bbox_iou(torch.from_numpy(box_t), torch.from_numpy(gt_bbxs)) # after refine
    iou_avg.append(ious.max())

    if torch.any(_ious >= 0.5):
        _corloc[im_id] = 1

    if torch.any(ious >= 0.5):
        corloc[im_id] = 1
        # visualize the better results
        if draw_cnt < MAX_VIS_NUM and not torch.any(_ious >= 0.5):
            image_path = 'datasets/PLSelfSup_sub1000/JPEGImages/'+im_name
            image = Image.open(image_path).convert("RGB") 
            masks = np.expand_dims(mask, axis=0)
            masks = masks.repeat(patch_size, axis = 1).repeat(patch_size, axis = 2) 
            W, H = image.size
            masks = masks[:, :H, :W] # (N, H, W)
            visualize_all_np(image, masks, _box_t, ct_point, OUTPUT_DIR, im_name, ext_name='raw')
            visualize_all_np(image, masks, box_t, ct_point, OUTPUT_DIR, im_name, ext_name='refine')
            draw_cnt += 1
    cnt += 1

print(f"before refine corloc: {100*np.sum(_corloc)/cnt:.2f} ({int(np.sum(_corloc))}/{cnt})")
print(f"after refine corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
print(f'after refine iou_avg: {sum(iou_avg)/cnt}')