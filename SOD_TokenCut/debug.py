
import numpy as np

from PIL import Image

import pickle
from pycocotools.mask import encode, decode
from visualizations import visualize_mask_np, visualize_all_np
from object_discovery import rle_decode
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

import math
def prepare_tr_cr(box_t, mask, img_w, patch_scale=16, lamda=1.4):
    """
    Prepare tower region box and conductor region box

    Arguments
    ---------
        box_t: raw tower region (on raw image).
        mask: tower mask (on mask image).
        img_w: width of raw image.
        patch_scale: raw_image / mask_image.
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
    box_t[0] = xmin * patch_scale
    box_t[2] = xmax * patch_scale

    # vertical projection
    rows = mask.shape[0]
    y = np.arange(0, rows, 1)

    # conductor region (cr) generation
    tower_h = box_t[3] - box_t[1]
    tower_cx = int((box_t[0] + (box_t[2] - box_t[0]) / 2) / patch_scale)
    ind = np.where(_y_proj > 0)[0]
    _lenth = (tower_cx - ind[0], ind[-1]-tower_cx)
    if _lenth[0] > _lenth[1]:
        cr_x = ind[0]
    else:
        cr_x = ind[-1]
    cr_y = np.where(mask[:, cr_x]>0)[0][0]

    cr_y1 = int(cr_y * patch_scale - tower_h / 3 / 2)
    cr_y2 = int(cr_y * patch_scale + tower_h / 3 / 2)
    cr_box = np.array([1, cr_y1, img_w, cr_y2])

    return box_t, cr_box

def win_max(array1d, win_size=3):
    slide_arr = np.lib.stride_tricks.sliding_window_view(
    array1d,  # 原数组
    win_size  # 窗口大小，也可以是元组，如(2，2)
    )
    win_sum = np.max(slide_arr, axis=1)
    ind = np.argmax(win_sum)

    return ind + 1

OUTPUT_DIR = 'outputs/debug/'
pkl_file = 'outputs/rtlod1300/TokenCut-vit_small16_k/tower_info.pkl'
patch_size = 16
with open(pkl_file, 'rb') as f:
    tower_info = pickle.load(f, encoding='bytes')

cnt = 0
for k, v in tower_info.items():
    im_name = k

    pred = v['box']
    mask = rle_decode(v['rles'])
    ct_point = v['ct_point']

    image_path = '/mnt/ssd2/lxy/database/rtlod1300/train/'+im_name
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    im_name = im_name.split('.jpg')[0]

    box_t, box_c = prepare_tr_cr(pred, mask, image.shape[1]) 

    # horizental projection
    # cols = mask.shape[1]
    # x = np.arange(0, cols, 1)
    # y_proj = np.sum(mask, axis=0)
    # _y_proj = y_proj.copy()
    # y_proj_mean = np.mean(y_proj)

    # tower region refinement
    # y_proj = savgol_filter(y_proj, 11, 3)
    # ind = np.where(y_proj > y_proj_mean)[0]
    # xmin, xmax = ind[0], ind[-1]
    # pred[0] = xmin * patch_size
    # pred[2] = xmax * patch_size

    # plt.figure()
    # plt.plot(x, y_proj)
    # plt.hlines(y_proj_mean, min(x), max(x), color="red") 
    # pltname = f"{OUTPUT_DIR}/{im_name}_projection_x.jpg"
    # plt.savefig(pltname, bbox_inches='tight')

    ## vertical projection
    # rows = mask.shape[0]
    # y = np.arange(0, rows, 1)
    # x_proj = np.sum(mask, axis=1)
    # x_proj_mean = np.mean(x_proj)

    # cr_y_cand_ind = find_peaks(x_proj, distance=5)
    # cr_y_cand = x_proj[cr_y_cand_ind]

    # max1 = np.sort(cr_y_cand)[-1]
    # max_index1 = np.argsort(cr_y_cand)[-1]

    # cr_y1 = int(max_index1 * patch_size - tower_h / 3 / 2)
    # cr_y2 = int(max_index1 * patch_size + tower_h / 3 / 2)
    # cr_box1 = np.array([1, cr_y1, image.shape[1]-1, cr_y2])

    # max2 = np.sort(cr_y_cand)[-2]
    # max_index2 = np.argsort(cr_y_cand)[-2]

    # cr_y1 = int(max_index2 * patch_size - tower_h / 3 / 2)
    # cr_y2 = int(max_index2 * patch_size + tower_h / 3 / 2)
    # cr_box2 = np.array([1, cr_y1, image.shape[1]-1, cr_y2])

    # plt.figure()
    # plt.plot(x_proj, y)
    # plt.vlines(x_proj_mean, min(y), max(y), color="red") 
    # plt.ylim(max(y),min(y)) # 翻转y轴
    # pltname = f"{OUTPUT_DIR}/{im_name}_projection_y.jpg"
    # plt.savefig(pltname, bbox_inches='tight')
    # plt.close()

    print(list(v.keys()))
    if 'resize_img_size' in list(v.keys()):
        print(v['raw_img_size'], v['resize_img_size'])
        raw_h, raw_w = v['raw_img_size']
        _, resize_h, resize_w = tuple(v['resize_img_size'])
        h_scale = raw_h / resize_h
        w_scale = raw_w / resize_w

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        box_t = box_t * scale_factor
        box_c = box_c * scale_factor
        ct_point = ct_point * np.array([w_scale, h_scale], dtype=np.float32)

        mask = mask.repeat(patch_size, axis = 0).repeat(patch_size, axis = 1)
        mask = mask.repeat(math.ceil(h_scale), axis = 0).repeat(math.ceil(w_scale), axis = 1)
        mask = mask[:raw_h, :raw_w]
        scales = (1, 1)
    else:
        scales = (patch_size, patch_size)

    visualize_all_np(image, mask, box_t, ct_point, OUTPUT_DIR, im_name, scales, cr_box=box_c)

    cnt += 1
    if cnt > 3: break

