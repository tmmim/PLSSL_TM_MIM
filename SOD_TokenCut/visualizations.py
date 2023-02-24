"""
Vis utilities. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image
import scipy

import matplotlib.pyplot as plt

def visualize_img(image, vis_folder, im_name):
    pltname = f"{vis_folder}/{im_name}"
    Image.fromarray(image).save(pltname)
    print(f"Original image saved at {pltname}.")

def visualize_predictions(img, pred, ct_point, vis_folder, im_name, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    cv2.rectangle(image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,)
    cv2.circle(image, (int(ct_point[0]),int(ct_point[1])), 5, (255, 0, 0), 3)
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_pred.jpg"
        Image.fromarray(image).save(pltname)
        print(f"Predictions saved at {pltname}.")
    return image
  
def visualize_predictions_gt(img, pred, gt, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )
    # Plot the ground truth box
    if len(gt>1):
        for i in range(len(gt)):
            cv2.rectangle(
                image,
                (int(gt[i][0]), int(gt[i][1])),
                (int(gt[i][2]), int(gt[i][3])),
                (0, 0, 255), 3,
            )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_BBOX.jpg"
        Image.fromarray(image).save(pltname)
        #print(f"Predictions saved at {pltname}.")
    return image

def visualize_eigvec(eigvec, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the second smallest eigvector
    """
    eigvec = scipy.ndimage.zoom(eigvec, scales, order=0, mode='nearest')
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_attn.jpg"
        plt.imsave(fname=pltname, arr=eigvec, cmap='cividis')
        print(f"Eigen attention saved at {pltname}.")

def visualize_mask(mask, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the mask after Ncut clustering.
    """
    mask = scipy.ndimage.zoom(mask, scales, order=0, mode='nearest')
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_mask.jpg"
        plt.imsave(fname=pltname, arr=mask, cmap='cividis')
        print(f"Ncut clustering mask saved at {pltname}.")

def visualize_mask_np(img, masks, pred, ct_point, vis_folder, 
                      im_name, scales, alpha=0.5) -> np.ndarray:
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
    # colors = np.array([[255,0,0],[0,255,0],[0,255,255]])
    colors = np.array([[255,0,0]]).reshape(-1,3)

    masks = np.expand_dims(masks,axis=0)
    masks = masks.repeat(scales[0], axis = 1).repeat(scales[1], axis = 2)

    H, W, _ = img.shape
    masks = masks[:, :H, :W]

    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)

    cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (255, 0, 0), 3)
    cv2.circle(img, (int(ct_point[0]),int(ct_point[1])), 5, (255, 255, 0), 3)

    pltname = f"{vis_folder}/{im_name}_TokenCut_maskimg.jpg"
    Image.fromarray(img.astype(np.uint8)).save(pltname)


def visualize_projection(attn_map, vis_folder, im_name, save=True):
    cols = attn_map.shape[1]
    x = np.arange(0, cols, 1)
    y_proj = np.sum(attn_map, axis=0)

    plt.plot(x, y_proj)

    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_projection.jpg"
        plt.savefig(pltname, bbox_inches='tight')
        print(f"Ncut clustering mask saved at {pltname}.")


def visualize_all_np(img, masks, bbox, ct_point, vis_folder, 
                      im_name, ext_name, gt=None, cr_box=None, alpha=0.5) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray or PIL.Image
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`
    bbox: numpy.ndarray (4,)
        Prediction box of the major object.
    ct_point: (int,int).
        center point with hihgest value in attn_map.
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    img = np.copy(img)
    # colors = np.array([[255,0,0],[0,255,0],[0,255,255]])
    colors = np.array([[255,0,0]]).reshape(-1,3)

    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)

    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)
    if isinstance(cr_box, np.ndarray):
        cv2.rectangle(img, (int(cr_box[0]), int(cr_box[1])), (int(cr_box[2]), int(cr_box[3])), (255, 0, 0), 3)

    cv2.circle(img, (int(ct_point[0]),int(ct_point[1])), 5, (255, 255, 0), 3)

    pltname = f"{vis_folder}/{im_name}_TokenCut_{ext_name}.jpg"
    Image.fromarray(img.astype(np.uint8)).save(pltname)


def visualize_projection(attn_map, vis_folder, im_name, save=True):
    cols = attn_map.shape[1]
    x = np.arange(0, cols, 1)
    y_proj = np.sum(attn_map, axis=0)

    plt.plot(x, y_proj)

    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_projection.jpg"
        plt.savefig(pltname, bbox_inches='tight')
        print(f"Ncut clustering mask saved at {pltname}.")