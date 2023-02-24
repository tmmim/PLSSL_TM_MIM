"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from pycocotools.mask import encode, decode


def rle_encode(masks: np.ndarray):
    """
    
    Arguments
    ---------
        masks (array): one_hot_masks, (H, W, N) or (H, W)
    
    Returns
    -------
        rles (dict): {'size':[H, W], 'counts':b'...'}

    """
    
    masks = masks.astype(np.uint8)
    rles: dict = encode(np.asfortranarray(masks))

    return rles


def rle_decode(rles: dict):
    """
    
    Arguments
    ---------
        rles (dict): {'size':[H, W], 'counts':b'...'}
        
    Returns
    -------
        masks (array): one_hot_masks, (H, W, N) or (H, W)

    """
    
    masks = decode(rles)

    return masks

def GMM(eigvec):
    gmm = GaussianMixture(n_components=2, max_iter=300)
    gmm.fit(eigvec)
    partition = gmm.predict(eigvec)
    return partition

def Kmeans_partition(eigvec):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(eigvec)
    return kmeans.labels_

def ncut(feats, dims, scales, init_image_size, 
         tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats (B, hxw, C): the pixel/patche features of an image 
      dims (h, w): dimension of the feature map
      scales: from image to map scale, patch size (16) or downsample stride (32)
      init_image_size (3, H, W): initial size of the image
      tau: thresold for graph construction. default 0.2
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0,0:1,:].cpu().numpy() 

    feats = feats[0,1:,:]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0)) 
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]

    # Average clustering
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    # k-means cluster
    # bipartition = Kmeans_partition(eigenvectors)

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects, cc, ct_point = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    rles = rle_encode(mask)

    return np.asarray(pred), objects, mask, rles, ct_point, eigenvec.reshape(dims)

def ncut_conv(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', 
              binary_graph=False, ncut_mode=''):
    """
    Implementation of NCut Method for convolutional networks.
    Inputs
      feats (B, hxw, C): the pixel/patche features of an image 
      dims (h, w): dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size (3, H, W): size of the image
      tau: thresold for graph construction. default 0.2
      eps: graph edge weight
      im_name: image_name
      binary_graph: ablation study for using similarity score as graph edge weight
    """
    feats = feats[0,:,:] # (hxw, C)
    # pair-wise cosine distance
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0)) # (hxw, hxw)
    A = A.cpu().numpy()
    if binary_graph:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    else:
        A=A      

    d_i = np.sum(A, axis=1)
    D = np.diag(d_i) 
    
    # Print second and third smallest eigenvector, start from second
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0]) # (hxw, )
    second_smallest_vec = eigenvectors[:, 0]

    # avg cluster
    # avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    # bipartition = second_smallest_vec > avg

    # k-means cluster
    bipartition = Kmeans_partition(eigenvectors)

    seed = np.argmax(np.abs(second_smallest_vec))
    # seed = np.argmax(second_smallest_vec)

    # ensure the highest attention point belonging to the foreground
    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    ## We only extract the principal object BBox
    pred, _, objects, cc, ct_point = detect_box(bipartition, 
                                        seed, 
                                        dims, 
                                        scales=scales, 
                                        initial_im_size=init_image_size[1:]) 
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    return np.asarray(pred), objects, mask, ct_point, eigenvec.reshape(dims)

def detect_box(bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch.  Among connected components extract 
    from the affinity matrix, select the one corresponding to the seed patch.

    ct_point: the point which has hihest abs value in attn_map, already mapping to raw input.
    """
    h_featmap, w_featmap = dims
    objects, num_objects = ndimage.label(bipartition) # get connected regions with different labels
    cc = objects[np.unravel_index(seed, dims)] # find the region-label where the attention is highest in eigenvec
    
    ct_row = seed // w_featmap
    ct_col = seed % w_featmap
    ct_point = (ct_col*scales[1], ct_row*scales[0]) # (x, y)

    if principle_object:
        mask = np.where(objects == cc) # the belonging region (index) is what we need
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        
        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask, ct_point
    else:
        raise NotImplementedError

