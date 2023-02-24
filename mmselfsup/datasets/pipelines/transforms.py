# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import math
import random
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F

from mmcv.utils import build_from_cfg
from PIL import Image, ImageFilter
from timm.data import create_transform
from torchvision import transforms as _transforms

from einops.einops import rearrange

from ..builder import PIPELINES

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module(force=True)
class ToTensor(object):
    """Convert image or a sequence of images to tensor.

    This module can not only convert a single image to tensor, but also a
    sequence of images.
    """

    def __init__(self) -> None:
        self.transform = _transforms.ToTensor()

    def __call__(self, imgs: Union[object, Sequence[object]]) -> torch.Tensor:
        if isinstance(imgs, Sequence):
            imgs = list(imgs)
            for i, img in enumerate(imgs):
                imgs[i] = self.transform(img)
        else:
            imgs = self.transform(imgs)
        return imgs


@PIPELINES.register_module()
class SimMIMMaskGenerator(object):
    """Generate random block mask for each Image.

    This module is used in SimMIM to generate masks.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
    """

    def __init__(self,
                 input_size: int = 192,
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.6) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        mask = torch.from_numpy(mask)  # H X W

        return img, mask

@PIPELINES.register_module()
class DualMIMMaskGenerator(object):
    """Generate random block mask for each Image.

    This module is used in DualMIM to generate masks.

    Args:
        input_size (int): Size of input image. Defaults to 224.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.5.
    """

    def __init__(self,
                 input_size: int = 224,
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.5) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size # 7
        self.scale = self.mask_patch_size // self.model_patch_size # 8

        self.token_count = self.rand_size**2 # 49
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))

        mask = torch.from_numpy(mask)  # (7, 7)

        return img, mask

@PIPELINES.register_module()
class UMMAEMaskGenerator(object):
    """Generate random block mask for each Image.

    This module is used in UM-MAE to generate masks.

    Args:
        input_size (int): Size of input image. Defaults to 256.
        mask_patch_size (int): Size of each block mask. Defaults to 16.
        mask_ratio (float): The mask ratio of image. Defaults to 0.75.
    """

    def __init__(self,
                 input_size: int = 256,
                 mask_patch_size: int = 16, 
                 mask_ratio: float = 0.75) -> None:
        self.token_size = int(input_size // mask_patch_size) 
        self.mask_ratio = mask_ratio
        self.token_count = self.token_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        candidate_list = []
        while True: # add more
            for j in range(4):
                candidate = np.ones(4,dtype=int)
                candidate[j] = 0
                candidate_list.append(candidate)
            if len(candidate_list) * 4 >= self.token_count * 2:
                break
        self.mask_candidate = np.vstack(candidate_list) 

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask_candidate.copy()
        np.random.shuffle(mask)
        mask = rearrange(mask[:self.token_count//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                            h=self.token_size//2, w=self.token_size//2, p1=2, p2=2)
        mask = mask.flatten()

        mask = torch.from_numpy(mask).to(torch.bool)  # (L,)

        return img, mask

@PIPELINES.register_module()
class MyMaskGenerator(object):
    """Generate random block mask for each Image.

    Args:
        input_size (int): Size of input image. Defaults to 256.
        mask_patch_size (int): Size of each block mask. Defaults to 16.
        stem_patch_size (int): a point in stem layer featuremap view how 
            big size region in the input RGB image. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.75.
    """

    def __init__(self,
                 input_size: int = 256,
                 mask_patch_size: int = 16, 
                 stem_patch_size: int = 4,
                 mask_ratio: float = 0.75) -> None:
        self.token_size = int(input_size // mask_patch_size) # 16
        self.mask_ratio = mask_ratio
        self.token_count = self.token_size**2 # 256
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio)) # 196
        self.scale = mask_patch_size // stem_patch_size # 4

        candidate_list = []
        while True: # add more
            for j in range(4):
                candidate = np.ones(4,dtype=int)
                candidate[j] = 0
                candidate_list.append(candidate)
            if len(candidate_list) * 4 >= self.token_count * 2: 
                break
        self.mask_candidate = np.vstack(candidate_list) # (128, 4)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask_candidate.copy()
        np.random.shuffle(mask)
        # (128, 4) -> (64, 4) -> (16,16)
        mask = rearrange(mask[:self.token_count//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                            h=self.token_size//2, w=self.token_size//2, p1=2, p2=2)

        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1) # (64, 64)
        mask = torch.from_numpy(mask)
        
        return img, mask

@PIPELINES.register_module()
class ResizeWithBox(object):
    def __init__(self, size):
        super().__init__()

        self.size = (size, size)

    def __call__(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img, bbox, rle_mask = data
        img_w, img_h = img.size

        img = img.resize(self.size, resample=0)

        w_scale = self.size[0] / img_w 
        h_scale =  self.size[0] / img_h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        bbox = bbox * scale_factor

        rle_mask = rle_mask.resize(self.size, resample=0)

        return (img, bbox, rle_mask)

@PIPELINES.register_module()
class RandomResizedCropWithBox(object):
    """Crop a random portion of image and resize it to a given size with a given box.
       Using for Tower mask MIM
    
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=''):
        super().__init__()
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        if isinstance(img, torch.Tensor):
            width, height = img.shape[-1], img.shape[-2]
        else: 
            width, height = img.size[0], img.size[1]
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, data):
        """
        Args:
            data (img, bbox, rle_mask):
                img (PIL Image): Image to be cropped and resized. (W, H)
                bbox (array): Boxes to be cropped and resized. (N, 4)
                rle_mask (PIL Image): RLE mask of foreground. (H, W)
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img, bbox, rle_mask = data

        top, left, height, width = self.get_params(img, self.scale, self.ratio)
        # crop img
        img = img.crop((left, top, left + width, top + height)) # PIL Image
        img_w, img_h = img.size

        # crop bbox
        bbox_offset = np.array([left, top, left, top], dtype=np.float32)
        bbox = bbox - bbox_offset

        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, img_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, img_h)

        # crop mask
        rle_mask = rle_mask.crop((left, top, left + width, top + height)) # PIL Image
        # print(rle_mask.size)

        # rle_mask = rle_mask[top:top+height, left:left+width] # array

        # resize img
        img = img.resize(self.size, resample=0)

        # resize bbox
        # if there is no bbox after crop, return bbox=None
        valid_inds = (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
        if not valid_inds.any():
            bbox = []
        else: 
            w_scale = self.size[0] / img_w 
            h_scale =  self.size[0] / img_h
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            bbox = bbox * scale_factor

        # resize mask
        rle_mask = rle_mask.resize(self.size, resample=0) # PIL Image
        # rle_mask = np.resize(rle_mask, self.size) # array

        return (img, bbox, rle_mask)

@PIPELINES.register_module()
class RandomHorizontalFlipWithBox(object):
    """Horizontally flip the given image randomly with a given probability.
       Using for Tower mask MIM 

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
            bbox (array): bbox to be flipped. (N, 4)
            rle_mask (PIL Image): RLE mask of foreground. (H, W)

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img, bbox, rle_mask = data

        if torch.rand(1) < self.p:
            if len(bbox) > 0:
                w = img.size[0]
                flipped = bbox.copy()
                flipped[..., 0::4] = w - bbox[..., 2::4]
                flipped[..., 2::4] = w - bbox[..., 0::4]
                flipped[np.abs(flipped) < 1e-3] = 0 # maybe some values are -1eN
                return (F.hflip(img), flipped, F.hflip(rle_mask))
            else:
                return (F.hflip(img), bbox, F.hflip(rle_mask))

        return (img, bbox, rle_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

@PIPELINES.register_module()
class RandomAppliedTranswithBox(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, data):
        img, bbox, rle_mask = data
        return (self.trans(img), bbox, rle_mask)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob = {self.prob}'
        return repr_str

@PIPELINES.register_module()
class RandomGrayscalewithBox(torch.nn.Module):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        img, bbox, rle_mask = data

        num_output_channels = F._get_image_num_channels(img) # _get_image_num_channels() in v0.10.0
        if torch.rand(1) < self.p:
            return (F.rgb_to_grayscale(img, num_output_channels=num_output_channels), bbox, rle_mask)
        return (img, bbox, rle_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

@PIPELINES.register_module()
class GaussianBlurwithBox(object):
    """GaussianBlur augmentation refers to `SimCLR.

    <https://arxiv.org/abs/2002.05709>`_.

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, sigma_min, sigma_max, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, data):
        img, bbox, rle_mask = data

        if np.random.rand() > self.prob:
            return (img, bbox, rle_mask)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return (img, bbox, rle_mask)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob}'
        return repr_str

@PIPELINES.register_module()
class TMToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
       Using for Tower mask MIM.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """
    def __call__(self, data):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic, bbox, rle_mask = data
        return (F.to_tensor(pic), bbox, rle_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

@PIPELINES.register_module()
class TMNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
       Using for Tower mask MIM.

    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor, bbox, rle_mask = data
        return (F.normalize(tensor, self.mean, self.std, self.inplace), bbox, rle_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


@PIPELINES.register_module()
class TowerMaskGenerator(object):
    """Generate random block mask in tower region for each Image.

    This module is used in Tower mask MIM to generate masks.

    tower mask (tm):
        tower_mask = True,
        tm_rand_mask = False,
        use_rle_mask = False,
        reverse = False,

    random mask (rm):
        tower_mask = False,
        reverse = False,

    rle mask (rle):
        use_rle_mask = True

    Args:
        input_size (int): Size of input image. Defaults to 224.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        tower_mask (boolean): if False, use total random masking.
        tm_rand_mask (boolean): if False, mask all the tower region.
        tm_rand_mask_ratio (float): The mask ratio of tower region. Defaults to 0.8.
        use_rle_mask (boolean): if True, use rle mask.
        stem_fea_size (boolean): if True, upsample the mask from final feature map
            to stem feature map.
    """

    def __init__(self,
                 input_size: int = 224,
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.6,
                 tower_mask = True,
                 tm_rand_mask = False,
                 tm_rand_mask_ratio: float = 0.8,
                 use_rle_mask = False,
                 reverse = False,
                 hollow = False,
                 hollow_size = 4,
                 stem_fea_size = True) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.tower_mask = tower_mask
        self.tm_rand_mask = tm_rand_mask
        self.tm_rand_mask_ratio = tm_rand_mask_ratio
        self.stem_fea_size = stem_fea_size
        self.reverse = reverse
        self.hollow = hollow
        self.hollow_size = hollow_size
        self.use_rle_mask = use_rle_mask

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.stem_size = self.input_size // self.model_patch_size # 224/4=56
        self.rand_size = self.input_size // self.mask_patch_size # 224/32=7
        self.scale = self.mask_patch_size // self.model_patch_size # 32/4=8
        self.hollow_scale = self.hollow_size // self.model_patch_size # 4/4=1

        self.token_count = self.rand_size**2 # 7**2=49
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, data):
        img, bbox, rle_mask = data

        if self.use_rle_mask:
            mask = torch.from_numpy(np.array(rle_mask))
            if self.stem_fea_size:
                # rle_mask (torch.tensor of (1, 224, 224) -> (1, 56, 56))
                mask = F.resize(mask.unsqueeze(0), 
                                (self.stem_size, self.stem_size), 
                                F._interpolation_modes_from_int(2)).squeeze()
            else:
                # rle_mask (torch.tensor of (1, 224, 224) -> (1, 7, 7))
                mask = F.resize(mask.unsqueeze(0), 
                                (self.rand_size, self.rand_size), 
                                F._interpolation_modes_from_int(2)).squeeze()
            
            if self.reverse:
                mask = 1 - mask
            
            return img, mask

        if (len(bbox) > 0) and self.tower_mask:
            # mask the tower region
            bbox = bbox / self.mask_patch_size
            bbox[:, 0:2] = np.floor(bbox[:, 0:2])
            bbox[:, 2:4] = np.ceil(bbox[:, 2:4])
            bbox = bbox.astype(np.int8)

            mask = np.zeros((self.rand_size, self.rand_size), dtype=int)

            for b in bbox:
                mask[b[1]:b[3], b[0]:b[2]] = 1

            if self.tm_rand_mask:
                mask = mask.flatten()
                mask_ind = np.where(mask==1)[0]
                cnt = int(np.sum(mask[mask == 1]))
                tower_mask_count = int(np.ceil(cnt * (1-self.tm_rand_mask_ratio)))
                mask_ind_ind = np.random.permutation(cnt)[:tower_mask_count]
                mask[mask_ind[mask_ind_ind]]=0
                mask = mask.reshape((self.rand_size, self.rand_size))
            
            # mask patches are too small
            if mask.mean() < 0.1:
                mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
                mask = np.zeros(self.token_count, dtype=int)
                mask[mask_idx] = 1
                mask = mask.reshape((self.rand_size, self.rand_size))                
        else: 
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
            mask = np.zeros(self.token_count, dtype=int)
            mask[mask_idx] = 1
            mask = mask.reshape((self.rand_size, self.rand_size))

        # if self.stem_fea_size:
        #     # (7, 7) -> (56, 56)
        #     mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        mask = torch.from_numpy(mask) # (7, 7)

        if self.hollow:
            shift = (self.scale - self.hollow_scale) // 2
            mask_ind = torch.nonzero(mask)
            mask_length = len(mask_ind)
            mask_ind = mask_ind.repeat_interleave(self.hollow_scale*self.hollow_scale, 0)
            increment = torch.ones(self.hollow_scale, self.hollow_scale)
            increment = torch.nonzero(increment)
            increment = increment.repeat(mask_length, 1)
            mask_ind = (mask_ind * self.scale + shift) + increment
            mask = mask.repeat_interleave(self.scale, 0).repeat_interleave(self.scale, 1).contiguous()
            mask[mask_ind[:,0], mask_ind[:,1]] = 0
        else:
            mask = mask.repeat_interleave(self.scale, 0).repeat_interleave(self.scale, 1).contiguous()

        
        if self.reverse:
            mask = 1 - mask

        return img, mask

@PIPELINES.register_module()
class BlockMaskGenerator(object):
    """Generate mask for TMMIM.

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit

    Args:
        input_size (int): The size of input image.
        num_masking_patches (int): The number of patches to be masked.
        min_num_patches (int): The minimum number of patches to be masked
            in the process of generating mask. Defaults to 4.
        max_num_patches (int, optional): The maximum number of patches to be
            masked in the process of generating mask. Defaults to None.
        min_aspect (float): The minimum aspect ratio of mask blocks. Defaults
            to 0.3.
        min_aspect (float, optional): The minimum aspect ratio of mask blocks.
            Defaults to None.
    """

    def __init__(self,
                 input_size: int,
                 num_masking_patches: int,
                 min_num_patches: int = 4,
                 max_num_patches: Optional[int] = None,
                 min_aspect: float = 0.3,
                 max_aspect: Optional[float] = None) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None \
            else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches,
                                         max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top:top + h, left:left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, data):
        img, bbox, rle_mask = data

        mask = np.zeros((self.height, self.width), dtype=np.int)
        mask_count = 0
        while mask_count != self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            mask_count += delta

        mask = mask.repeat(8, axis=0).repeat(8, axis=1) # (7, 7) -> (56, 56)
        mask = torch.from_numpy(mask)

        return img, mask

@PIPELINES.register_module()
class BEiTMaskGenerator(object):
    """Generate mask for image.

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit

    Args:
        input_size (int): The size of input image.
        num_masking_patches (int): The number of patches to be masked.
        min_num_patches (int): The minimum number of patches to be masked
            in the process of generating mask. Defaults to 4.
        max_num_patches (int, optional): The maximum number of patches to be
            masked in the process of generating mask. Defaults to None.
        min_aspect (float): The minimum aspect ratio of mask blocks. Defaults
            to 0.3.
        min_aspect (float, optional): The minimum aspect ratio of mask blocks.
            Defaults to None.
    """

    def __init__(self,
                 input_size: int,
                 num_masking_patches: int,
                 min_num_patches: int = 4,
                 max_num_patches: Optional[int] = None,
                 min_aspect: float = 0.3,
                 max_aspect: Optional[float] = None) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None \
            else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self) -> None:
        repr_str = 'Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)' % (
            self.height, self.width, self.min_num_patches,
            self.max_num_patches, self.num_masking_patches,
            self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches,
                                         max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top:top + h, left:left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(
        self, img: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count != self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            mask_count += delta

        return img[0], img[1], mask


@PIPELINES.register_module()
class RandomResizedCropAndInterpolationWithTwoPic(object):
    """Crop the given PIL Image to random size and aspect ratio with random
    interpolation.

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size. This is popularly used
    to train the Inception networks. This module first crops the image and
    resizes the crop to two different sizes.

    Args:
        size (Union[tuple, int]): Expected output size of each edge of the
            first image.
        second_size (Union[tuple, int], optional): Expected output size of each
            edge of the second image.
        scale (tuple[float, float]): Range of size of the origin size cropped.
            Defaults to (0.08, 1.0).
        ratio (tuple[float, float]): Range of aspect ratio of the origin aspect
            ratio cropped. Defaults to (3./4., 4./3.).
        interpolation (str): The interpolation for the first image. Defaults
            to ``bilinear``.
        second_interpolation (str): The interpolation for the second image.
            Defaults to ``lanczos``.
    """

    interpolation_dict = {
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

    def __init__(self,
                 size: Union[tuple, int],
                 second_size=None,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 second_interpolation='lanczos') -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn('range should be of kind (min, max)')

        if interpolation == 'random':
            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self.interpolation_dict.get(
                interpolation, Image.BILINEAR)
        self.second_interpolation = self.interpolation_dict.get(
            second_interpolation, Image.BILINEAR)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: np.ndarray, scale: tuple,
                   ratio: tuple) -> Sequence[int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect
                ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(
            self, img: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size,
                                  interpolation), F.resized_crop(
                                      img, i, j, h, w, self.second_size,
                                      self.second_interpolation)


@PIPELINES.register_module()
class RandomAug(object):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation
    with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    This code is borrowed from <https://github.com/pengzhiliang/MAE-pytorch>
    """

    def __init__(self,
                 input_size=None,
                 color_jitter=None,
                 auto_augment=None,
                 interpolation=None,
                 re_prob=None,
                 re_mode=None,
                 re_count=None,
                 mean=None,
                 std=None):

        self.trans = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            mean=mean,
            std=std,
        )

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob = {self.prob}'
        return repr_str


# custom transforms
@PIPELINES.register_module()
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise).

    Args:
        alphastd (float, optional): The parameter for Lighting.
            Defaults to 0.1.
    """

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self, alphastd=0.1):
        self.alphastd = alphastd
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            f'Expect torch.Tensor, got {type(img)}'
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'alphastd = {self.alphastd}'
        return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """GaussianBlur augmentation refers to `SimCLR.

    <https://arxiv.org/abs/2002.05709>`_.

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, sigma_min, sigma_max, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module()
class Solarization(object):
    """Solarization augmentation refers to `BYOL.

    <https://arxiv.org/abs/2006.07733>`_.

    Args:
        threshold (float, optional): The solarization threshold.
            Defaults to 128.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, threshold=128, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'

        self.threshold = threshold
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 - img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob}'
        return repr_str
