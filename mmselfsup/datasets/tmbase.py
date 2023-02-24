
# Copyright (c) OpenMMLab. All rights reserved.
from cmath import pi
import warnings
from abc import ABCMeta, abstractmethod

from mmcv.utils import build_from_cfg
from torch.utils.data import Dataset

import collections

from .builder import PIPELINES, build_datasource


class Compose:
    """Compose multiple transforms sequentially for Tower mask MIM.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string


class TMBaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset class for Tower mask MIM.

    The base dataset can be inherited by different algorithm's datasets. After
    `__init__`, the data source and pipeline will be built. Besides, the
    algorithm specific dataset implements different operations after obtaining
    images from data sources.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        warnings.warn('The dataset part will be refactored, it will soon '
                      'support `dict` in pipelines to save more information, '
                      'the same as the pipeline in `MMDet`.')
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.CLASSES = self.data_source.CLASSES

    def __len__(self):
        return len(self.data_source)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def evaluate(self, results, logger=None, **kwargs):
        pass
