import math
import os
import json
import random

import torch
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset, Subset, Sampler
from torchvision.datasets.folder import default_loader
from src.data_loader.augmentation import TrivialAugmentation
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict, deque
import copy


class BirdSoup(Dataset):

    def __init__(self, root: str, meta_path: str, box_dir: str = None, transform: Compose = None, use_meta_dir: bool = False,
                 train=True, subset: list[str] = None, return_path: bool = False):

        self.root_dir = root

        self.transform = transform
        self.train = train

        self.meta_df = pd.read_hdf(meta_path)

        self.use_meta_dir = use_meta_dir
        self.box_dir = box_dir
        self.subset = subset        # list of dataset names

        self.return_path = return_path

        if root is None and not use_meta_dir:
            raise ValueError("root and use_meta_dir cannot be both None")

        self._load_meta()

        # Data augmentation
        self.tri_aug = TrivialAugmentation(operations=[
            # 'ShearX',       # view point rotation
            # 'ShearY',
            # 'TranslateX',   # shifting
            # 'TranslateY',
            # 'Rotate',       # image rotation
            'Brightness',
            'Color',
            'Contrast',
            'Sharpness',
            # 'Posterize',  # quantize the color in 2^n, does not support float image
            # 'Solarize',   # invert the color below a threshold
            'AutoContrast',
            # 'Equalize'    # histogram equalization (the color will change), does not support float image
        ])
    
    def _load_meta(self,):
        # hot fix, remove underscore in class_name if keyword `stanford` in root.
        if 'stanford' in self.root_dir:
            self.meta_df['class_name'] = self.meta_df['class_name'].apply(lambda x: x.replace('_', ' '))
        
        data_sources = []
        if 'data_source' in self.meta_df.columns:
            data_sources = self.meta_df['data_source'].unique().tolist()

        if self.subset is not None:
            # TODO: CHECK RE_INDEXING OF CLASS_ID FOR SUB_CLASSES
            if "all" not in self.subset and set(self.subset).issubset(set(data_sources)):
                self.meta_df = self.meta_df[self.meta_df['data_source'].isin(self.subset)]

                # SORT dataframe by class_id
                self.meta_df = self.meta_df.sort_values(by=['class_id'])

                # and RE-INDEX the class_ids
                self.meta_df['class_id'] = self.meta_df['class_id'].astype('category').cat.codes

        self.targets = self.meta_df['class_id'].values.tolist()
        if self.use_meta_dir:
            if 'abs_path' in self.meta_df.columns:
                image_paths = self.meta_df['abs_path'].values.tolist()
            else:
                # combine `image_path` with root
                image_paths = self.meta_df['image_path'].apply(lambda x: os.path.join(self.root_dir, x)).values.tolist()
            self.samples = list(zip(image_paths, self.targets))
        else:
            self.meta_df['new_image_path'] = self.meta_df['new_image_name'].apply(lambda x: os.path.join(self.root_dir, "images", x))
            self.samples = list(zip(self.meta_df['new_image_path'].values.tolist(), self.targets))

        # if has "new_iamge_name" column, use it as the image name otherwise use base name of image_path
        if 'new_image_name' in self.meta_df.columns:
            self.birdsoup_image_name = self.meta_df['new_image_name'].values.tolist()
        else:
            self.birdsoup_image_name = self.meta_df['image_path'].apply(lambda x: os.path.basename(x)).values.tolist()
        
        # Note: The unique() function returns the unique elements of the dataframe in top-bottom order.
        #      The order of the unique elements in the returned array generally preserves the original ordering.
        self.idx2class = dict(zip(self.meta_df['class_id'].unique(), self.meta_df['class_name'].unique()))
        self.class2idx = {v: k for k, v in self.idx2class.items()}

        if len(list(self.meta_df['class_id'].unique())) != len(list(self.meta_df['class_name'].unique())):
            print("WARNING: class_id and class_name are not 1-1 mapping")
        else:
            # sanity check: correctness of this mapping
            for k, v in self.idx2class.items():
                assert self.meta_df[self.meta_df['class_id'] == k]['class_name'].unique()[0] == v

        self.classes = list(self.class2idx.keys())
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        sample = default_loader(image_path)
        soup_name = self.birdsoup_image_name[idx]
        size = torch.tensor(sample.size[::-1])      # (h, w)

        if self.train:
            sample = self.tri_aug(sample)

        if self.transform is not None:
            sample = self.transform(images=sample, return_tensor='pt')

        if self.return_path:
            return sample, target, image_path
        
        return sample, target, size, image_path, soup_name


