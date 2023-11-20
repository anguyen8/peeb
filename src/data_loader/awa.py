import os
import json
import random

import torch
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class AWA(Dataset):
    allowed_keys = ['trivial_aug', 'ops']
    def __init__(self, root: str, 
                 meta_path: str, 
                 box_dir: str = None, 
                 transform: Compose = None, 
                 use_meta_dir: bool = False, 
                 return_path: bool = False, 
                 subset: list[str] = None, 
                 dataset_key: str = 'awa', 
                 is_train: bool = True,
                 all_data: bool = False,
                 **kwargs):
        self.root = root
        self.meta_df = pd.read_hdf(meta_path, key=dataset_key)
        self.subset = subset
        self.use_meta_dir = use_meta_dir
        self.return_path = return_path
        self.box_dir = box_dir
        self.transform = transform
        self.is_train = is_train
        self.all_data = all_data
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)
        
        if hasattr(self, 'trivial_aug') and self.trivial_aug:
            self.tri_aug = TrivialAugmentCustom(operations=self.ops)
        else:
            self.tri_aug = None

        if root is None and not use_meta_dir:
            raise ValueError("root and use_meta_dir cannot be both None")    
        
        self._get_meta()
    
    def _get_meta(self,):
        self.att_weight_org = torch.load(os.path.join(self.root, 'original_att.pth'))
        self.att_weight = torch.load(os.path.join(self.root, 'att.pth'))
        
        if self.subset is not None and set(data_sources).issubset(set(self.subset)):
            data_sources = self.meta_df['data_source'].unique().tolist()
            self.meta_df = self.meta_df[self.meta_df['data_source'].isin(self.subset)]

        if not self.all_data:
            if self.is_train:
                self.meta_df = self.meta_df[self.meta_df['is_train'] == 1]
            else:
                self.meta_df = self.meta_df[self.meta_df['is_train'] == 0]
            subset_class_idx = self.meta_df['class_id'].unique().tolist()
            self.att_weight = self.att_weight[:, subset_class_idx]
            self.att_weight_org = self.att_weight_org[:, subset_class_idx]
                
            # reset class_id to start from 0
            new_id, uniques = pd.factorize(self.meta_df['class_id'])
            self.meta_df['class_id'] = new_id

        self.targets = self.meta_df['class_id'].values.tolist()

        if self.use_meta_dir:
            image_paths = self.meta_df['abs_path'].values.tolist()
            self.samples = list(zip(image_paths, self.targets))
        elif 'relative_path' in self.meta_df.columns:
            self.meta_df['image_path'] = self.meta_df['relative_path'].apply(lambda x: os.path.join(self.root, x))
            self.samples = list(zip(self.meta_df['image_path'].values.tolist(), self.targets))
        else:
            self.meta_df['image_path'] = self.meta_df['image_name'].apply(lambda x: os.path.join(self.root, x))
            self.samples = list(zip(self.meta_df['image_path'].values.tolist(), self.targets))


        # if there are no duplicated classes
        if len(self.meta_df['class_id'].unique()) == len(self.meta_df['class_name'].unique()):
            self.idx2class = dict(zip(self.meta_df['class_id'].unique(), self.meta_df['class_name'].unique()))
        else:
            unique_class_ids = sorted(self.meta_df['class_id'].unique())
            # get the corresponding class names
            class_names = [self.meta_df[self.meta_df['class_id'] == class_id]['class_name'].unique()[0] for class_id in unique_class_ids]
            self.idx2class = dict(zip(unique_class_ids, class_names))

        self.class2idx = {v: k for k, v in self.idx2class.items()}
        self.classes = list(self.class2idx.keys())
        # # check correctness of this mapping
        # for k, v in self.idx2cls.items():
        #     assert self.meta_df[self.meta_df['class_id'] == k]['class_name'].unique()[0] == v
        
        self.att_list = json.load(open(os.path.join(self.root, 'att_dict.json')))
        self.att_binary = torch.load(os.path.join(self.root, 'att_binary.pth'))
        

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        sample = default_loader(image_path)
        # size = torch.tensor(sample.size[::-1]) # (h, w)
        
        if hasattr(self, 'trivial_aug') and self.trivial_aug:
            sample = self.tri_aug(sample)
        if self.transform is not None:
            sample = self.transform(images=sample, return_tensor='pt')
        
        return sample, target, image_path
            
        