import os
import json
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from .augmentation import TrivialAugmentation
from .nabirds_horn import load_bounding_box_annotations, load_part_annotations, load_part_names, load_class_names, \
     load_image_labels, load_image_paths, load_image_sizes, load_hierarchy, load_photographers, load_train_test_split

import random


class NABirdsDataset(Dataset):
    allowed_keys = ['crop', 'box_dir', 'return_path', 'trivial_aug', 'ops', 'high_res', 'n_pixel', 'return_mask', 'all_data', 'zeroshot_split']
    part_mapping = {'beak': [0], 'crown': [1], 'nape': [3], 'eyes': [3, 4], 'belly': [5], 'breast': [6], 'back': [7], 'tail': [8], 'wings': [9, 10]}

    def __init__(self, root_dir: str, transform: Compose = None, train: bool = True, zeroshot_split: bool = False, **kwargs):

        self.root_dir = root_dir
        self.zeroshot_split = zeroshot_split

        self.transform = transform
        self.train = train
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.allowed_keys})
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)

        self._load_meta()

        self.loader = default_loader
        self.totensor = transforms.ToTensor()

        # Data augmentation
        self.tri_aug = TrivialAugmentation(operations=[
            # 'ShearX',  # view point rotation
            # 'ShearY',
            # 'TranslateX',  # shifting
            # 'TranslateY',
            # 'Rotate',  # image rotation
            'Brightness',
            'Color',
            'Contrast',
            'Sharpness',
            # 'Posterize',  # quantize the color in 2^n, does not support float image
            # 'Solarize',   # invert the color below a threshold
            'AutoContrast',
            # 'Equalize'    # histogram equalization (the color will change), does not support float image
        ])

    def _load_meta(self):

        # ================== Load meta data ==================
        # Loading code adopt form NABirds sample script with minor modification
        self.image_paths = load_image_paths(self.root_dir, path_prefix='images')
        self.image_sizes = load_image_sizes(self.root_dir)
        self.image_bboxes = load_bounding_box_annotations(self.root_dir)
        self.image_parts = load_part_annotations(self.root_dir)
        self.image_class_labels = load_image_labels(self.root_dir)

        # Load in the class data
        self.class_names = load_class_names(self.root_dir)
        self.class_hierarchy = load_hierarchy(self.root_dir)

        # Load in the part data
        self.dataset_parts = load_part_names(self.root_dir)
        self.part_ids = list(self.dataset_parts.keys())
        self.part_ids = sorted(self.part_ids)

        # Load in the train / test split
        self.train_images, self.test_images = load_train_test_split(self.root_dir)

        # ===================================================
        # replace part names 'bill' with 'beak'
        # (this is the parts from the dataset, not to confused with part_name used for training, e.g., self.part_names)
        self.dataset_parts = {0: 'beak', 1: 'crown', 2: 'nape', 3: 'left eye', 4: 'right eye', 5: 'belly',
                              6: 'breast', 7: 'back', 8: 'tail', 9: 'left wing', 10: 'right wing'}

        all_samples = self.train_images + self.test_images
        all_classes = [int(self.image_class_labels[image_id]) for image_id in all_samples]
        unique_classes = set(all_classes)
        unique_class_names = [self.class_names[str(class_id)] for class_id in unique_classes]

        # there are 555 classes in total, re-index them from 0 to 554
        self.class2idx = {v: k for k, v in enumerate(unique_class_names)}
        self.idx2class = dict(enumerate(unique_class_names))
        self.nabirds_idx2class_idx = dict(zip(unique_classes, self.idx2class))
        self.class_idx2nabirds_idx = dict(zip(self.idx2class, unique_classes))

        # training samples
        self.target_classes = unique_classes
        self.zs_class2class_id = None
        self.class_id2zs_class = None

        if self.zeroshot_split:
            # choose 25% of the classes for testing (with seed 77)
            random.seed(77)     # Fixed random seed
            test_classes = set(random.sample(unique_classes, int(len(unique_classes) * 0.25)))
            train_classes = set(unique_classes) - test_classes
            all_imgs = self.train_images + self.test_images

            self.target_classes = train_classes if self.train else test_classes
            # reindex the classes to 0, 1, 2, ...
            self.zs_class2class_id = dict(enumerate(self.target_classes))
            self.class_id2zs_class = {v: k for k, v in self.zs_class2class_id.items()}
            image_ids = [image_id for image_id in all_imgs if int(self.image_class_labels[image_id]) in self.target_classes]
        elif hasattr(self, 'all_data') and self.all_data:
            image_ids = all_samples
        else:
            image_ids = self.train_images if self.train else self.test_images

        self.samples = []
        self.targets = []
        self.sample_parts = {}
        self.image_boxes = {}

        for image_id in image_ids:
            image_path = self.image_paths[image_id]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            class_label = self.nabirds_idx2class_idx[int(self.image_class_labels[image_id])]

            if self.class_id2zs_class is not None:
                class_label = self.class_id2zs_class[class_label]

            self.targets.append(class_label)
            self.samples.append((os.path.join(self.root_dir, image_path), class_label))
            self.sample_parts[image_name] = self.image_parts[image_id]
            self.image_boxes[image_name] = self.image_bboxes[image_id]

        self.classes = list(self.class2idx.keys())
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    # overwrite __getitem__
    def __getitem__(self, idx):
        image_path, target = self.samples[idx]

        sample = self.loader(image_path)
        sample_size = torch.tensor(sample.size[::-1])  # (h, w)

        if self.tri_aug is not None and self.train:
            sample = self.tri_aug(sample)

        if self.transform is not None:
            sample = self.transform(images=sample, return_tensor='pt')

        if hasattr(self, 'return_path') and self.return_path:
            return sample, target, image_path

        return sample, target