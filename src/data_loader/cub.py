import os
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, InterpolationMode, ToTensor, Compose
from .augmentation import TrivialAugmentation

import random


class CUB(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset.
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 image_type=None,
                 zeroshot_split=False,
                 return_path=False):

        self.root_dir = root
        image_type = "images" if image_type is None else f"images_{image_type}"
        img_root = os.path.join(root, image_type)

        super(CUB, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        self.return_path = return_path

        if zeroshot_split:
            indices_to_use = None   # Use all examples
            unique_classes = set(self.targets)
            # choose 25% of the classes for testing (with seed 77)
            random.seed(77)
            test_classes = set(random.sample(unique_classes, int(len(unique_classes) * 0.25)))
            train_classes = set(unique_classes) - test_classes
            self.target_classes = train_classes if self.train else test_classes

            # reindex the classes to 0, 1, 2, ...
            self.zs_class2class_id = dict(enumerate(self.target_classes))
            self.class_id2zs_class = {v: k for k, v in self.zs_class2class_id.items()}

            self.imgs = self.samples = [(img, self.class_id2zs_class[id]) for img, id in self.imgs if id in self.target_classes]
            self.img_paths = [img_path for img_path, _ in self.imgs]
            _, self.targets = list(zip(*self.samples))
        else:
            self.target_classes = set(self.targets)
            self.zs_class2class_id = None
            self.class_id2zs_class = None

            # obtain sample ids filtered by split
            path_to_splits = os.path.join(root, 'train_test_split.txt')
            indices_to_use = []
            with open(path_to_splits, 'r') as in_file:
                for line in in_file:
                    idx, use_train = line.strip('\n').split(' ', 2)
                    if bool(int(use_train)) == self.train:
                        indices_to_use.append(int(idx))

            # obtain filenames of images
            path_to_index = os.path.join(root, 'images.txt')
            filenames_to_use = []
            with open(path_to_index, 'r') as in_file:
                for line in in_file:
                    idx, fn = line.strip('\n').split(' ', 2)
                    if int(idx) in indices_to_use:
                        filenames_to_use.append(fn)

            img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
            imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

            _, targets_to_use = list(zip(*imgs_to_use))

            self.imgs = self.samples = imgs_to_use
            self.img_paths = [img_path for img_path, _ in self.imgs]
            self.targets = targets_to_use

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

        self._load_meta(indices_to_use)

    def _load_meta(self, indices_to_use):
        # load part names idx to dict
        with open(os.path.join(self.root_dir, 'parts', 'parts.txt'), 'r') as f:
            self.dataset_parts = {int(line.split(' ')[0]): " ".join(line.split(' ')[1:]) for line in f.read().splitlines()}

        # load train test split
        with open(os.path.join(self.root_dir, 'train_test_split.txt'), 'r') as f:
            self.train_test_split = {int(line.split(' ')[0]): bool(int(line.split(' ')[1])) for line in f.readlines()}

        # load images.txt to get image id and corresponding image path
        with open(os.path.join(self.root_dir, 'images.txt'), 'r') as f:
            self.image_paths = {int(line.split(' ')[0]): line.split(' ')[1].strip('\n') for line in f.readlines()}

        # load part locations, e.g., {img_id: [[x, y, visible],  ...]], ...} note that each image has 15 parts
        self.part_locs = {}
        with open(os.path.join(self.root_dir, 'parts', 'part_locs.txt'), 'r') as f:
            for line in f.read().splitlines():
                img_id, part_id, x, y, visible = line.split(' ')
                self.part_locs.setdefault(int(img_id), []).append([int(float(x)), int(float(y)), int(float(visible))])

        # load bounding boxes
        with open(os.path.join(self.root_dir, 'bounding_boxes.txt'), 'r') as f:
            self.bounding_boxes = {int(line.split(' ')[0]): [int(float(x)) for x in line.split(' ')[1:]] for line in f.readlines()}

        # replace the img_id with image name (no extension)
        self.sample_parts = {}
        self.image_boxes = {}
        for img_id, image_path in self.image_paths.items():
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.sample_parts[image_name] = self.part_locs[img_id]
            self.image_boxes[image_name] = self.bounding_boxes[img_id]

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUB, self).__getitem__(index)

        if self.train:
            sample = self.tri_aug(sample)

        if self.transform_ is not None:
            sample = self.transform_(images=sample, return_tensor='pt')
        if self.target_transform_ is not None:
            target = self.target_transform_(images=target, return_tensor='pt')

        if self.return_path:
            return sample, target, self.img_paths[index]

        return sample, target

    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2:
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict


class CUBBoxed(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset.
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    allowed_keys = ['crop', 'box_dir', 'return_path', 'trivial_aug', 'ops', 'high_res', 'n_pixel', 'return_mask']
    def __init__(self,
                 root: str,
                 transform: transforms.Compose= None,
                 target_transform: transforms.Compose= None,
                 loader= datasets.folder.default_loader,
                 is_valid_file= None,
                 train: bool = True,
                 **kwargs):

        img_root = os.path.join(root, 'images')
        # img_root = os.path.join(root, 'images_synthetic')
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)
        if hasattr(self, 'high_res') and self.high_res:
            img_root = os.path.join(root, 'images_upsampler')

        super(CUBBoxed, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        self.totensor = transforms.ToTensor()

        self.resize = transforms.Resize((self.n_pixel, self.n_pixel), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True, max_size=None) # change this may result in better performance

        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = []
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use
        
        mean = list(self.transform_.transforms)[-1].mean
        self.gray_image = torch.ones((1, 3, self.n_pixel, self.n_pixel), dtype=torch.float32) * torch.tensor(mean)[None, :, None, None]

    def __getitem__(self, index):
        # generate one sample
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_size = torch.tensor(sample.size[::-1]) # (w, h) -> (h, w)
        hight, width = sample_size
        image_id = os.path.splitext(os.path.basename(path))[0]
        img_tensor = self.totensor(sample)
        # repeat samples
        boxes = torch.load(os.path.join(self.box_dir, image_id + '.pth'))["boxes_info"]
        sample = torch.cat([self.resize(img_tensor).unsqueeze(0)] * len(boxes))
        box_labels = torch.ones(len(boxes))
        masks = torch.zeros((len(boxes), self.n_pixel, self.n_pixel), dtype=torch.uint8)
        
        if hasattr(self, 'crop') or hasattr(self, 'return_mask'):
            crop_samples = []
            for idx, (part_name, box) in enumerate(boxes.items()):
                # boxes = {'Part name0': [x0, y0, x1, y1], 'Part name1': [x0, y0, x1, y1], ...}
                if self.crop:
                    if sum(box) == 0:
                        box_image = self.gray_image
                    else:
                        box_image = self.resize(img_tensor[:, box[1]:box[3], box[0]:box[2]].unsqueeze(0))
                    crop_samples.append(box_image)
                    
                if self.return_mask:
                    # scale box from image_size (h, w) to (n_pixel, n_pixel)
                    scale_factor = torch.tensor([self.n_pixel / width, self.n_pixel / hight, self.n_pixel / width, self.n_pixel / hight])
                    box = (torch.tensor(box) * scale_factor).int()
                    masks[idx, box[1]:box[3], box[0]:box[2]] = 1
                
                if sum(box) == 0:
                    box_labels[idx] = 0
            crop_sample = torch.cat(crop_samples)
        if len(crop_samples) == 0:
            crop_samples = torch.zeros_like(sample)


        if self.transform_ is not None:
            sample = self.transform_(sample)
            if hasattr(self, 'crop') and self.crop:
                crop_sample = self.transform_(crop_sample)

        # return (sample, target, path, sample_size) if hasattr(self, 'return_path') and self.return_path else (sample, target)
        if hasattr(self, 'return_path') and self.return_path:
            return sample, target, path, sample_size
        # if hasattr(self, 'return_mask') and self.return_mask:
        #     return sample, target, masks
        return sample, target, masks, crop_sample, path, box_labels

    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2:
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class2idx = adjusted_dict
        self.idx2class = {v: k for k, v in self.class_to_idx.items()}