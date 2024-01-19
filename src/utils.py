import os
import clip
import json
import random
import shutil

from PIL import Image
from torch.nn import functional as F
import torchvision.transforms as transforms

from configs import *


''' CLIP TRANSFORM
0 = {Resize} Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
1 = {CenterCrop} CenterCrop(size=(224, 224))
2 = {function} <function _convert_image_to_rgb at 0x7fd16ab12830>
3 = {ToTensor} ToTensor()
4 = {Normalize} Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
'''

def img_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load_descriptions(dataset_name, classes_to_load=None, prompt_type=None, desc_type="sachit", part_based=False,
                      target_classes: list[int] = None, descriptor_path: str = None, unmute: bool = True):
    templated_descriptions, descriptions_mappings = {}, {}

    # ImageNet and ImageNet-v2 share the same list of descriptions
    dataset_to_load = "imagenet" if dataset_name in ["imagenet-v2", "imagenet-a", "imagenet-c"] else dataset_name

    if descriptor_path is None:
        descriptor_path = f"{PROJECT_ROOT}/data/descriptors/{desc_type}/descriptors_{dataset_to_load}.json"

        if dataset_to_load == "bird_11K":
            descriptor_path = descriptor_path.replace(dataset_to_load, f"{dataset_to_load}")

    if unmute:
        print("Using descriptors from: ", descriptor_path)

    with open(descriptor_path) as input_file:
        descriptions = json.load(input_file)

    if classes_to_load is not None:
        descriptions = {c: descriptions[c] for c in classes_to_load}
    elif dataset_name == "imagenet-a":
        descriptions = {c: descriptions[c] for idx, c in enumerate(descriptions.keys()) if idx in indices_in_1k}

    if prompt_type is not None:
        for idx, (class_name, class_descriptors) in enumerate(descriptions.items()):
            if target_classes is not None and class_name not in target_classes and \
               (class_name.lower() not in target_classes or class_name.lower().replace("-", " ") not in target_classes ):
                continue

            if len(class_descriptors) == 0:
                class_descriptors = ['']

            class_name = wordify(class_name) # replace '_' with ' '
            
            # Sachit's prompt
            if prompt_type == 0:
                templated_descriptors = class_descriptors
            elif prompt_type == 1:
                templated_descriptors = [f"{make_descriptor_sentence(class_name, descriptor, part_based)}" for descriptor in class_descriptors]
            elif prompt_type == 2:
                templated_descriptors = [f"{descriptor} of {class_name}" for descriptor in class_descriptors]
            elif prompt_type == 3:
                templated_descriptors = [f"a photo of {descriptor} of {class_name}" for descriptor in class_descriptors]
            elif prompt_type == 4:
                templated_descriptors = [f"a cropped photo of {descriptor} of {class_name}" for descriptor in class_descriptors]
            elif prompt_type == 5:
                templated_descriptors = [f"a photo of a {make_descriptor_sentence(class_name, descriptor, part_based)}" for descriptor in class_descriptors]
            elif prompt_type == 6:
                templated_descriptors = [f"a cropped photo of a {make_descriptor_sentence(class_name, descriptor, part_based)}" for descriptor in class_descriptors]
            elif prompt_type == 7:
                templated_descriptors = [f"a photo of a {make_descriptor_sentence('', descriptor, part_based)}" for descriptor in class_descriptors]
            # Used for finetuning CLIP
            elif prompt_type == 8:
                template = "a {} {} of {}."
                templated_descriptors = [template.format(descriptor.split(":")[1].strip(), descriptor.split(":")[0].strip(), class_name) for descriptor in class_descriptors]
            elif prompt_type == 9:      # Same as Peijie's prompt X-2
                template = "a {} {}"    # convert '{part}: {features}' to 'a {features} {part}'
                templated_descriptors = [template.format(descriptor.split(":")[1].strip(), descriptor.split(":")[0].strip()) for descriptor in class_descriptors]
            elif prompt_type == 10:     # Same as Peijie's prompt X-3
                template = "{}. {}."    # convert '{part}: {features}' to '{features}. {part}.'
                templated_descriptors = [template.format(descriptor.split(":")[1].strip(), descriptor.split(":")[0].strip()) for descriptor in class_descriptors]
            else:
                raise ValueError("Unknown prompt type")

            templated_descriptions[class_name] = templated_descriptors
            descriptions_mappings[class_name] = {templated_descriptor: descriptor for descriptor, templated_descriptor in zip(class_descriptors, templated_descriptors)}

            # Print an example for checking
            if idx == 0 and unmute:
                print(f"\nExample description for prompt type {prompt_type}: \"{templated_descriptions[class_name][0]}\"\n")

    return templated_descriptions, descriptions_mappings


def wordify(string):
    word = string.replace('_', ' ')
    return word


def make_descriptor_sentence(class_name, descriptor, part_based=False):

    if part_based:
        part, descriptor = descriptor.split(":")
        descriptor = descriptor.strip()

        if class_name != "":
            if descriptor.startswith(('a', 'an', 'used')):
                return f"{class_name}, whose {part} is {descriptor}"
            elif descriptor.startswith(('has', 'often', 'typically', 'may', 'can')):
                return f"{class_name}, whose {part} {descriptor}"
            else:
                return f"{class_name}, whose {part} has {descriptor}"
        else:
            if part == "bird":
                return descriptor

            return f"{part}, which is {descriptor}"

    if descriptor.startswith(('a', 'an', 'used')):
        return f"{class_name}, which is {descriptor}"
    elif descriptor.startswith(('has', 'often', 'typically', 'may', 'can')):
        return f"{class_name}, which {descriptor}"
    else:
        return f"{class_name}, which has {descriptor}"


def check_device_availability(devices: list[int or str] or int or str):
    # check if the devices are in format 'cuda:X' or 'X' or X, where X is an integer
    def _check_device_format(device_name: str or int) -> int:
        if isinstance(device_name, int):
            return device_name
        elif isinstance(device_name, str):
            if device_name.startswith("cuda:"):
                return int(device_name.split(":")[1])
            else:
                return int(device_name)      
    
    # check devices if devices is a list
    if not isinstance(devices, list):
        devices = [devices]
    devices = [_check_device_format(device) for device in devices]
    # check if the devices are available
    
    available_list = list(range(torch.cuda.device_count()))
    # check if the device is available
    for device in devices:
        if device not in available_list:
            raise ValueError(f"Device {device} is not available. Available devices are {available_list}")



