import json
from nltk.corpus import wordnet as wn

from utils.sachit_descriptors import get_sachit_hparams, load_gpt_descriptions

def get_prompt_list(file_path: str) -> list[list[str]]:
    
    with open(file_path, 'r') as f:
        prompt_list = json.load(f)
        
    return prompt_list

class GetPromptList(object):
    _SUPPORTED_SOURCE = {'Sachit-descriptors',}
    def __init__(self, file_path: str, name2idx: dict[str: int] = None, class_names: list[str] = None) -> None:
        self.class_names = class_names
        self.file_path = file_path
        self.desc = get_prompt_list(file_path)
        if isinstance(self.desc, dict):
            self.__get_parts()
        if name2idx is not None:
            self.name2idx = name2idx
        elif class_names is not None:
            self.name2idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        else:
            self.name2idx = {cls_name: idx for idx, cls_name in enumerate(self.desc.keys())} if isinstance(self.desc, dict) else None
        
    
    def __get_sachit_desc(self, file_path: str): 
        params = get_sachit_hparams(file_path)
        return load_gpt_descriptions(params)
    
    def __get_parts(self, ):
        # get part names from one of the descriptions
        self.part_names = [d.split(":")[0].strip() for d in self.desc[list(self.desc.keys())[0]]]
    
    
    @staticmethod
    def replace_class_names(self, descs: dict, target_class: list[str], new_classes: list[str]):
        new_descs = []
        for desc, cls_name, new_name in zip(descs, target_class, new_classes):
            temp = [d.replace(cls_name, new_name) for d in desc]
            new_descs.extend(temp)
        return new_descs
    
    def __call__(self, source: str, pad: bool = False, max_len: int = 15, pad_text: str = "", target_classes: list[int] = None, pad_neg_index: bool = True):
        """
        This function will return a list of prompts based on the source (format) and file_path provided. 
        If name2idx is provided, the prompts will be mapped based on the provied class indexes. Otherwise,
        the prompts will be mapped based on the order of class name in the file.
        Note: this function is will apply trucation when padding is True to make sure to have fixed length prompts.

        Args:
            source (str): The sorce (format) of the prompts. Supported sources are: {self._SUPPORTED_SOURCE}
            file_path (str): The file that contains the original prompts.
            pad (bool, optional): Whether to pad the prompts to the same length. Defaults to False.
            max_len (int, optional): The maximum length of the prompts. Defaults to 15.
            pad_text (str, optional): The text to pad the prompts. Defaults to "Padding".
            target_classes (list[int], optional): A list of class indexes to include in the prompts. Defaults to None (include all classes).
        Returns:
            prompts (list[str]): A list of engineered prompts.
            class_idxs (list[int]): A list of class indexes for each prompt.
            class_mapping (dict[int: str]): A mapping of class indexes to class names.
        """
        org_desc_mapper = None
        match source:
            case 'Sachit-descriptors':
                desc, org_dict = self.__get_sachit_desc(self.file_path)

            case 'Sachit-no-template':
                desc = self.desc
                
            case 'Sachit-CLIP-template-5':
                desc, org_dict = self.__get_sachit_desc(self.file_path)
                desc = {k: [f'a photo of a {d}.' for d in v] for k, v in desc.items()}
            case 'cub-12-parts':
                return self.desc, None, None, None
            case 'chatgpt-no-template':
                desc = self.desc
            case 'chatgpt-template-0':
                # convert 'part: features' to 'a features part'
                template = 'a {} {}.'
                desc = {k: [template.format(d.split(":")[1].strip(), d.split(":")[0].strip()) for d in v] for k, v in self.desc.items()}
            case 'chatgpt-template-8':
                # convert '{part}: {features}' to 'a {features} {part} of {class_name}'
                template = 'a {} {} of {}.'
                desc = {k: [template.format(d.split(":")[1].strip(), d.split(":")[0].strip(), k) for d in v] for k, v in self.desc.items()}
            case 'chatgpt-template-5':
                # convert '{part}: {features}' to 'a photo of {class_name}, which is/has/etc {descriptor}
                desc, org_dict = self.__get_sachit_desc(self.file_path)
                template = 'a photo of a {}'
                desc = {k: [template.format(d) for d in v] for k, v in desc.items()}
            case 'chatgpt-template-x':
                # convert '{part}: {features}' to '{features}. {part}. {class_name}'
                desc = {k: [f'{d.split(":")[1].strip()}. {d.split(":")[0].strip()}. {k}' for d in v] for k, v in self.desc.items()}
            case 'chatgpt-template-x-2': # no class name
                # convert '{part}: {features}' to 'a {features} {part}'
                desc = {k: [f'a {d.split(":")[1].strip()} {d.split(":")[0].strip()}' for d in v] for k, v in self.desc.items()}
            case 'chatgpt-template-x-3': # no class name
                # convert '{part}: {features}' to '{features}. {part}.'
                desc = {k: [f'{d.split(":")[1].strip()}. {d.split(":")[0].strip()}.' for d in v] for k, v in self.desc.items()}
            case 'chatgpt-template-x-4': 
                # convert '{part}: {features}' to 'a {part} of {class_name}: {features}'
                desc = {k: [f'a {d.split(":")[0].strip()} of {k}: {d.split(":")[1].strip()}' for d in v] for k, v in self.desc.items()}
            
            case _:
                raise ValueError(f"Source {source} is not supported. Check {self._SUPPORTED_SOURCE}")
        
        # get the subset of descriptrions that match the target classes
        if len(self.name2idx) < len(desc):
            desc = {k: desc[k] for k in self.name2idx}     
            
        prompts, class_idxs, class_list = [], [], []
        class_mapping = {v: k for k, v in self.name2idx.items()}
        for class_name, class_idx in self.name2idx.items():
            descriptions = desc[class_name]
            if target_classes is not None and class_idx not in target_classes:
                continue
            if pad:
                pad_id = -1 if pad_neg_index else class_idx
                ids = [class_idx] * len(descriptions) + [pad_id] * (max_len - len(descriptions)) if len(descriptions) < max_len else [class_idx] * max_len
                if len(descriptions) < max_len:
                    descriptions.extend([pad_text] * (max_len - len(descriptions)))
                else:
                    descriptions = descriptions[:max_len]
            else:
                ids = [class_idx] * len(descriptions)
            prompts.extend(descriptions)
            class_idxs.extend(ids)
            class_list.append(class_name)
        
            if org_desc_mapper is not None:
                org_desc_mapper = {des: org_dict[class_name][des] for des in descriptions}

                
        return prompts, class_idxs, class_mapping, org_desc_mapper, class_list
    
    
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


# deprecated
def get_hierarchy_class_name(wnids: list[str], k: int = 0) -> list[str]:
    # k (int): the level of the hierarchy to return. 0 is the root class, 1 is the parent node from the WordNet hierarchy, etc.
    new_classes = []
    for wnid in wnids:
        prefix, id = wnid[0], int(wnid[1:])
        node = wn.synset_from_pos_and_offset(prefix, id)
        for i in range(k):
            if parent_node := node.hypernyms():
                node = parent_node[-1] # take the nearest parent node
            else:
                print(f"Failed to find {i+1}-th parent class for {wnid}, returing {i}-th parent class instead.")
                break
        new_classes.append(node.name().split('.')[0].replace('_', ' '))
    
    return new_classes

def get_parent_nodes(wnids: list[str],  n: int = None, return_node: bool = False,) -> dict[str: list[str]]:
    parent_nodes = {}
    max_len = 0
    for wnid in wnids:
        prefix, id = wnid[0], int(wnid[1:])
        node = wn.synset_from_pos_and_offset(prefix, id)
        parent_nodes[wnid] = [node.name().split('.')[0]]
        # get up to n-th parent node if n is not None
        while parent_node := node.hypernyms():
            node = parent_node[-1]
            if return_node:
                parent_nodes[wnid].append(node)
            else:
                parent_nodes[wnid].append(node.name().split('.')[0].replace('_', ' '))
            
            if n is not None and len(parent_nodes[wnid]) >= n:
                break
        max_len = max(max_len, len(parent_nodes[wnid]))
    return parent_nodes, max_len

