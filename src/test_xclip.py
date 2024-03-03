#%%
import os
import json
import random
import warnings

import fire
import torch
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTModel
from src.owlvit_inference import OwlViTForClassification
from data_loader import BirdSoup

from misc.plot_tools import center_to_corners_format_torch, get_pre_define_colors
from utils import load_descriptions
from configs import BIRD_SOUP_DIR, STANFORDDOGS_DIR, DOG_SOUP_DIR

ORG_PART_ORDER = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eyes', 'legs', 'wings', 'nape', 'tail', 'throat']
COLORS_INT = get_pre_define_colors(12, is_float=False, cmap_set=['Set2', 'tab10'])
PART_FREQUENCY_ORDER = ['beak', 'throat', 'forehead', 'crown', 'wings', 'eyes', 'nape', 'breast', 'tail', 'belly', 'legs', 'back']

DOG_PART_ORDER = ['head', 'ears', 'muzzle', 'body', 'legs', 'tail']
def encode_descs(owlvit_det_processor: callable, model: callable, descs: list[str], device: str, max_batch_size: int = 512):
    total_num_batches = len(descs) // max_batch_size + 1
    with torch.no_grad():
        text_embeds = []
        for batch_idx in range(total_num_batches):
            query_descs = descs[batch_idx*max_batch_size:(batch_idx+1)*max_batch_size]
            query_tokens = owlvit_det_processor(text=query_descs, padding="max_length", truncation=True, return_tensors="pt").to(device)
            query_embeds = model.owlvit.get_text_features(**query_tokens)
            text_embeds.append(query_embeds.cpu().float())
    text_embeds = torch.cat(text_embeds, dim=0)
    return text_embeds

def get_teacher_logits(image_paths: list[str], owlvit_precomputed_dir: str, is_birdsoup: bool = False) -> list[torch.Tensor]:
    teacher_logits = []
    boxes = []
    for image_path in image_paths:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        precomputed_embed_file = f"{owlvit_precomputed_dir}/{image_id}.pth"
        precomputed_result = torch.load(precomputed_embed_file)
        logits = precomputed_result["part_logits_owlvit_base"]
        image_boxes = precomputed_result["boxes_info"] if "boxes_info" in precomputed_result else precomputed_result["part_boxes"]
        if not is_birdsoup:
            logits = torch.sigmoid(logits)
        teacher_logits.append(logits)
        boxes.append(image_boxes)
    return torch.stack(teacher_logits), boxes

def get_box_lists(image_paths: list[str], box_dir: str, part_names: list[str]) -> list[list[int]]:
    box_lists = []
    for image_path in image_paths:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        box_file = f"{box_dir}/{image_id}.pth"
        boxes = torch.load(box_file)["boxes_info"]
        boxes = [boxes[part_name] for part_name in part_names]
        box_lists.append(boxes)
    return box_lists

def get_dataset(dataset_name: str, 
                transform: dict[str: torchvision.transforms.Compose] | torchvision.transforms.Compose, 
                test_file: str,  
                sub_dataset_names: str = None,
                train_file: str = None,
                val_file: str = None,
                split: str = 'test',
                use_meta_dir: bool = False,
                ) -> tuple[Dataset, int, str, str, bool, str]:
    
    if dataset_name == "bird_soup":
        sub_datasets = [sub_dataset.strip() for sub_dataset in sub_dataset_names.split(',')] if sub_dataset_names is not None else None

        if split != 'test':
            dataset = BirdSoup(BIRD_SOUP_DIR, transform=transform, train=True, return_path=False, meta_path=train_file, subset=sub_datasets)
            val_dataset = BirdSoup(BIRD_SOUP_DIR, transform=transform, train=False, return_path=False, meta_path=val_file, subset=sub_datasets)
        else:
            dataset = BirdSoup(BIRD_SOUP_DIR, transform=transform, train=False, return_path=False, meta_path=test_file, subset=sub_datasets)
            val_dataset = None
    
        owlvit_precomputed_dir = None
    
    elif dataset_name == 'stanforddogs':
        if split != 'test':
            dataset = BirdSoup(STANFORDDOGS_DIR, transform=transform, train=True, return_path=False, meta_path=train_file, use_meta_dir=True)
            val_dataset = BirdSoup(STANFORDDOGS_DIR, transform=transform, train=False, return_path=False, meta_path=val_file, use_meta_dir=True)
        else:
            dataset = BirdSoup(STANFORDDOGS_DIR, transform=transform, train=False, return_path=False, meta_path=test_file, use_meta_dir=True)
            val_dataset = None
        
        owlvit_precomputed_dir = None
        
    elif dataset_name == 'dog_soup':
        if split != 'test':
            dataset = BirdSoup(DOG_SOUP_DIR, transform=transform, train=True, return_path=False, meta_path=train_file, use_meta_dir=False)
            val_dataset = BirdSoup(DOG_SOUP_DIR, transform=transform, train=False, return_path=False, meta_path=val_file, use_meta_dir=False)
        else:
            dataset = BirdSoup(DOG_SOUP_DIR, transform=transform, train=False, return_path=False, meta_path=test_file, use_meta_dir=False)
            val_dataset = None
        owlvit_precomputed_dir = "/home/lab/xclip/owlvit_boxes/dogsoup_v1/part_boxes/owlvit-large-patch14_stanforddog-6-parts-dog_update_logits"
        
    class_names = dataset.classes
    n_classes = len(class_names)
    return dataset, n_classes, class_names, owlvit_precomputed_dir


def eval_xclip(model: torch.nn.Module, 
               owlvit_det_processor: callable, 
               owlvit_det_model: callable, 
               query_descs: list[str], 
               device: int, 
               dataset: callable, 
               crop_bird: bool = False, 
               part_names: list[str] = None, 
               out_folder: str = None,
               save_logits: bool = False,
               num_samples: int | None = 32,
               owlvit_precomputed_dir: str = None,
               is_birdsoup: bool = False,
               descriptions: dict = None,
               use_teacher_logits: bool = True,
               save_teacher_boxes: bool = False,
               save_everything: bool = False,
               ):
    

    if out_folder is not None and not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if save_logits:
        pred_list = []
        
    img2tensor = transforms.ToTensor()

    query_embeds = encode_descs(owlvit_det_processor, model, query_descs, device)
    pred = []
    image_cnt = 0
    if num_samples is None:
        num_samples = len(dataset)
    with torch.no_grad():
        
        for batch in tqdm(dataset, desc="Evaluating xclip", total=num_samples):
            
            if is_birdsoup:
                sample, target, image_sizes, image_path, soup_name = batch
            else:
                sample, target, image_path, sample_size = batch 
            if crop_bird:
                # get the inputs for bird detection
                bird_query_tokens = owlvit_det_processor(text=['bird']*len(image_path), padding="max_length", truncation=True, return_tensors="pt")
                sample['pixel_values'] = sample['pixel_values'].squeeze(1)
                bird_inputs = sample | bird_query_tokens
                bird_inputs = bird_inputs.to(device)
                
                # detect boxes of birds
                bird_owl_output = owlvit_det_model(**bird_inputs)
                bird_preds = owlvit_det_processor.post_process_object_detection(bird_owl_output, 0, sample_size.to(device))
                bird_scores = torch.sigmoid(bird_owl_output.logits)
                bird_topk_scores, bird_topk_idxs = torch.topk(bird_scores, k=1, dim=1)
                bird_boxes = torch.stack([bird_pred['boxes'][bird_topk_idx.item()].cpu().long() for bird_pred, bird_topk_idx in zip(bird_preds, bird_topk_idxs)])
                
                # crop the bird from the image
                bird_crops = []
                bird_sizes = []
                new_bird_boxes = []
                for i, image_file in enumerate(image_path):
                    image = Image.open(image_file).convert("RGB")
                    image_tensor = img2tensor(image)
                    box = bird_boxes[i]
                    image_cnt += 1
                    
                    # extend the box by 10% if possible
                    max_height, max_width = sample_size[i]
                    extend_ratio = 0.1
                    box = torch.tensor([
                        max(0, box[0] - (box[2] - box[0]) * extend_ratio),
                        max(0, box[1] - (box[3] - box[1]) * extend_ratio),
                        min(max_width, box[2] + (box[2] - box[0]) * extend_ratio),
                        min(max_height, box[3] + (box[3] - box[1]) * extend_ratio)
                    ])
                    box = box.int().tolist()
                    new_bird_boxes.append(box)
                    
                    # change tensor to pil image
                    bird = image_tensor[:, box[1]:box[3], box[0]:box[2]]
                    bird = image.crop(box)
                    bird_crops.append(bird)
                    # bird_crops.append(transforms.functional.to_pil_image(bird))
                    # collect the box sizes in height and width
                    bird_sizes.append([box[3] - box[1], box[2] - box[0]])
                # bird_crops = torch.stack(bird_crops)
                bird_sizes = torch.tensor(bird_sizes)
                
                # get part logits
                sample = owlvit_det_processor(images=bird_crops, return_tensors='pt').to(device)
                part_query_tokens = owlvit_det_processor(text=part_names * len(image_path), padding="max_length", truncation=True, return_tensors="pt").to(device)
                part_inputs = sample | part_query_tokens
                part_inputs = part_inputs.to(device)
                
                part_owl_output = owlvit_det_model(**part_inputs)
                teacher_logits = [{"logits": logits} for logits in part_owl_output.logits.cpu()]

            else:
                sample['pixel_values'] = sample['pixel_values'].squeeze(1)
                # get teacher logits
                if use_teacher_logits or save_teacher_boxes: 
                    teacher_logits, boxes = get_teacher_logits(image_path, owlvit_precomputed_dir, is_birdsoup)
                    input_ids = None
                    attention_mask = None
                if not use_teacher_logits:
                    teacher_logits = None
                    text_input_parts = owlvit_det_processor(text=part_names*len(target), return_tensors='pt').to(device)
                    input_ids = text_input_parts.input_ids
                    attention_mask = text_input_parts.attention_mask
                    
            text_embeds = query_embeds.unsqueeze(0).to(device)
            text_embeds = torch.repeat_interleave(text_embeds, len(image_path), dim=0)
            # pred_logits, part_logits, output_dict = model(sample.to(device), text_input_parts, text_embeds, teacher_logits)
            pred_logits, image_text_logits, part_logits, pred_boxes_selected, loss_dict = model(pixel_values=sample.pixel_values.to(device), 
                                                                                                input_ids=input_ids, 
                                                                                                attention_mask=attention_mask,
                                                                                                text_embeds=text_embeds, 
                                                                                                targets={'logits': teacher_logits})
            
            pred_class = torch.argmax(pred_logits, dim=-1).cpu()
            # get the top 2 prediction
            # top2_class = torch.topk(pred_logits, k=2, dim=-1)[1].cpu()
            is_correct = pred_class == target
            pred.append(is_correct)
            # print current accuracy like:  Current Top1: 0.00
            print(f"Current Top1: {torch.cat(pred).float().mean().item()}")
            
            softmax_scores = torch.softmax(pred_logits, dim=-1, ).cpu()
            # softmax_scores = torch.nn.functional.gumbel_softmax(pred_logits, dim=-1, tau=0.5).cpu()
            softmax_score_top1 = torch.topk(softmax_scores, k=1, dim=-1)[0].squeeze(-1)
            
            if not use_teacher_logits and not save_teacher_boxes:
                image_boxes = pred_boxes_selected
                # post process the boxes, convert float (center_x, center_y, width, height) to int (x1, y1, x2, y2)
                image_boxes = center_to_corners_format_torch(image_boxes)
                # convert float (0, 1) to actual coordinates
                pred_boxes = []
                for boxes, image_size in zip(image_boxes, image_sizes):
                    max_height, max_width = image_size
                    boxes = boxes * torch.tensor([max_width, max_height, max_width, max_height])
                    # remove negatives in the box
                    boxes = torch.clamp(boxes, min=0).int().tolist()
                    part_boxes = dict(zip(ORG_PART_ORDER, boxes))
                    pred_boxes.append(part_boxes)


            # save logits for testing
            if save_logits:
                for inbatch_idx, (image_file, part_logit, prediction, pred_class_id, gt_id, softmax_top1) in enumerate(zip(image_path, part_logits, is_correct, pred_class, target, softmax_score_top1)):
                    bs_file_name = f'{os.path.basename(image_file)}'
                    org_file_name = dataset.dataset.meta_df[dataset.dataset.meta_df['new_image_name'] == bs_file_name]['abs_path'].values[0]
                    org_file_name = os.path.basename(org_file_name)
                    part_scores = part_logit.cpu()[pred_class_id].tolist()
                    part_scores = dict(zip(part_names, part_scores))
                    gt_scores = part_logit.cpu()[gt_id].tolist()
                    gt_scores = dict(zip(part_names, gt_scores))
                    gt_name = dataset.dataset.idx2class[gt_id.item()]
                    pred_name = dataset.dataset.idx2class[pred_class_id.item()]
                    # top2_score = sum(part_logit.cpu()[top2_class[inbatch_idx][1]].tolist())
                    top2_score = None
                    # torch.save(part_logit, os.path.join(out_folder, file_name))
                    output_dict = {'file_name': org_file_name, 'bs_file_name': bs_file_name, 'prediction': prediction.item(), 'ground_truth': gt_name, 'pred': pred_name, "final_score": sum(part_scores.values()), "top2_score": top2_score, "softmax_score": softmax_top1.item()} | {'pred_scores': part_scores} | {'gt_scores': gt_scores} 
                    if not use_teacher_logits and not save_teacher_boxes:
                        output_dict = output_dict | {'pred_boxes': pred_boxes[inbatch_idx]} 
                    else:
                        output_dict = output_dict | {'pred_boxes': boxes[inbatch_idx]}
                    if descriptions is not None:
                        desc = descriptions[pred_name]
                        desc = {"descriptions":{k: v for k, v in zip(ORG_PART_ORDER, desc)}}
                        output_dict = output_dict | desc
                        
                    if save_everything:
                        all_data = {}
                        for i in range(model.cls_head.num_classes):
                            class_name = dataset.dataset.idx2class[i]
                            softmaxed_score = softmax_scores[inbatch_idx][i].item()
                            pred_scores = part_logits[inbatch_idx][i].tolist()
                            pred_scores = dict(zip(part_names, pred_scores))
                            class_descriptions = descriptions[class_name]
                            all_data[class_name] = {"softmax_score": softmaxed_score, "pred_scores": pred_scores, "descriptions": {k: v for k, v in zip(ORG_PART_ORDER, class_descriptions)}}
                        output_dict = output_dict | {"all_data": all_data}
                        
                    pred_list.append(output_dict)
                    
                    image_cnt += 1
                     
    if save_logits:
        # pred_df = pd.DataFrame(pred_list)
        # pred_df.to_csv(os.path.join(out_folder, 'pred.csv'), index=False)
        json.dump(pred_list, open(os.path.join(out_folder, 'xclip_pred.json'), 'w'), indent=4)
    
    return torch.cat(pred).float().mean().item()

def get_model_path(model_name: str) -> str:
    # if model_name ends in .pt or .pth, it is the path to the model, otherwise match names
    if model_name.endswith(".pt") or model_name.endswith(".pth"):
        return model_name
    
    match model_name:
        case "level_1_pretrain":
            checkpoint = "/home/thang/Projects/xclip_bk/results/bird_soup/level_1/training_contrastive/chatgpt-owlvit-base-patch32/09_11_2023-17:18:34_lr_2e-05_64ep_prompt0_v1_from_ft_01120/wandb/run-20230911_171835-k6pasqbn/files/e0007.pt"
        case "level1_cub_finetuned":
            checkpoint = '/home/thang/Projects/xclip_bk/results/bird_soup/level_1/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/09_13_2023-16:21:28_lr_2e-05_32ep_prompt0_all_components/wandb/run-20230913_162129-5os0ieiv/files/e0029.pt'
        case "level_3_cub":
            checkpoint = "/home/thang/Projects/xclip_bk/results/bird_soup/level_3/training_contrastive/chatgpt-owlvit-base-patch32/08_14_2023-21:39:36_lr_0.0002_100ep_prompt0/wandb/run-20230814_213937-6ggnspmo/files/e0020.pt"
        case "level_3_cub_clore":
            checkpoint = '/home/thang/Projects/xclip_bk/results/bird_soup/level_3/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/09_15_2023-13:52:22_lr_2e-05_32ep_prompt0_all_components_lv3_clore/wandb/run-20230915_135225-332xxpov/files/e0004.pt'
        case "level3_cub_SCS":
            checkpoint = "/home/thang/Projects/xclip_bk/results/bird_soup/level_3/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/09_15_2023-13:52:27_lr_2e-05_32ep_prompt0_all_components_lv3_cub_easy/wandb/run-20230915_135228-np57u461/files/e0004.pt"
        case "level3_cub_SCE":
            checkpoint = "/home/thang/Projects/xclip/results/bird_soup/level_3/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/09_15_2023-13:52:32_lr_2e-05_32ep_prompt0_all_components_lv3_cub_hard/wandb/run-20230915_135232-j9de2cqr/files/e0004.pt"
        case "level3_nabird_SCS":
            checkpoint = "/home/thang/Projects/xclip/results/bird_soup/level_3/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/09_15_2023-13:55:22_lr_2e-05_32ep_prompt0_all_components_lv3_nabirds_easy/wandb/run-20230915_135524-kl6cauv1/files/e0004.pt"
        case "level3_nabird_SCE":
            checkpoint = "/home/thang/Projects/xclip/results/bird_soup/level_3/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/09_15_2023-13:55:31_lr_2e-05_32ep_prompt0_all_component_lv3_nabirds_hard/wandb/run-20230915_135532-zglocpde/files/e0004.pt"    
        case "dogsoup_pretrain": # 87.37%
            checkpoint = "/home/thang/Projects/Adobe/results/dog_soup/level_1/training_contrastive/chatgpt-owlvit-base-patch32/02_23_2024-09:58:29_lr_5e-05_64ep_prompt0_lv1_from_scratch_dog_soup/wandb/run-20240223_095830-uriz9t04/files/e0009.pt"
        case "dogsoup_sd_finetune": # 92.20%
            checkpoint = "/home/thang/Projects/Adobe/results/dog_soup/level_1/finetune_classification/vision_encoder_mlp/chatgpt-owlvit-base-patch32/02_24_2024-07:59:25_lr_1e-05_64ep_prompt0_vision_encoder_mlp_SD_01111_from_pt/wandb/run-20240224_075926-44tqyoho/files/e0008.pt"
        
        case _:            
            raise ValueError(f"Model {model_name} not supported")
    return checkpoint

def get_test_df(test_df_name: str) -> str:
    # if test_df ends in .h5, it is the path to the test_df, otherwise match names
    if test_df_name is not None and test_df_name.endswith(".h5"):
        return test_df_name
    
    match test_df_name:
        case "cub_test":
            test_df = '/home/lab/datasets/bird_soup/metadata/level_1_exclude_cub_nabirds_inat/test_cub.h5'
        case "inaturalist_test":
            test_df = '/home/lab/datasets/bird_soup/metadata/level_1_exclude_cub_nabirds_inat/test_inat_a200_unbalanced.h5'
        case "cub_human_study":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_cub/human-study/test.h5"
        case "cub_scs_easy":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_cub/super_category_split/easy_test.h5"
        case "cub_scs_hard":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_cub/super_category_split/hard_test.h5"
        case "cub_clore_seen":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_cub/clore_split/test_seen.h5"
        case "cub_clore_unseen":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_cub/clore_split/test_unseen.h5"
        case "nabird":
            test_df = '/home/lab/datasets/bird_soup/metadata/level_1_exclude_cub_nabirds_inat/test_nabirds.h5'
        case "nabird_scs_easy":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_nabirds/super_category_split/easy_test.h5"
        case "nabird_scs_hard":
            test_df = "/home/lab/datasets/bird_soup/metadata/level_3_exclude_nabirds/super_category_split/hard_test.h5"
        case "sd_test":
            test_df = "/home/lab/datasets/dogsoup/sd_test.h5"
        case _:
            raise ValueError(f"Test df {test_df_name} not supported, please privde *.h5 file instead.")
    return test_df

def eval_warpper(device: str = 'cuda:7',
                 out_folder: str = 'results/logits/temp',
                 subfolder: str = None,
                 save_logits: bool = False,
                 batch_size: int = 32,
                 num_samples: int | None = None,
                 dataset_name: str = 'birdsoup_cub',
                 model_path: str = None,
                 test_df: str = 'cub_test',
                 birdsoup_mean_std: bool = False,
                #  revised_desc: bool = False, # only valid for 'cub_human_study' dataset
                 use_teacher_logits: bool = False, # set false for step2 or step3 models
                 save_teacher_boxes: bool = False, # save the teacher boxes disregrad the use of teacher logits
                 descriptor_path: str = None, # will override the descriptor path in get_dataset if provided
                 n_parts: int = 12, # number of parts to use, only for ablation study. Keep it 12 for other cases
                 save_everything: bool = False, # save everything for ploting
                #  use_meta_dir: bool = False, # use meta_dir instead of meta_df
                 ):
    
    if subfolder is not None:
        out_folder = os.path.join(out_folder, subfolder)

    model_path = get_model_path(model_path)

    test_df = get_test_df(test_df)
    
    owlvit_det_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owlvit_det_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    preprocess = lambda x: owlvit_det_processor(images=x, return_tensors='pt')
    if birdsoup_mean_std:
        # BirdSoup mean std
        mean = [0.48168647, 0.49244233, 0.42851609]
        std = [0.18656386, 0.18614962, 0.19659419]
        owlvit_det_processor.image_processor.image_mean = mean
        owlvit_det_processor.image_processor.image_std = std

    dataset, n_classes, class_names, owlvit_precomputed_dir = get_dataset(dataset_name, 
                                                                        transform=preprocess, 
                                                                        test_file=test_df, 
                                                                        use_meta_dir=False)

    # load finetuned owl-vit model
    weight_dict = {"loss_ce": 0, "loss_bbox": 0, "loss_giou": 0,
                    "loss_sym_box_label": 0, "loss_xclip": 0}
    n_parts = 6 if "dog" in dataset_name else n_parts
    model = OwlViTForClassification(owlvit_det_model=owlvit_det_model, num_classes=n_classes, device=device, weight_dict=weight_dict, logits_from_teacher=use_teacher_logits, num_parts=n_parts)
    if model_path is not None:
        ckpt = torch.load(model_path, map_location='cpu')
        if any(k.startswith('module.') for k in ckpt.keys()):
            ## temporary fix for loading model trained on multiple gpus
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt.items():
                # Remove the 'module.' prefix if it exists
                name = k.replace('module.', '') if 'module.' in k else k
                new_state_dict[name] = v
            ckpt = new_state_dict

        # add dummpy parameters for position_ids if not exist (due to the change in defalut setting of huggingface transformers)
        if "owlvit.text_model.embeddings.position_ids" not in ckpt:
            ckpt["owlvit.text_model.embeddings.position_ids"] = torch.arange(0, 16).unsqueeze(0)
        if "owlvit.vision_model.embeddings.position_ids" not in ckpt:
            ckpt["owlvit.vision_model.embeddings.position_ids"] = torch.arange(0, 577).unsqueeze(0)
        model.load_state_dict(ckpt, strict=True) 
    model.to(device)
    model.eval()

    owlvit_det_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device) # load the model again for debugging

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    descs_dict, class_mapping = load_descriptions(dataset_name=dataset_name, prompt_type=0, target_classes=class_names, descriptor_path=descriptor_path, unmute=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    part_names = DOG_PART_ORDER if "dog" in dataset_name else ORG_PART_ORDER
    descs = []
    for class_name in class_names:
        descs.extend(descs_dict[class_name])
        
    # sort the descs_dict based on class_name order
    descriptions = {class_name: descs_dict[class_name] for class_name in class_names}

    acc = eval_xclip(model=model,
                    owlvit_det_processor=owlvit_det_processor,
                    owlvit_det_model=owlvit_det_model,
                    query_descs=descs,
                    device=device,
                    dataset=dataloader,
                    part_names=part_names,
                    out_folder=out_folder,
                    save_logits=save_logits,
                    num_samples=num_samples,
                    owlvit_precomputed_dir=owlvit_precomputed_dir,
                    is_birdsoup=True,
                    descriptions=descriptions,
                    use_teacher_logits=use_teacher_logits,
                    save_teacher_boxes=save_teacher_boxes,
                    save_everything=save_everything,
                    )
    print(f'Accuracy: {acc}')

#%%
if __name__ == '__main__':
    fire.Fire({"eval": eval_warpper})