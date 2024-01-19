import argparse
import gc
import json
import math
import os.path
from datetime import datetime

import pytz
import wandb

import numpy as np
import spacy
import torchvision
import torchmetrics
from sklearn.model_selection import train_test_split
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch import nn
from tqdm import tqdm
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import CUB, NABirdsDataset, BirdSoup
import torch.multiprocessing as mp

from functools import reduce
import operator
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from transformers.image_transforms import corners_to_center_format
from transformers.models.detr.modeling_detr import generalized_box_iou, center_to_corners_format, box_iou

from src.owlvit_cls import OwlViTForClassification

from utils import *

nlp_spacy = spacy.load("en_core_web_sm")

# add python path if not exist:
# export PYTHONPATH=$(pwd):$(pwd)/src


def add_extra_negatives(runtime: str, description_embeds: torch.Tensor, all_cls_ids: set, num_negatives: int, targets_cls: torch.Tensor, verbose: bool = False):
    unique_class_ids = set(targets_cls.tolist())
    if (num_extra_negatives := num_negatives - len(unique_class_ids)) > 0:
        # all_cls_ids = set(range(len(templated_descriptions)))
        all_negatives = all_cls_ids - unique_class_ids
        all_negatives = list(all_negatives)
        unique_class_ids = list(unique_class_ids)
        if verbose:
            print(f"Adding {num_extra_negatives} extra negatives to the batch for {runtime}")
        unique_class_ids += random.sample(all_negatives, num_extra_negatives)
    else:
        unique_class_ids = list(unique_class_ids)
    
    
    selected_text_embeds = description_embeds.view(-1, len(all_parts), description_embeds.shape[-1])[unique_class_ids]
    text_desc_embeds = selected_text_embeds.view(-1, description_embeds.shape[-1])   #.to(device)
    
    # Update targets when the order of text_embeds is changed (reindexing the target classes)
    class_ids2target_cls = dict(zip(unique_class_ids, range(len(unique_class_ids))))
    reindexed_targets_cls = torch.tensor([class_ids2target_cls[class_id] for class_id in targets_cls.tolist()]).to(device)
    
    return text_desc_embeds, reindexed_targets_cls

def get_timestamp():
    local_tz = pytz.timezone("America/Chicago")
    return datetime.now(tz=local_tz).strftime("%Y%m%d_%H%M%S")


def seed_everything(random_state: int):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    if random_state == 0:  # slower, more reproducible
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    else:  # faster, less reproducible
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False


def check_device_availability(devices: list[int]):
    # get all available devices
    available_devices = torch.cuda.device_count()
    available_list = list(range(available_devices))
    # check if the device is available
    for device in devices:
        if int(device) not in available_list:
            raise ValueError(f"Device {device} is not available. Available devices are {available_list}")


# get a train, val, and test split
def get_data_split(dataset: Dataset, labels: list = None, eval_size: float = 0.2, random_state: int = 42):
    if labels is None:
        try:
            labels = dataset.labels
        except:
            raise ValueError("Labels are not provided and the dataset does not have a labels attribute")
    if eval_size == 0:
        train_split = dataset
        val_split = None
    else:
        train_indices, evaluate_indices, _, _ = train_test_split(range(len(dataset)), labels, stratify=labels, test_size=eval_size, random_state=random_state)
        train_split = Subset(dataset, train_indices)
        val_split = Subset(dataset, evaluate_indices)

    return train_split, val_split


def load_training_dataset(dataset_name: str, sub_dataset_names: str, eval_size: float, transform: dict[str: torchvision.transforms.Compose] or torchvision.transforms.Compose, random_state: int = 42, split: str = 'train', zeroshot_split: bool = False):
    """_summary_

    Args:
        dataset_name (str): name of dataset for training
        eval_size (float): proportion of validation set extracted from the train set. Default to 0.2
        transform (dict): dict[str: torchvision.transforms.Compose], the key is the name of the transform, the value is the transform module
        random_state (int, optional): random seed for spliting train and val when eval_size is not 0. Defaults to 42.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if dataset_name == 'cub':
        if split != 'test':
            dataset_org = CUB(CUB_DIR, transform=transform, train=True, return_path=True, zeroshot_split=zeroshot_split)
            dataset, val_dataset = get_data_split(dataset_org, labels=dataset_org.targets, eval_size=eval_size, random_state=random_state)
            val_dataset.transform = transform
        else:
            dataset = CUB(CUB_DIR, transform=transform, train=False, return_path=True, zeroshot_split=zeroshot_split)
            val_dataset = None

    elif dataset_name == "nabirds":
        if split != 'test':
            dataset_org = NABirdsDataset(NABIRDS_DIR, transform=transform, train=True, return_path=True, zeroshot_split=zeroshot_split)
            dataset, val_dataset = get_data_split(dataset_org, labels=dataset_org.targets, eval_size=eval_size, random_state=random_state)
            val_dataset.transform = transform
        else:
            dataset = NABirdsDataset(NABIRDS_DIR, transform=transform, train=False, return_path=True, zeroshot_split=zeroshot_split)
            val_dataset = None

    elif dataset_name == "bird_soup":
        sub_datasets = [sub_dataset.strip() for sub_dataset in sub_dataset_names.split(',')] if sub_dataset_names is not None else None

        if split != 'test':
            dataset = BirdSoup(BIRD_SOUP_DIR, transform=transform, train=True, return_path=True, meta_path=args.train_file, subset=sub_datasets)
            val_dataset = BirdSoup(BIRD_SOUP_DIR, transform=transform, train=False, return_path=True, meta_path=args.val_file, subset=sub_datasets)
        else:
            dataset = BirdSoup(BIRD_SOUP_DIR, transform=transform, train=False, return_path=True, meta_path=args.test_file, subset=sub_datasets)
            val_dataset = None
    
    elif dataset_name == 'stanforddogs':
        if split != 'test':
            dataset = BirdSoup(STANFORDDOGS_DIR, transform=transform, train=True, return_path=True, meta_path=args.train_file, use_meta_dir=True)
            val_dataset = BirdSoup(STANFORDDOGS_DIR, transform=transform, train=False, return_path=True, meta_path=args.val_file, use_meta_dir=True)
        else:
            dataset = BirdSoup(STANFORDDOGS_DIR, transform=transform, train=False, return_path=True, meta_path=args.test_file, use_meta_dir=True)
            val_dataset = None

    return dataset, val_dataset


def visualize_bbox(image_path, gt_bboxes, 
                   base_bboxes, base_bbox_losses, base_giou_losses, base_iou_scores, 
                   pred_bboxes, bbox_losses, giou_losses, iou_scores, store_path=None):
    """
    Draw bounding boxes on the image
    :param image_path: path to image
    :param gt_bboxes: tensor of ground truth bounding boxes. shape (12, 4)
    :param pred_bboxes: tensor of predicted bounding boxes. shape (12, 4)
    :returns: Image object with bounding boxes drawn on it
    """

    def completely_overlapped(box1, box2):
        """Check if box1 is completely overlapped by box2."""
        return box2[0] <= box1[0] and box2[1] <= box1[1] and box2[2] >= box1[2] and box2[3] >= box1[3]

    # Open images
    image_gt = Image.open(image_path)
    draw_gt = ImageDraw.Draw(image_gt)

    image_base = Image.open(image_path)
    draw_base = ImageDraw.Draw(image_base)

    image_pred = Image.open(image_path)
    draw_pred = ImageDraw.Draw(image_pred)

    # Define a list of 12 colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),          # RGB colors: red, green, blue
        (255, 255, 0), (0, 255, 255), (255, 0, 255),    # RGB colors: yellow, cyan, magenta
        (128, 0, 0), (0, 128, 0), (0, 0, 128),          # Darker versions of above colors
        (128, 128, 0), (0, 128, 128), (128, 0, 128)     # Darker versions of above colors
    ]

    part_names = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eyes', 'legs', 'wings', 'nape', 'tail', 'throat']

    font = ImageFont.load_default()
    base_legend_text = []
    legend_text = []

    # Detect overlap between boxes and, if one is completely overlapped, add margins
    for i in range(len(pred_bboxes)):
        for j in range(i + 1, len(pred_bboxes)):
            random_margin = np.random.uniform(0, 5, size=4)

            if completely_overlapped(pred_bboxes[i], pred_bboxes[j]):
                pred_bboxes[i] += random_margin

            if completely_overlapped(base_bboxes[i], base_bboxes[j]):
                base_bboxes[i] += random_margin

            if completely_overlapped(gt_bboxes[i], gt_bboxes[j]):
                gt_bboxes[i] += random_margin

    # Draw bounding boxes. Each box will be drawn with a different color.
    for box, color, part_name in zip(gt_bboxes, colors, part_names):
        draw_gt.rectangle(box.tolist(), outline=color, width=2)
        # draw_gt.text((box[0], box[1]), part_name, fill=color)

    for box, color, part_name, loss_bbox, loss_giou, iou_score in zip(base_bboxes, colors, part_names, base_bbox_losses, base_giou_losses, base_iou_scores):
        draw_base.rectangle(box.tolist(), outline=color, width=2)
        base_legend_text.append((part_name + "|" + str(round(loss_bbox.item(), 2)) + "|" + str(round(loss_giou.item(), 2)) + "|" + str(round(iou_score.item(), 2)), color))

    for box, color, part_name, loss_bbox, loss_giou, iou_score in zip(pred_bboxes, colors, part_names, bbox_losses, giou_losses, iou_scores):
        draw_pred.rectangle(box.tolist(), outline=color, width=2)
        legend_text.append((part_name + "|" + str(round(loss_bbox.item(), 2)) + "|" + str(round(loss_giou.item(), 2)) + "|" + str(round(iou_score.item(), 2)), color))

    base_legend_text.append((f"Average|{round(base_bbox_losses.mean().item(), 2)}|{round(base_giou_losses.mean().item() ,2)}|{round(base_iou_scores.mean().item() ,2)}", (255, 128, 0)))
    legend_text.append((f"Average|{round(bbox_losses.mean().item(), 2)}|{round(giou_losses.mean().item() ,2)}|{round(iou_scores.mean().item() ,2)}", (255, 128, 0)))

    # Draw legend texts
    for i, (text, color) in enumerate(base_legend_text):
        draw_base.text((0, i*10), text, fill=color, font=font)

    for i, (text, color) in enumerate(legend_text):
        draw_pred.text((0, i*10), text, fill=color, font=font)

    if store_path is not None:
        # Concatenate horizontally
        image = Image.new('RGB', (image_gt.width + image_base.width + image_pred.width, image_gt.height))
        image.paste(image_gt, (0, 0))
        image.paste(image_base, (image_gt.width, 0))
        image.paste(image_pred, (image_gt.width + image_base.width, 0))

        # Save the output image
        image.save(store_path + "/" + image_path.split("/")[-1].replace(".jpg", ".png"))

def reduce_losses(loss: torch.Tensor, batch_size: int):
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= (dist.get_world_size() * batch_size)  # Average the loss values
    return loss

def forward_inputs(model, images, text_inputs_parts, text_embeds, targets, weight_dict, visualize_batch_count=0, store_path=None):
    '''
        image_embeds.shape = {Size} torch.Size([10, 60, 60, 1024])
        text_embeds.shape = {Size} torch.Size([10, 2400, 768])
        owlvit_logits.shape = {Size} torch.Size([10, 3600, 12])
    '''
    #3160MB
    # remove unused information from targets
    # batch all inputs to support DP.
    class_labels = torch.stack([t["class_labels"] for t in targets], dim=0)
    logits = torch.stack([t["logits"] for t in targets], dim=0)
    target_cls = torch.stack([t["targets_cls"] for t in targets], dim=0)
    boxes = torch.stack([torch.tensor(t["boxes"]) for t in targets], dim=0)
    batched_targets = {'class_labels': class_labels, 'logits': logits, 'targets_cls': target_cls, 'boxes': boxes}

    pixel_values = images['pixel_values']
    attention_mask = text_inputs_parts['attention_mask']
    input_ids = text_inputs_parts['input_ids']
    if isinstance(model, torch.nn.DataParallel):
        desc_embeds = text_embeds.repeat(pixel_values.shape[0], 1, 1)
    else:
        desc_embeds = text_embeds.repeat(pixel_values.shape[0], 1)

    pred_logits, image_text_logits, pred_boxes, loss_dict = model(pixel_values, attention_mask, input_ids, desc_embeds, batched_targets)

    # compute symmetric cross entropy loss (take out from the forward such that we can use DP to "increase batch size")
    if weight_dict['loss_xclip'] > 0:
        if hasattr(model, "module"):
            xclip_loss = model.module.compute_sce_loss(pred_logits, image_text_logits, target_cls)
        else:
            xclip_loss = model.compute_sce_loss(pred_logits, image_text_logits, target_cls)
        loss_dict['loss_xclip'] = xclip_loss

    # Compute total loss, as a weighted sum of the various losses (22.13GB)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    target_boxes = torch.stack([torch.tensor(t["boxes"]) for t in targets], dim=0)
    base_boxes = torch.stack([torch.tensor(t["boxes_base"]) for t in targets], dim=0)
    pred_boxes = pred_boxes.detach().cpu()

    # giou_scores = torch.diag(generalized_box_iou(pred_boxes.view(-1, 4), target_boxes.view(-1, 4))).view(-1, 12)
    iou_scores = torch.diag(box_iou(pred_boxes.view(-1, 4), target_boxes.view(-1, 4))[0]).view(-1, 12)
    base_iou_scores = torch.diag(box_iou(base_boxes.view(-1, 4), target_boxes.view(-1, 4))[0]).view(-1, 12)

    # loss_dict["giou_score"] = giou_scores.mean()
    loss_dict["iou_score"] = iou_scores.mean()
    loss_dict["base_iou_scores"] = base_iou_scores.mean()

    if visualize_batch_count > 0:
        image_paths = [t["image_path"] for t in targets]
        image_sizes = [t["image_size"] for t in targets]

        base_bbox_losses = torch.mean(nn.functional.l1_loss(corners_to_center_format(base_boxes), corners_to_center_format(target_boxes), reduction="none"), dim=-1)
        base_giou_losses = (1 - torch.diag(generalized_box_iou(base_boxes.view(-1, 4), target_boxes.view(-1, 4)))).view(-1, 12)

        bbox_losses = torch.mean(nn.functional.l1_loss(corners_to_center_format(pred_boxes), corners_to_center_format(target_boxes), reduction="none"), dim=-1)
        giou_losses = (1 - torch.diag(generalized_box_iou(pred_boxes.view(-1, 4), target_boxes.view(-1, 4)))).view(-1, 12)

        for image_path, image_size, gt_bbox, base_bbox, base_bbox_loss, base_giou_loss, base_iou_score, pred_bbox, bbox_loss, giou_loss, iou_score in zip(image_paths, image_sizes, target_boxes, 
                                                                                                                                                          base_boxes, base_bbox_losses, base_giou_losses, base_iou_scores, 
                                                                                                                                                          pred_boxes, bbox_losses, giou_losses, iou_scores):
            gt_bbox *= torch.tensor([image_size[0], image_size[1], image_size[0], image_size[1]])
            base_bbox *= torch.tensor([image_size[0], image_size[1], image_size[0], image_size[1]])
            pred_bbox *= torch.tensor([image_size[0], image_size[1], image_size[0], image_size[1]])

            visualize_bbox(image_path, gt_bbox, base_bbox, base_bbox_loss, base_giou_loss, base_iou_score, pred_bbox, bbox_loss, giou_loss, iou_score, store_path=store_path)

    # # reduce loss if DDP
    # if dist.is_initialized():
    #     loss = reduce_losses(loss, images['pixel_values'].shape[0])
    # else:
    #     loss = loss / images['pixel_values'].shape[0]
    loss = loss / images['pixel_values'].shape[0] # average over batch size
    return pred_logits, loss, loss_dict

def compute_text_embeds(model, processor, all_descriptions, all_descriptions_val, args, device):
    print("")
    with torch.no_grad():
        text_inputs_parts = processor(text=all_parts, padding="max_length", truncation=True, return_tensors="pt").to(device)
        total_descriptors_part = text_inputs_parts['input_ids'].shape[0]
        text_inputs_parts['input_ids'] = text_inputs_parts['input_ids'].repeat(args.batch_size, 1)
        text_inputs_parts['attention_mask'] = text_inputs_parts['attention_mask'].repeat(args.batch_size, 1)

        text_embeds = []
        num_batches = math.ceil(len(all_descriptions) / args.batch_size)
        for i in range(num_batches):
            start = i * args.batch_size
            end = (i+1) * args.batch_size
            text_inputs = processor(text=all_descriptions[start:end], padding="max_length", truncation=True, return_tensors="pt").to(device)
            if hasattr(model, "module"):
                text_embeds.append(model.module.owlvit.get_text_features(**text_inputs))
            else:
                text_embeds.append(model.owlvit.get_text_features(**text_inputs))

        text_embeds_val = []
        num_batches = math.ceil(len(all_descriptions_val) / args.batch_size)
        for i in range(num_batches):
            start = i * args.batch_size
            end = (i+1) * args.batch_size
            text_inputs_val = processor(text=all_descriptions_val[start:end], padding="max_length", truncation=True, return_tensors="pt").to(device)
            if hasattr(model, "module"):
                text_embeds_val.append(model.module.owlvit.get_text_features(**text_inputs_val))
            else:
                text_embeds_val.append(model.owlvit.get_text_features(**text_inputs_val))

        text_embeds = torch.cat(text_embeds, dim=0).cpu().detach()
        text_embeds_val = torch.cat(text_embeds_val, dim=0).cpu().detach()

    return text_embeds, text_embeds_val, text_inputs_parts, total_descriptors_part

def train_loop(dataset: str,
               model: callable,
            #    processor: callable,
               data_loader: DataLoader,
               device: str,
               optimizer: torch.optim.Optimizer = None,
               num_classes: int = None,
               train_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               log_interval: int = 10,
               epoch: int = 0,
               rank: int = 0,
               wandbLogger: wandb.wandb_sdk.wandb_run.Run = None,
               precompute: bool = True,
               eval_only: bool = False,
               weight_dict: dict = None,
               is_dp: bool = False,
               text_embeds: torch.Tensor = None,
               text_inputs_parts: dict = None,
               total_descriptors_part: int = None,
               num_negatives: int = None,
               templated_descriptions: dict = None,
               ):
    
    # to pretend training script as evaluation script, need separate evaluation script for faster inference.
    runtime = 'val' if eval_only else 'train'
    if args.eval_test:
        runtime = 'test'

    model.train()
    #BUG: when using DDP or DP, model should be wrapped by DDP or DP module, taking it out will make DDP or DP not working.
    # local_model = model.module if world_size > 1 or is_dp else model

    epoch_loss = 0
    batch_losses = []
    acc_metric_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    acc_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)
    accuracy_dict = {f"{runtime}/top1(%)": 100 * 0, f"{runtime}/top5(%)": 100 * 0}
    obj_det_epoch_loss_dict = {}

    #1620MB
    # feed parts' names to OwlViT for localization
    torch.autograd.set_detect_anomaly(True)

    # 3152MB 
    store_path = None
    visualize_batch_count = 0
    if args.visualize > 0:
        visualize_batch_count = args.visualize

        # Create folder for today's date and time
        store_path = f"../results/bird_soup/visualization/{runtime}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(store_path, exist_ok=True)

    load_boxes_base = args.eval_test and "cub_test_reindexed" in args.test_file
    if load_boxes_base:
        df_cub_test = pd.read_hdf(args.test_file)

    all_cls_ids = set(range(len(templated_descriptions)))
    for batch_idx, batch_data in tqdm(enumerate(data_loader), desc=f"{runtime} epoch {epoch}", total=len(data_loader)):
        if not precompute:
            raise NotImplementedError("Current Training script only support precomputed (boxes) mode")

        images, targets_cls, image_paths = batch_data
        images, targets_cls = images.to(device), targets_cls.to(device)

        batch_size = images.pixel_values.shape[0]
        query_size = text_inputs_parts['input_ids'].shape[0]
        #Handle the last batch separately
        if isinstance(model, torch.nn.DataParallel) and batch_size*total_descriptors_part != query_size:
            # drop some images to make sure the query size is a multiple of number of GPUs (for DP)
            print(f"Batch size: {batch_size}, Query size: {query_size}, num descriptors: {total_descriptors_part}")
            batch_size = (batch_size // len(device_list)) * len(device_list)
            images['pixel_values'] = images['pixel_values'][:batch_size]
            targets_cls = targets_cls[:batch_size]
            image_paths = image_paths[:batch_size]
            # change the query size to be a multiple of batch size
            text_inputs_parts['input_ids'] = text_inputs_parts['input_ids'][:total_descriptors_part].repeat(batch_size, 1)
            text_inputs_parts['attention_mask'] = text_inputs_parts['attention_mask'][:total_descriptors_part].repeat(batch_size, 1)
                
        elif batch_idx == len(data_loader) - 1 or batch_size != args.batch_size:
            text_inputs_parts['input_ids'] = text_inputs_parts['input_ids'][:total_descriptors_part].repeat(batch_size, 1)
            text_inputs_parts['attention_mask'] = text_inputs_parts['attention_mask'][:total_descriptors_part].repeat(batch_size, 1)
        

        images['pixel_values'] = images['pixel_values'].squeeze(1).to(device)

        # 3160MB
        # ------------------------------------------------------------------
        # Increase/Reduce number of classes for contrastive learning
        # ------------------------------------------------------------------
        if args.network_type == "contrastive" and runtime in ["train", "val"]:
            # Select X from the remaining classes with X = num_negatives - set(targets_cls)
            text_desc_embeds, targets_cls = add_extra_negatives(runtime, text_embeds, all_cls_ids, num_negatives, targets_cls)

            # Also update number of classes in the model for contrastive loss in the upper branch
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, DDP):
                model.module.update_num_classes(num_negatives)
            else:
                model.update_num_classes(num_negatives)
        else:
            text_desc_embeds = text_embeds.clone()
            
        # ------------------------------------------------------------------
        # 3160MB
        # Update targets for box and class losses in addition to the xclip loss
        image_ids = [".".join(image_path.split("/")[-1].split(".")[:-1]) for image_path in image_paths]
        targets = []
        for idx, image_id in enumerate(image_ids):
            if dataset == "bird_soup":
                boxes_dir = f"{PRECOMPUTED_DIR}/{dataset}/data_updated_v2/{image_id}.pth"
                key_boxes = "boxes_info"    # For precomputed boxes from BirdSoup-v1
                key_logits = "part_logits_owlvit_base"

                boxes_dir_base = None
                if load_boxes_base:
                    ori_image_name = df_cub_test.loc[df_cub_test['new_image_name'] == image_id + ".jpg"]["org_image_name"].values[0]
                    boxes_dir_base = f"{PRECOMPUTED_DIR}/cub/test/owlvit-base-patch32_cub-12-parts/data/{ori_image_name.replace('.jpg', '.pth')}"
            elif dataset == 'stanforddogs':
                boxes_dir = f"{PRECOMPUTED_DIR}/{dataset}/part_boxes/owlvit-large-patch14_stanforddog-6-parts-dog_dog_update_logits/{image_id}.pth"
                key_boxes = "boxes_info"
                key_logits = 'part_logits'
                
            else:
                boxes_dir = f"../pred_boxes/{dataset}/owl_vit_owlvit-large-patch14_descriptors_chatgpt_groundtruths/{image_id}.pth"
                key_boxes = "boxes"
                key_logits = "logits_owlvit_base"

            info_boxes = torch.load(boxes_dir, map_location=device)
            if dataset == "bird_soup" and key_boxes not in info_boxes:
                key_boxes = "part_boxes"    # For precomputed boxes from eBird

            if len(all_parts) != len(info_boxes[key_boxes].keys()):
                info_boxes[key_boxes] = {part: info_boxes[key_boxes][part] for part in all_parts}
                info_boxes[key_logits] = info_boxes[key_logits][:, sel_part_indices]

            assert list(info_boxes[key_boxes].keys()) == all_parts

            # Bounding box coordinates in corner format to [0, 1]
            bboxes = []
            bboxes_ori = []
            image_width, image_height = info_boxes["image_size"]

            for bbox in list(info_boxes[key_boxes].values()):
                bboxes_ori.append(bbox)

                x1, y1, x2, y2 = bbox
                x1_normalized = x1 / image_width
                y1_normalized = y1 / image_height
                x2_normalized = x2 / image_width
                y2_normalized = y2 / image_height
                bboxes.append([x1_normalized, y1_normalized, x2_normalized, y2_normalized])

            bboxes_base = []
            if load_boxes_base:
                info_boxes_base = torch.load(boxes_dir_base, map_location=device)

                for bbox in list(info_boxes_base["boxes_info"].values()):
                    x1, y1, x2, y2 = bbox
                    x1_normalized = x1 / image_width
                    y1_normalized = y1 / image_height
                    x2_normalized = x2 / image_width
                    y2_normalized = y2 / image_height
                    bboxes_base.append([x1_normalized, y1_normalized, x2_normalized, y2_normalized])

            targets.append({"class_labels": torch.tensor(list(range(len(all_parts)))).to(device),
                            "boxes_ori": bboxes_ori,                # 12 boxes from OwlViT-large
                            "boxes": bboxes,                        # 12 boxes from OwlViT-large - normalized
                            "boxes_base": bboxes_base,              # 12 boxes from OwlViT-base - normalized                            
                            "logits": info_boxes[key_logits],       # box logits (576, 12) from OwlViT-base
                            "targets_cls": targets_cls[idx],
                            "image_size": info_boxes["image_size"],
                            "image_path": info_boxes["image_path"],
                            })

        if eval_only:
            with torch.no_grad():
                logits, loss, loss_dict = forward_inputs(model, images, text_inputs_parts, text_desc_embeds, targets, weight_dict, visualize_batch_count, store_path)

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
        else:
            optimizer.zero_grad()
            logits, loss, loss_dict = forward_inputs(model, images, text_inputs_parts, text_desc_embeds, targets, weight_dict, visualize_batch_count, store_path)

            loss.backward()
            optimizer.step()
            if train_scheduler is not None:
                train_scheduler.step()

            # compute loss and update model's weights
            # if dist.is_initialized():
            #     loss = reduce_losses(loss, 1) # reduce losses over all GPUs for logging, batch size already considered so set to 1.
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            
        # log model accuracy
        if acc_metric_top1.num_classes == logits.shape[-1]:
            acc_metric_top1.update(logits, targets_cls)
            acc_metric_top5.update(logits, targets_cls)
        else:
            acc_metric_top1.update(logits.argmax(dim=-1), targets_cls)
            # acc_metric_top5.update(torch.tensor([-1] * len(targets_cls)), targets_cls)    # dummy values

        # a dictionary of current accuracy, e.g., {"top1": 0.1, "top5": 0.2}
        accuracy_dict = {f"{runtime}/top1(%)": 100 * acc_metric_top1.compute().item(),
                         f"{runtime}/top5(%)": 100 * acc_metric_top5.compute().item()}

        # od_batch_loss_dict = {f"{runtime}/{k}": v for k, v in loss_dict.items()}
        for key in set(obj_det_epoch_loss_dict.keys()) | set(loss_dict.keys()):
            obj_det_epoch_loss_dict[key] = obj_det_epoch_loss_dict.get(key, []) + [loss_dict[key].item()]

        if visualize_batch_count > 0:
            visualize_batch_count -= 1

        if rank in {-1, 0}:
            if wandbLogger is not None:
                wandbLogger.log(accuracy_dict)
                wandbLogger.log({f"{runtime}/batch_loss": batch_loss})

                # If training to get rid of teacher model => Monitor the loss of the object detection branch
                if not args.logits_from_teacher:
                    wandbLogger.log({f"{runtime}/{key}": values.item() for key, values in loss_dict.items()})

            if batch_idx % log_interval == 0:
                print(f"{runtime} epoch {epoch}: [{batch_idx * len(targets)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tBatch Loss: {batch_loss:.6f}\tTop1: {accuracy_dict[f'{runtime}/top1(%)']:.2f}")
                print(" ".join([f"{k}: {v:.5f}" for k, v in loss_dict.items()]))

    epoch_loss = sum(batch_losses) / len(batch_losses)
    obj_det_epoch_loss_dict = {f"{runtime}/epoch_{key}": (sum(values) / len(values)) for key, values in obj_det_epoch_loss_dict.items()}

    if eval_only:
        return epoch_loss, accuracy_dict[f'{runtime}/top1(%)'], accuracy_dict[f'{runtime}/top5(%)'], obj_det_epoch_loss_dict

    return model, epoch_loss, accuracy_dict[f'{runtime}/top1(%)'], accuracy_dict[f'{runtime}/top5(%)'], obj_det_epoch_loss_dict


def parse_arguments():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------
    #   Must-check arguments for experiments but usually FIXED
    # ------------------------------------------------------------
    parser.add_argument('--model', help='select model', default="owlvit-large-patch14", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
    parser.add_argument('--dataset', help='select dataset', default="cub", choices=["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "places365", "cub", "nabirds", "bird_soup", "stanforddogs"])
    parser.add_argument('--sub_datasets', help='select a group of datasets in Bird Soup', default="all")
    parser.add_argument('--distortion', help='select distortion type if using ImageNet-C', default="defocus_blur", choices=["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "shot_noise", "gaussian_noise", "impulse_noise"])
    parser.add_argument('--distortion_severity', type=int, help='select distortion severity if using ImageNet-C', default=1, choices=[1, 2, 3, 4, 5])

    parser.add_argument('--epochs', type=int, help='num epochs', default=50)
    parser.add_argument('--batch_size', type=int, help='num training batch size', default=32)
    parser.add_argument('--batch_size_val', type=int, help='num validation batch size for contrastive learning only', default=32)
    parser.add_argument('--eval_size', type=float, help='to select a proportion for val from train set', default=0.2)
    parser.add_argument('--num_workers', type=int, help='num workers for batch processing', default=16)
    parser.add_argument('--num_samples', type=int, help='num images per class', default=-1)
    parser.add_argument('--devices', help='select device', default="0", type=str)
    parser.add_argument('--random_seed', help='random seed (for data subsampling only)', default=42, type=int)
    parser.add_argument('--early_stopping', type=int, help='specify a number greater than 0 if using early stopping', default=0)

    # ------------------------------------------------------------
    #   Must-check arguments for experiments: FREQUENTLY CHANGE
    # ------------------------------------------------------------
    parser.add_argument('--descriptors', help='select descriptors for OwlViT', default="chatgpt", choices=["sachit", "chatgpt", 'stanforddogs'])
    parser.add_argument('--prompt_type', type=int, help='select prompt type', default=5)
    parser.add_argument('--owlvit_threshold', type=float, help='select threshold for owl_vit', default=-1)
    parser.add_argument('--owlvit_conf_scores', help='use owlvit scores as confidence scores', action="store_true")

    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.0001)
    parser.add_argument('--save_freq', type=int, help='save results after a number of epochs', default=0)
    parser.add_argument('--scheduler_mode', type=str, help='select mode for scheduler', default="min")
    parser.add_argument('--scheduler_factor', type=float, help='select factor for scheduler', default=0.5)
    parser.add_argument('--scheduler_patience', type=int, help='select patience for scheduler', default=5)
    parser.add_argument('--scheduler_verbose', action="store_true", help='print logs for scheduler')

    parser.add_argument('--num_negatives_train', type=int, help='number of train negatives for contrastive learning', default=32)
    parser.add_argument('--num_negatives_val', type=int, help='number of val negatives for contrastive learning', default=32)

    parser.add_argument('--eval_test', help='run evaluation on test set', action="store_true")
    parser.add_argument('--zeroshot_split', help='split 75% classes for training and 25% classes for testing', action="store_true")
    parser.add_argument("--best_model", type=str, default="", help="load the best model for evaluation.")

    parser.add_argument('--loss_weights', type=str, help='loss weights for loss_ce, loss_bbox, loss_iou, loss_sym_box_label and loss_xclip separated by commas, respectively', default="1,1,1,1,1")
    parser.add_argument('--freeze_box_heads', help='freeze weights of box and class heads of OwlViT', action="store_true")
    parser.add_argument('--train_box_heads_only', help='freeze weights of vision encoder and MLP CLS head of OwlViT', action="store_true")
    parser.add_argument('--network_type', help='select network type for training: contrastive or classification', default=None, choices=["contrastive", "classification"])
    parser.add_argument('--classification_loss', help='select the main loss for image classification', default=None, choices=["ce_loss", "focal_loss"])
    parser.add_argument('--contrastive_sampler', help='select the sampling strategy for contrastive', default=None, choices=["refilled_empty_classes", "removed_empty_classes"])
    parser.add_argument('--logits_from_teacher', help='use boxes logits from teacher model (e.g., OwlViT-base) for 12 box selection ', action="store_true")

    parser.add_argument("--part_names", type=str, default=None, help="select part names for training and evaluation in "
                                                                     "'back,beak,belly,breast,crown,forehead,eyes,legs,wings,nape,tail,throat'. Default to None")

    parser.add_argument('--train_file', type=str, help='set path to the train file', default=BIRD_SOUP_META_PATH_TRAIN)
    parser.add_argument('--val_file', type=str, help='set path to the validation file', default=BIRD_SOUP_META_PATH_VAL)
    parser.add_argument('--test_file', type=str, help='set path to the test file', default=BIRD_SOUP_META_PATH_TEST)
    parser.add_argument('--descriptor_path', type=str, help='if set, descriptors are loaded from file', default=None)

    parser.add_argument('--birdsoup_level', type=int, help='specify bird soup level from 1 to 3 for logging purposes', choices=[1, 2, 3])
    parser.add_argument('--image_mean_std', help='use image mean and std precomputed from CLIP or BirdSoup images', default="bird_soup", choices=["bird_soup", "clip"])
    parser.add_argument('--finetuning', help='finetune pretrained models on downstream tasks', default=None, choices=["vision_encoder_mlp", "mlp_only"])    # "proj_layer", "linear"

    # For FOCAL loss
    parser.add_argument('--alpha', type=float, help='alpha for focal loss', default=0.25)
    parser.add_argument('--gamma', type=float, help='gamma for focal loss', default=2.0)

    parser.add_argument('--box_head_num_layers', type=int, help='define number of linear layers for box head', default=3)
    parser.add_argument('--ablation', help='run ablation study on number of classes for contrastive training', action="store_true")

    # ------------------------------------------------------------
    #   Analysis
    # ------------------------------------------------------------
    parser.add_argument('--visualize', type=int, help='visualization', default=0)
    parser.add_argument('--test_class_list', help='a file containing a list of class to test', default=[], nargs='+')

    # ------------------------------------------------------------
    #   Others
    # ------------------------------------------------------------
    parser.add_argument('--note', type=str, help='write whatever to quickly remind experiment settings', default="")
    parser.add_argument('--debugging', help='do not save results and logs when debugging', action="store_true")
    parser.add_argument('--overwrite_boxes', help='save boxes for tuning', action="store_true")
    parser.add_argument('--check_box_files', help='check if box file exists (otherwise box computing will skip if the box folder exist.)', action="store_true")
    parser.add_argument('--verbose', help='print logs', action="store_true")
    parser.add_argument("--no_log", action='store_true', help="disable wandb logging.")
    parser.add_argument("--project_name", type=str, default="xclip", help="name of the wandb project.")
    parser.add_argument("--enable_dp", action="store_true", help="enable DataParallel for training. Note: Not efficient, but allow us to train with larger batch size.")
    parser.add_argument("--run_name", type=str, default="", help="name of the wandb run.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    start_time = datetime.now()

    seed_everything(args.random_seed)

    # load general configs for training
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1 # we only have one node so the global rank should always be 0

    if global_rank != -1:
        rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", init_method="env://")
        mp.set_start_method("spawn", force=True)  # change to spawn to avoid issues with pytorch dataloader when num_workers > 0
    else:
        rank = -1

    wandbLogger = None
    if not args.no_log or args.eval_test:
        if args.run_name is None:
            run_name = f"{str(datetime.now().strftime('%m_%d_%Y-%H:%M:%S'))}_lr_{args.lr}_{args.epochs}ep_prompt{args.prompt_type}" 
        else:
            run_name = args.run_name
        sub_dir = f"evaluation_{args.network_type}" if args.eval_test else f"training_{args.network_type}"
        if args.finetuning is not None:
            sub_dir = f"finetune_{args.network_type}/{args.finetuning}"

        out_dir = f'{PROJECT_ROOT}/results/{args.dataset}/level_{args.birdsoup_level}/{sub_dir}/{args.descriptors}-{args.model.replace("/", "")}/{run_name}'
        if args.note != "":
            out_dir += f'_{args.note}'
            run_name += f'_{args.note}'

        if rank in {-1, 0}:
            os.makedirs(out_dir, exist_ok=True)

            if not args.no_log:
                wandbLogger = wandb.init(project=args.project_name, name=run_name, resume=False, dir=out_dir, mode='disabled' if args.no_log else 'online')
            else:
                print("Warning: wandb logging is disabled. Make sure this action is intended.")

            # Save the modified version of the config files
            with open(f'{out_dir}/configs.json', 'w') as f:
                json.dump(args.__dict__, f, indent=4)

    # split the devices into a list and convert to ints
    device_list = [int(x) for x in args.devices.split(",")]
    check_device_availability(device_list)
    device = f'cuda:{device_list[rank]}' if world_size > 1 else f'cuda:{device_list[0]}'
    if not args.enable_dp:
        torch.cuda.set_device(device)

    # load pre-trained model
    owlvit_det_processor = OwlViTProcessor.from_pretrained(f"google/{args.model}")
    owlvit_det_model = OwlViTForObjectDetection.from_pretrained(f"google/{args.model}")

    # TODO: Update MEAN + STD for image normalization
    ''' CLIP VALUES
    image_mean = {list} [0.48145466, 0.4578275, 0.40821073]
    image_std = {list} [0.26862954, 0.26130258, 0.27577711]
    '''
    if args.image_mean_std == "bird_soup":
        owlvit_det_processor.image_processor.image_mean = [0.48168647, 0.49244233, 0.42851609]
        owlvit_det_processor.image_processor.image_std = [0.18656386, 0.18614962, 0.19659419]

    # load dataset
    # The order/indices of classes in target_classes MUST BE the same as the order/indices of classes in the dataset
    train_dataset, val_dataset = load_training_dataset(args.dataset, args.sub_datasets, args.eval_size, transform=owlvit_det_processor, random_state=args.random_seed, zeroshot_split=args.zeroshot_split)
    target_classes = train_dataset.classes if hasattr(train_dataset, "classes") else train_dataset.dataset.classes
    target_classes_val = val_dataset.classes if hasattr(val_dataset, "classes") else val_dataset.dataset.classes
    # ### for debugging ###
    # # for testing purposes, sample 12345 images from train set
    # from torch.utils.data import Subset
    # sample_idxs = np.random.choice(len(train_dataset), 12345, replace=False)
    # train_dataset = Subset(train_dataset, sample_idxs)

    test_dataset = None
    if args.eval_test:
        test_dataset, _ = load_training_dataset(args.dataset, args.sub_datasets, 1.0, transform=owlvit_det_processor, random_state=args.random_seed, zeroshot_split=args.zeroshot_split, split="test")
        target_classes = test_dataset.classes if hasattr(test_dataset, "classes") else test_dataset.dataset.classes

    if args.dataset == "bird_soup":
        target_classes = [c.lower().replace("-", " ").replace("'s", "") for c in target_classes]
        target_classes_val = [c.lower().replace("-", " ").replace("'s", "") for c in target_classes_val]

    # prepare text embeddings
    # Use target_classes to filter out classes that are not in the dataset (for BirdSoup)
    descriptions_only, _ = load_descriptions(dataset_name=args.dataset, prompt_type=0, desc_type=args.descriptors, target_classes=target_classes, descriptor_path=args.descriptor_path, unmute=False)
    templated_descriptions, _ = load_descriptions(dataset_name=args.dataset, prompt_type=args.prompt_type, desc_type=args.descriptors, target_classes=target_classes, descriptor_path=args.descriptor_path, unmute=rank in {-1, 0})
    templated_descriptions_val, _ = load_descriptions(dataset_name=args.dataset, prompt_type=args.prompt_type, desc_type=args.descriptors, target_classes=target_classes_val, descriptor_path=args.descriptor_path, unmute=False)

    # Sorted the keys in templated_descriptions to match the order of classes in target_classes
    assert set(templated_descriptions.keys()) == set(target_classes)
    templated_descriptions = {k: templated_descriptions[k] for k in sorted(templated_descriptions, key=target_classes.index)}
    templated_descriptions_val = {k: templated_descriptions_val[k] for k in sorted(templated_descriptions_val, key=target_classes_val.index)}

    # The class names in class_list must match target_classes to correctly map the descriptors to the classes
    assert list(templated_descriptions.keys()) == target_classes
    assert list(templated_descriptions_val.keys()) == target_classes_val

    num_classes = len(templated_descriptions.keys())
    all_descriptions = list(templated_descriptions.values())
    all_descriptions_val = list(templated_descriptions_val.values())

    # Use parts only for localization
    all_parts = []
    if args.descriptors in {"chatgpt", "stanforddogs"}:
        all_parts = [[descriptor.split(":")[0] for descriptor in descriptors if ":" in descriptor] for descriptors in descriptions_only.values()][0]
        sel_part_indices = list(range(len(all_parts)))

        # The order of part names is still preserved
        if args.part_names is not None:
            part_names = args.part_names.split(",")
            sel_part_indices = [i for i, part in enumerate(all_parts) if part in part_names]
            all_parts = [part for part in all_parts if part in part_names]

        all_descriptions = [[descriptor for i, descriptor in enumerate(descriptors) if i in sel_part_indices] for descriptors in all_descriptions]
        all_descriptions_val = [[descriptor for i, descriptor in enumerate(descriptors) if i in sel_part_indices] for descriptors in all_descriptions_val]

    # TODO: For contrastive training, target classes of train and validation are the same EXCEPT FOR ablation study
    # Ablation study uses a separate validation set whose number of classes is different from the train set
    # The number of classes in the validation set for normal training is slightly less than the train set
    # because some classes in the train set do not have enough samples for validation (i.e., >= 3 images)
    if not args.ablation:
        all_descriptions_val = all_descriptions

    # Loss weights for training
    loss_weights = [float(weight) for weight in args.loss_weights.split(",")]
    weight_dict = {"loss_ce": loss_weights[0], "loss_bbox": loss_weights[1], "loss_giou": loss_weights[2],
                   "loss_sym_box_label": loss_weights[3], "loss_xclip": loss_weights[4]}

    # Initialize OwlViT model for Classification
    model = OwlViTForClassification(owlvit_det_model=owlvit_det_model, num_classes=num_classes, num_parts=len(all_parts),
                                    freeze_box_heads=args.freeze_box_heads, train_box_heads_only=args.train_box_heads_only,
                                    network_type=args.network_type, classification_loss=args.classification_loss,
                                    weight_dict=weight_dict, logits_from_teacher=args.logits_from_teacher,
                                    finetuning=args.finetuning, alpha=args.alpha, gamma=args.gamma,
                                    device=None if args.enable_dp else device,)

    if rank in {-1, 0}:
        trained_params, frozen_params = 0, 0
        for name, param in model.named_parameters():
            if not args.no_log and rank in {-1, 0}:
                print(f"{name}: {param.shape if len(param.shape) > 0 else param.type()} - Required grad: {param.requires_grad}")

            params = reduce(operator.mul, list(param.shape)) if len(param.shape) > 0 else 1
            if param.requires_grad:
                trained_params += params
            else:
                frozen_params += params

        print(f"Trainable parameters: {format(trained_params, ',')}")
        print(f"Frozen parameters: {format(frozen_params, ',')}")

    if args.best_model:
        # Load best model: OwlViT-Base + OwlViT_CLS_PLUS + Finetune MLP + Vision encoder
        ckpt = torch.load(args.best_model, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)

    # TODO: FORCE UPDATING BOX HEAD
    if args.box_head_num_layers > 3:
        model.update_box_head(args.box_head_num_layers)

    # Write model architecture to file
    if not args.no_log and rank in {-1, 0}:
        # write model architecture to file for tracking purposes
        with open(f'{out_dir}/model_arch.txt', 'w') as f:
            f.write(str(model) + "\n\n")

            for name, param in model.named_parameters():
                f.write(f"{name}: {param.shape if len(param.shape) > 0 else param.type()} - Required grad: {param.requires_grad}\n")

            f.write(f"\nTrainable parameters: {format(trained_params, ',')}\n")
            f.write(f"Frozen parameters: {format(frozen_params, ',')}\n")

    if world_size > 1:
        # TODO: Batch_size 200 (i.e., 200 images per batch) OR Batch_size 32 + num_negatives = 200 ???
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if test_dataset else None
        train_shuffle = val_shuffle = False
    else:
        train_sampler, val_sampler, test_sampler = None, None, None
        train_shuffle = val_shuffle = True

    test_shuffle = True
    # test_shuffle = False
    # if args.network_type == "contrastive":
    #     test_shuffle = True

    # prepare dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, shuffle=train_shuffle, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, num_workers=args.num_workers, sampler=val_sampler, shuffle=val_shuffle, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=test_sampler, shuffle=test_shuffle, pin_memory=True) if args.eval_test else None

    # SyncBatchNorm
    if world_size > 1 and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

        # TODO: Check if the data is randomly distributed across the GPUs
        # for i, (input, target) in enumerate(train_loader):
        #     # If the GPU id is accessible as variable `gpu`
        #     if i < 10:  # print only for first 10 batches
        #         print(f"GPU: {rank}, Batch: {i}, First item ID: {input[0]['id']}")

    if rank != -1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    else:
        model = model.to(device)
        
    if args.enable_dp:
        model = torch.nn.DataParallel(model, device_ids=device_list, output_device=device)

    # if resume:
        # load the latest checkpoint
        # resume_path = train_config["log_dir"] if resume_path is None else resume_path
        # checkpoint = torch.load(os.path.join(resume_path, "last.pt"))
        # model.load_state_dict(checkpoint["model"])

    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # use plateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.scheduler_mode, factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=args.scheduler_verbose)

    # compute the text embeddings 
    text_embeds, text_embeds_val, text_inputs_parts, total_descriptors_part = compute_text_embeds(model, owlvit_det_processor, all_descriptions, all_descriptions_val, args, device)

    # TODO: try a different scheduler
    # train_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
    # At https://github.com/openai/CLIP/issues/107
    # The LR schedule didn't have restarts, the multiplier becomes (almost) 1.0 after warmup, and monotonically decreases 0.0 drawing a cosine curve over 32 epochs.
    # A linear warmup was done over the first 2,000 iterations and is not dependent of the period of the cosine function.
    
    # train loops
    if not args.eval_test:
        val_best_acc, val_best_loss, best_epoch = 0, 9999, 0
        early_stopping = 0

        for epoch in range(args.epochs):
            # train
            train_results = train_loop(dataset=args.dataset, model=model, data_loader=train_loader,
                                       num_classes=num_classes, device=device, rank=rank, wandbLogger=wandbLogger,
                                       optimizer=optimizer, log_interval=10, epoch=epoch, weight_dict=weight_dict, is_dp=args.enable_dp,
                                       text_inputs_parts=text_inputs_parts, total_descriptors_part=total_descriptors_part,
                                       text_embeds=text_embeds, num_negatives=args.num_negatives_train, templated_descriptions=templated_descriptions)

            model, train_loss, train_top1, train_top5, train_od_epoch_loss_dict = train_results

            # eval
            eval_results = train_loop(dataset=args.dataset, model=model, data_loader=val_loader,
                                      num_classes=num_classes, device=device, rank=rank,  wandbLogger=wandbLogger,
                                      eval_only=True, weight_dict=weight_dict, is_dp=args.enable_dp,
                                      text_inputs_parts=text_inputs_parts, total_descriptors_part=total_descriptors_part,
                                      text_embeds=text_embeds_val, num_negatives=args.num_negatives_val, templated_descriptions=templated_descriptions_val)

            val_loss, val_top1, val_top5, val_od_epoch_loss_dict = eval_results

            if world_size > 1:
                # reduce the validation loss across all processes
                dist.all_reduce(torch.tensor(val_loss).to(device), op=dist.ReduceOp.SUM)
            scheduler.step(val_loss)

            if rank in {-1, 0}:
                train_log_dict = {"train/epoch_loss": train_loss, 'train/epoch_top1': train_top1, 'train/epoch_top5': train_top5}
                val_log_dict = {"val/epoch_loss": val_loss, 'val/epoch_top1': val_top1, 'val/epoch_top5': val_top5}
                
                if not args.no_log:
                    wandb.log(train_log_dict | val_log_dict | train_od_epoch_loss_dict | val_od_epoch_loss_dict | {"epochs": epoch}, commit=False)

                sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                if args.save_freq > 0 and epoch % args.save_freq == 0:
                    torch.save(sd, os.path.join(wandb.run.dir, f"e{epoch:04d}.pt"))

                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    val_best_acc = val_top1
                    best_epoch = epoch
                    torch.save(sd, os.path.join(wandb.run.dir, "best.pt"))
                    early_stopping = 0
                else:
                    early_stopping += 1

                torch.save(sd, os.path.join(wandb.run.dir, "last.pt"))

            if 0 < args.early_stopping <= early_stopping:
                print("EARLY STOPPING TRIGGERED... QUIT TRAINING")
                break

        best_acc = val_best_acc
        best_loss = val_best_loss

        print(f'Best val/val_loss top1: {val_best_acc:.4f}/{val_best_loss:.4f} at epoch {best_epoch}.')
        print(" ".join([f"{k}: {v:.5f}" for k, v in val_od_epoch_loss_dict.items()]))
    else:
        # test_results = train_loop(dataset=args.dataset, model=model, processor=owlvit_det_processor, data_loader=test_loader,
        #                           num_classes=num_classes, device=device, rank=rank, wandbLogger=wandbLogger,
        #                           eval_only=True, weight_dict=weight_dict)
        test_results = train_loop(dataset=args.dataset, model=model, data_loader=test_loader,
                                    num_classes=num_classes, device=device, rank=rank,  wandbLogger=wandbLogger,
                                    eval_only=True, weight_dict=weight_dict, is_dp=args.enable_dp,
                                    text_inputs_parts=text_inputs_parts, total_descriptors_part=total_descriptors_part,
                                    text_embeds=text_embeds_val, num_negatives=args.num_negatives_val, templated_descriptions=templated_descriptions_val)


        test_loss, test_top1, test_top5, test_od_epoch_loss_dict = test_results

        best_acc = test_top1
        best_loss = test_loss
        best_epoch = None

        print(f'Best test / loss top1: {best_acc:.4f} / {best_loss:.4f}.')
        print(" ".join([f"{k}: {v:.5f}" for k, v in test_od_epoch_loss_dict.items()]))

    end_time = datetime.now()

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump({f"Best Accuracy ({'test' if args.eval_test else 'val'})": best_acc,
                   "Best epoch": best_epoch,
                   "Number of examples": len(val_dataset),
                   "Start time": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                   "End time": end_time.strftime("%d/%m/%Y %H:%M:%S"),
                   "Duration": round((end_time - start_time).total_seconds() * 1.0 / 3600, 2)}, f, indent=4)

