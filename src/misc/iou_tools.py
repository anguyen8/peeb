import json
from pathlib import Path
from collections import defaultdict

import fire
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# VIS_DICT = {'cub': json.load(open("data/cub_vis_binary.json", 'r'))}
# cub_part_keypoints = json.load(open("data/cub_12_parts_vis_dicts.json", 'r'))
# PART_NAMES = ['crown', 'forehead', 'nape', 'eyes', 'beak', 'throat', 'breast', 'belly',  'back', 'wings', 'legs', 'tail']
# # create dict that use image_id as key
# cub_dict = {}
# for item in cub_part_keypoints:
#     cub_dict[item['image_id']] = {}
#     for part_name in PART_NAMES:
#         cub_dict[item['image_id']][part_name] = item[part_name]
                               
# nabird_parts = ['beak', 'crown', 'nape', 'eyes', 'belly', 'breast', 'back', 'tail', 'wings']
# nabird_dict = json.load(open("data/nabirds_9_parts_vis_dict.json", 'r'))
# nabird_dict_ = {}
# for key, item in nabird_dict.items():
#     nabird_dict_[key] = {}
#     for part_name in nabird_parts:
#         nabird_dict_[key][part_name] = item[part_name]
# PART_VIS_DICT = {'cub': cub_dict, 'nabird': nabird_dict_}


def corner_to_center(boxes: torch.Tensor, sizes: torch.Tensor = None, normalized: bool = True):
    """
    Convert boxes from (xmin, ymin, xmax, ymax) to (center_x, center_y, width, height)
    :param boxes: (Tensor[N, 4]) boxes in corner format
    :param sizes: (Tensor[N, 2]) sizes of boxes in (width, height) format
    :param normalized: (bool) whether the boxes are normalized or not
    :return: (Tensor[N, 4]) boxes in center format
    """
    # compute the centers
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    # compute the width and height
    wh = boxes[:, 2:] - boxes[:, :2]
    if not normalized:
        centers = centers / sizes
        wh = wh / sizes
    # concatenate the centers and wh tensors
    return torch.cat([centers, wh], dim=1)

def compute_giou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute the generalized IoU between two boxes
    :param boxes1: (Tensor[N, 4]) boxes in corner format
    :param boxes2: (Tensor[M, 4]) boxes in corner format
    :return: (Tensor[N, M]) generalized IoU between boxes1 and boxes2
    """
    # Compute the area of boxes1 and boxes2
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute the coordinates of the intersection of boxes1 and boxes2
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    # Compute the area of the intersection
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Compute the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    
    # Compute the area of the smallest enclosing box
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # Compute the IoU
    iou = inter_area / (area1[:, None] + area2 - inter_area)
    
    # Compute the GIoU
    giou = iou - (enclose_area - area1[:, None] - area2 + inter_area) / enclose_area
    
    return giou, iou

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute the pair-wise IoU between boxes1 and boxes2, i.e. IoU(box1[0], box2[0]), IoU(box1[1], box2[1]), ...
    :param boxes1: (Tensor[N, 4]) boxes in corner format
    :param boxes2: (Tensor[M, 4]) boxes in corner format 
    return: (Tensor[N, M]) IoU between boxes1 and boxes2
    """
    # Compute the area of boxes1 and boxes2
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute the coordinates of the intersection of boxes1 and boxes2
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    # Compute the area of the intersection
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Compute the IoU
    iou = inter_area / (area1[:, None] + area2 - inter_area)
    
    return iou

def compute_l1_loss(boxes: torch.Tensor, targets: torch.Tensor, reduce: str = "mean"):
    """
    Compute the L1 loss between boxes and targets
    :param boxes: (Tensor[N, 4]) boxes in center format
    :param targets: (Tensor[N, 4]) targets in center format
    :return: (Tensor[N]) L1 loss between boxes and targets
    """
    loss = torch.abs(boxes - targets)
    if reduce == "mean":
        loss = loss.mean(dim=1)
    elif reduce == "sum":
        loss = loss.sum(dim=1)
    return loss

def normalize_boxes(boxes: torch.Tensor, sizes: torch.Tensor):
    """
    Normalize boxes according to the image size
    :param boxes: (Tensor[N, 4]) boxes in corner format
    :param sizes: (Tensor[N, 2]) sizes of the images
    :return: (Tensor[N, 4]) normalized boxes
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w, h = sizes.unbind(dim=1)
    return torch.stack([(x1 / w), (y1 / h), (x2 / w), (y2 / h)], dim=1)

def compute_map_generic(pred_boxes, gt_boxes, scores, iou_thresholds):
    """
    Compute the mean average precision for given IoU thresholds
    :param pred_boxes: Predicted boxes [N, 4]
    :param gt_boxes: Ground truth boxes [M, 4]
    :param scores: Confidence scores for the predicted boxes [N]
    :param iou_thresholds: IoU thresholds
    :return: mean average precision
    """
    num_gts = gt_boxes.shape[0]
    num_preds = pred_boxes.shape[0]
    # Compute IoU matrix [N, M]
    iou_matrix = compute_iou(pred_boxes, gt_boxes)  # This function needs to be defined

    # For each IoU threshold
    APs = []
    for iou_threshold in iou_thresholds:
        tp = 0
        fp = 0
        fn = num_gts
        precisions = []
        recalls = []
        # Sort detections by score
        sorted_indices = torch.argsort(scores, descending=True)
        for idx in sorted_indices:
            matched_gts = (iou_matrix[idx] > iou_threshold).nonzero(as_tuple=True)[0]
            if matched_gts.shape[0] == 0:
                fp += 1
            else:
                tp += 1
                fn -= 1
                # Remove this matched gt from future consideration
                iou_matrix[:, matched_gts[0]] = -1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            precisions.append(precision)
            recalls.append(recall)

        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        AP = torch.trapz(precisions, recalls)
        APs.append(AP)

    mAP = torch.mean(torch.tensor(APs))
    return mAP

def compute_map_simplified(ious: torch.Tensor, threshold: list[float]) -> torch.Tensor:
    """
    Compute the mean average precision for a given IoU thresholds
    :param ious: (Tensor[N,]) IoUs between the predicted boxes and the ground truth boxes
    :param threshold: (list[float]) IoU thresholds
    :return: (Tensor[1,]) mean average precision
    """
    APs = []
    for t in threshold:
        precision = (ious >= t).float().mean()
        recall = 1.0  # Recall is always 1 in this scenario
        AP = precision  # Since recall spans from 0 to 1, AP is equal to precision
        APs.append(AP)

    mAP = torch.mean(torch.tensor(APs))
    return mAP

def point_in_box(point: list[int, int], box: list[int, int, int, int]) -> bool:
    """
    Check whether the point is inside the box
    :param point: (list[int, int]) point in (x, y) format
    :param box: (list[int, int, int, int]) box in (x1, y1, x2, y2) format
    :return: (bool) whether the point is inside the box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_inside_box(keypoints: list[list[int, int]], box: list[int, int, int, int])-> bool:
    # sourcery skip: use-any
    """
    Check whether the list of keypoints are inside the box
    :param keypoints: (list[list[int, int]]) list of keypoints in (x, y) format
    :param box: (list[int, int, int, int]) box in (x1, y1, x2, y2) format
    :return: (bool) whether the keypoints are inside the box
    """

    for keypoint in keypoints:
        if point_in_box(keypoint, box):
            return True
    return False

def compute_box_losses(result_path: str, gt_folder: str, visible_only: bool = False, gt_json: str = None, dataset: str = "cub", correct_only: bool = False, test_df: str = None):
    results = json.load(open(result_path, 'r'))
    loss_dict = {'giou': [], 'l1': [], 'iou': [], 'pred_boxes': [], 'box_scores': [], 'gt_boxes': [], 'inside_keypoint': [], 'inside_keypoint_gt': []}
    part_loss_dict = {'giou': defaultdict(list), 'l1': defaultdict(list), 'iou': defaultdict(list), 'pred_boxes': defaultdict(list), 'box_scores': defaultdict(list), 'gt_boxes': defaultdict(list), 'inside_keypoint': defaultdict(list), 'inside_keypoint_gt': defaultdict(list)}
    
    if test_df is not None:
        test_df = pd.read_hdf(test_df)
        target_classes = set(test_df['class_name'].unique().tolist())

    for item in tqdm(results, desc="Computing losses"):
        if test_df is not None and item['ground_truth'] not in target_classes:
            continue
        
        bs_file_name = item['bs_file_name']
        bs_base_name = Path(bs_file_name).stem
        file_name = item['file_name']
        base_name = Path(file_name).stem

        # if we don't have the gt json, consider the teacher boxes as gt
        if gt_json is None:
            gt_file_path = Path(gt_folder) / f'{bs_base_name}.pth'
            gt_info = torch.load(gt_file_path)
            gt_boxes = gt_info['boxes_info'] if "boxes_info" in gt_info else gt_info['part_boxes']
            image_size = gt_info['image_size']

        # visible only and correct only are mutually exclusive
        if visible_only and correct_only:
            raise ValueError("visible_only and correct_only are mutually exclusive")

        if dataset in PART_VIS_DICT:
            part_vis_dict = PART_VIS_DICT[dataset][base_name]
            pred_boxes = item['pred_boxes']
            for part_name, keypoints in part_vis_dict.items():
                if np.sum(keypoints) == 0:
                    continue
                # for pred boxes
                if is_inside_box(keypoints, pred_boxes[part_name]):
                    loss_dict['inside_keypoint'].append(1)
                    part_loss_dict['inside_keypoint'][part_name].append(1)
                else:
                    loss_dict['inside_keypoint'].append(0)
                    part_loss_dict['inside_keypoint'][part_name].append(0)
                
                # for gt
                if is_inside_box(keypoints, gt_boxes[part_name]):
                    loss_dict['inside_keypoint_gt'].append(1)
                    part_loss_dict['inside_keypoint_gt'][part_name].append(1)
                else:
                    loss_dict['inside_keypoint_gt'].append(0)
                    part_loss_dict['inside_keypoint_gt'][part_name].append(0)
                    
        if visible_only:
            # only evaluate the visible boxes
            if dataset in VIS_DICT:
                visible_boxes = VIS_DICT[dataset][file_name]
                pred_boxes = {part_name: box for part_name, box in item['pred_boxes'].items() if visible_boxes[part_name]}
                gt_boxes = {part_name: box for part_name, box in gt_boxes.items() if visible_boxes[part_name]}
        elif correct_only:
            box_mask = gt_info['box_mask']
            pred_boxes = {part_name: box for i, (part_name, box) in enumerate(item['pred_boxes'].items()) if box_mask[i]}
            gt_boxes = {part_name: box for i, (part_name, box) in enumerate(gt_boxes.items()) if box_mask[i]}
        else:
            pred_boxes = item['pred_boxes']
            gt_boxes = gt_boxes



        assert len(pred_boxes) == len(gt_boxes), f"Number of boxes mismatch for {file_name}"
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        # repeat the image size for each box
        image_size = torch.tensor(image_size).repeat(len(gt_boxes), 1)
        pred_boxes_norm = normalize_boxes(torch.tensor(list(pred_boxes.values())), image_size)
        gt_boxes_norm = normalize_boxes(torch.tensor(list(gt_boxes.values())), image_size)

        giou, iou = compute_giou(pred_boxes_norm, gt_boxes_norm)

        # get the diagonal of the gious
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        giou_loss = 1 - giou
        l1_loss = compute_l1_loss(pred_boxes_norm, gt_boxes_norm)

        loss_dict['giou'].extend(giou_loss.tolist())
        loss_dict['l1'].extend(l1_loss.tolist())
        loss_dict['iou'].extend(iou.tolist())
        if 'pred_scores' in item:
            loss_dict['box_scores'].extend(list(item['pred_scores'].values()))
        loss_dict['pred_boxes'].extend(list(pred_boxes.values()))
        loss_dict['gt_boxes'].extend(list(gt_boxes.values()))

        for part_name, g_loss, l_loss, i_loss in zip(pred_boxes.keys(), giou_loss.tolist(), l1_loss.tolist(), iou.tolist()):
            part_loss_dict['giou'][part_name].append(g_loss)
            part_loss_dict['l1'][part_name].append(l_loss)
            part_loss_dict['iou'][part_name].append(i_loss)
            if 'pred_scores' in item:
                part_loss_dict['box_scores'][part_name].append(item['pred_scores'][part_name])
            part_loss_dict['pred_boxes'][part_name].append(pred_boxes[part_name])
            part_loss_dict['gt_boxes'][part_name].append(gt_boxes[part_name])

    return loss_dict, part_loss_dict


def eval_json(pred_json: str = "",
              teacher_folder: str = "/home/lab/xclip/owlvit_boxes/bird_soup/data_updated_v3",
              gt_json: str = None,
              visible_only: bool = False,
              correct_only: bool = False,
              dataset: str = "cub",
              test_df: str = None, # if provide, only evaluate on the classes in the test df
              ):
    loss_dict, part_loss_dict = compute_box_losses(pred_json, teacher_folder, visible_only, gt_json, dataset, correct_only=correct_only, test_df=test_df)
    
    # Adding overall statistics to part_loss_dict under the name 'all'
    part_loss_dict['giou']['ALL'] = loss_dict['giou']
    part_loss_dict['l1']['ALL'] = loss_dict['l1']
    part_loss_dict['iou']['ALL'] = loss_dict['iou']
    part_loss_dict['box_scores']['ALL'] = loss_dict['box_scores']
    part_loss_dict['inside_keypoint']['ALL'] = loss_dict['inside_keypoint']
    part_loss_dict['inside_keypoint_gt']['ALL'] = loss_dict['inside_keypoint_gt']
    
    # Displaying the combined statistics
    header = f"{'Part':<15} | {'giou':<10} | {'l1':<10} | {'iou':<10} | {'mAP':<10} | {'box precision':<10} | {'gt precision':<10}"
    print(header)
    print("-" * len(header))
    
    avg_giou = []
    avg_l1 = []
    avg_iou = []
    avg_mAP = []
    avg_box_precision = []
    avg_gt_precision = []
    for part, giou_losses in part_loss_dict['giou'].items():
        giou = sum(torch.tensor(giou_losses)) / len(torch.tensor(giou_losses))
        l1 = sum(torch.tensor(part_loss_dict['l1'][part])) / len(torch.tensor(part_loss_dict['l1'][part]))
        iou = sum(torch.tensor(part_loss_dict['iou'][part])) / len(torch.tensor(part_loss_dict['iou'][part]))
        box_precision = sum(torch.tensor(part_loss_dict['inside_keypoint'][part])) / len(torch.tensor(part_loss_dict['inside_keypoint'][part]))
        gt_precision = sum(torch.tensor(part_loss_dict['inside_keypoint_gt'][part])) / len(torch.tensor(part_loss_dict['inside_keypoint_gt'][part]))
        
        # Compute mAP for the part [0.5: 0.95: 0.05]
        part_mean_ap = compute_map_simplified(torch.tensor(part_loss_dict['iou'][part]), np.arange(0.5, 1.0, 0.05))
        
        print(f"{part:<15} | {giou:<10.4f} | {l1:<10.4f} | {iou:<10.4f} | {part_mean_ap:<10.4f} | {box_precision:<10.4f} | {gt_precision:<10.4f}")
        if part != "ALL":
            avg_giou.append(giou)
            avg_l1.append(l1)
            avg_iou.append(iou)
            avg_mAP.append(part_mean_ap)
            avg_box_precision.append(box_precision)
            avg_gt_precision.append(gt_precision)
    print(f"{'12 part mean':<15} | {sum(avg_giou)/len(avg_giou):<10.4f} | {sum(avg_l1)/len(avg_l1):<10.4f} | {sum(avg_iou)/len(avg_iou):<10.4f} | {sum(avg_mAP)/len(avg_mAP):<10.4f} | {sum(avg_box_precision)/len(avg_box_precision):<10.4f} | {sum(avg_gt_precision)/len(avg_gt_precision):<10.4f}")

    print("-" * len(header))

    
if __name__ == '__main__':
    fire.Fire({"eval": eval_json})
    # eval_json("runs/logits/xclip_birdsoupv2/final/owlvit_base_patch32_nabird.json", correct_only=True, dataset='nabird')

# %%
