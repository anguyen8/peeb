import os
import copy
import shutil
import argparse
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.transforms import Resize, InterpolationMode, ToPILImage

from data_loader import BirdSoup


def xy_in_box(box: list[int], x: int, y: int):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def check_exist(image_ids: list[str], box_dir: str, ext: str = ".pth") -> tuple[bool, list[str]]:
    missing_ids = [
        image_id
        for image_id in image_ids
        if not os.path.exists(os.path.join(box_dir, image_id + ext))
    ]
    all_exist = not missing_ids
    return all_exist, missing_ids


def bound_boxes(boxes: torch.Tensor, img_sizes: torch.Tensor):
    # note: img_sizes are in (h, w) format
    
    # Create a tensor of boundaries based on image sizes
    boundaries = torch.cat((torch.zeros_like(img_sizes), img_sizes.flip(1)), dim=1)
    # Bounding the boxes
    bounded_boxes = torch.zeros_like(boxes)
    bounded_boxes[:, 0] = torch.clamp(boxes[:, 0], boundaries[:, 0], boundaries[:, 2])
    bounded_boxes[:, 1] = torch.clamp(boxes[:, 1], boundaries[:, 1], boundaries[:, 3])
    bounded_boxes[:, 2] = torch.clamp(boxes[:, 2], boundaries[:, 0], boundaries[:, 2])
    bounded_boxes[:, 3] = torch.clamp(boxes[:, 3], boundaries[:, 1], boundaries[:, 3])

    return bounded_boxes.long()


# to return PIL images
def custom_collate_fn(batch):
    images, targets, paths, image_sizes = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.long)
    images_sizes = torch.stack(image_sizes)
    return images, targets, paths, images_sizes


# sourcery skip: assign-if-exp, swap-if-expression
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # OwlViT settings
    parser.add_argument('--owl_model', help='select owl-vit version (huggingface)', default="owlvit-large-patch14", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
    parser.add_argument('--owl_prompt_type', type=str, help='Prompt type for OwlViT', default="cub-12-parts")

    # Misc
    parser.add_argument('--meta_path', type=str, help='Path to meta file', default="../data/bird_11K/metadata/bird_soup_uncased_v2.h5")
    parser.add_argument('--batch_size', type=int, help='num batch size', default=12)
    parser.add_argument('--num_workers', type=int, help='num workers for batch processing', default=8)
    parser.add_argument('--device', help='select device', default="cuda:0", type=str)
    parser.add_argument('--overwrite', help='overwrite existing boxes results', action='store_true')
    parser.add_argument('--output_subfix', type=str, help='output subfix', default=None)
    parser.add_argument('--filter_size', type=int, default=None, help='Filter size for bird, default is None (no filter)')
    parser.add_argument('--save_img', action='store_true', help='Save the image to a new folder')
    parser.add_argument('--image_output_dir', type=str, help='Output directory', default="../data/bird_11K/images")
    parser.add_argument('--owl_threshold', type=float, help='OwlViT threshold, recommand set to 0, so we can filter later.', default=0.0)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    parser.add_argument('--save_logits', action='store_true', help='Save logits')

    args = parser.parse_args()

    if args.owl_prompt_type == "cub-12-parts":
        prompts = ["bird", "back", "beak", "belly", "breast", "crown", "forehead", "eyes", "legs", "wings", "nape", "tail", "throat"]
    else:
        raise NotImplementedError(f"Prompt type {args.owl_prompt_type} is not implemented")

    boxes_dir = f"../data/bird_11K/part_boxes/{args.owl_model}_{args.owl_prompt_type}"
    if args.output_subfix is not None:
        boxes_dir = f"{boxes_dir}_{args.output_subfix}"

    os.makedirs(boxes_dir, exist_ok=True)
    os.makedirs(args.image_output_dir, exist_ok=True)
    device = args.device

    # Load model
    processor = OwlViTProcessor.from_pretrained(f"google/{args.owl_model}")
    model = OwlViTForObjectDetection.from_pretrained(f"google/{args.owl_model}")
    model.to(device)
    model.eval()

    # Load dataset and prepare data loader
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = BirdSoup(root=None, meta_path=args.meta_path, use_meta_dir=True, transform=processor)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)

    # Prepare text embeddings
    idx2name = dataset.idx2class

    with torch.no_grad():
        # tokenize descs
        text_input_tokens = processor(prompts, padding="max_length", truncation=True, return_tensors="pt")
        text_input_tokens['input_ids'] = text_input_tokens['input_ids'].repeat(args.batch_size, 1)
        text_input_tokens['attention_mask'] = text_input_tokens['attention_mask'].repeat(args.batch_size, 1)
        text_input_tokens = text_input_tokens.to(device)

        for batch_idx, batch in tqdm(enumerate(dataloader), desc='Computing boxes', total=len(dataloader)):
            images, class_labels, img_sizes, paths, soup_names = batch

            image_ids = []
            output_image_paths = []
            for image_name in soup_names:
                image_ids.append(os.path.splitext(image_name)[0])
                output_image_paths.append(os.path.join(args.image_output_dir, image_name))

            # check if boxes already exist
            if not args.overwrite:
                all_exist, missing_ids = check_exist(image_ids, boxes_dir)
                if all_exist:
                    continue  
                else:
                    keep_missing = [id in missing_ids for id in image_ids]
            else:
                keep_missing = [True] * len(image_ids)

            # forward to get owl-vit outputs
            images['pixel_values'] = images['pixel_values'].squeeze(1).to(device)
            owl_inputs = images | text_input_tokens
            owl_outputs = model(**owl_inputs)
            preds = processor.post_process_object_detection(outputs=owl_outputs, threshold=0, target_sizes=img_sizes.to(device))
            all_boxes = torch.stack([pred['boxes'] for pred in preds]).cpu()
            logits = owl_outputs.logits
            scores_ = torch.sigmoid(logits).cpu()

            top1_scores, top1_idxs = torch.topk(scores_, k=1, dim=1)

            # derive the boxes of the bird
            bird_box_lower_bound = args.filter_size ** 2 if args.filter_size is not None else 0
            bird_scores = top1_scores[:, 0, 0]
            bird_idxs = top1_idxs[:, 0, 0]

            # filter out bird boxes that are too small
            bird_boxes = all_boxes[torch.arange(len(all_boxes)), bird_idxs]

            # calculate the area of the bird boxes, and remove birds that are too small
            areas = (bird_boxes[:, 2] - bird_boxes[:, 0]) * (bird_boxes[:, 3] - bird_boxes[:, 1])

            if args.filter_size is not None:
                score_keep = bird_scores > args.owl_threshold
                box_keep = areas > bird_box_lower_bound
                keep = score_keep & box_keep & torch.tensor(keep_missing)
            else:
                keep = [True] * len(all_boxes)

            part_scores = top1_scores[:, 0, 1:]
            part_idxs = top1_idxs[:, 0, 1:]

            # filter by keep
            part_scores = part_scores[keep]
            part_idxs = part_idxs[keep]
            bird_scores = bird_scores[keep]
            bird_boxes = bird_boxes[keep]
            part_logits = scores_[keep][:, :, 1:].cpu()

            # choose boxes per query and store class-wise per image
            for i, (boxes, part_idx, part_score, part_logit, bird_box, bird_score) in enumerate(zip(all_boxes, part_idxs, part_scores, part_logits, bird_boxes, bird_scores)):
                
                part_boxes = boxes[part_idx]
                part_boxes = bound_boxes(boxes=part_boxes, img_sizes=img_sizes[i].repeat(len(part_boxes), 1))

                img_pred_dict = {
                    part_name: box.cpu().tolist()
                    for part_name, box in zip(prompts[1:], part_boxes)
                }

                torch.save({"image_id": image_ids[i],
                            "image_path": output_image_paths[i],
                            "org_image_path": paths[i],
                            "image_size": img_sizes[i].tolist()[::-1], # in (w, h)
                            "class_name": idx2name[class_labels[i].item()],
                            "boxes_info": img_pred_dict, # previous key should be 'part_boxes', keep it for consistency for Thang's code
                            "part_scores": part_score.tolist(),
                            "part_logits": part_logit if args.save_logits else None,
                            
                            "bird_boxes": bird_box.tolist(),
                            "bird_score": bird_score.item(),
                            "bird_areas": areas[i].item(),
                            "bird_ratio": (areas[i].item() / (img_sizes[i][0] * img_sizes[i][1])).item(),
                            }, f"{boxes_dir}/{image_ids[i]}.pth")
                
                if args.save_img:
                    shutil.copy(paths[i], output_image_paths[i])



