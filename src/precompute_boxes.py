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
    parser.add_argument('--image_output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--owl_threshold', type=float, help='OwlViT threshold, recommand set to 0, so we can filter later.', default=0.0)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    parser.add_argument('--save_logits', action='store_true', help='Save logits')
    parser.add_argument('--dataset', type=str, help='Dataset name', default="bird_11K")
    parser.add_argument('--logit_dir', type=str, help='Logit directory', default="/home/lab/xclip/owlvit_boxes")
    parser.add_argument("--image_root", type=str, default=None, help="root path to the images.")
    parser.add_argument("--update_logits", action="store_true", help="update logits to the model we use for training")
    parser.add_argument("--metadata_folder", type=str, default=None, help="metadata folder for update logits")
    parser.add_argument("--use_abs_path", action="store_true", help="use absolute path for image")
    parser.add_argument("--subset", type=str, default=None, help="subset of the dataset. If not None, it should be the name of data source or a percentage code (e.g. '0010' for the first 10% \of the data)")

    args = parser.parse_args()

    if args.owl_prompt_type == "cub-12-parts":
        prompts = ["bird", "back", "beak", "belly", "breast", "crown", "forehead", "eyes", "legs", "wings", "nape", "tail", "throat"]
    elif args.owl_prompt_type == "stanforddog-6-parts":
        prompts = ["head", "ears", "muzzle", "body", "legs","tail"]
    elif args.owl_prompt_type == "stanforddog-6-parts-dog":
        prompts = ["dog", "dog head", "dog ears", "dog muzzle", "dog body", "dog legs","dog tail"]

    else:
        raise NotImplementedError(f"Prompt type {args.owl_prompt_type} is not implemented")

    if args.update_logits and args.metadata_folder is None:
        raise ValueError("metadata_folder is required for update logits")

    if not args.update_logits:
        boxes_dir = f"{args.logit_dir}/{args.dataset}/part_boxes/{args.owl_model}_{args.owl_prompt_type}"
    else:
        boxes_dir = f"{args.metadata_folder}_update_logits"

    if args.output_subfix is not None:
        boxes_dir = f"{boxes_dir}_{args.output_subfix}"

    os.makedirs(boxes_dir, exist_ok=True)
    if args.save_img:
        os.makedirs(args.image_output_dir, exist_ok=True)
    device = args.device

    # Load model
    processor = OwlViTProcessor.from_pretrained(f"google/{args.owl_model}")
    model = OwlViTForObjectDetection.from_pretrained(f"google/{args.owl_model}")
    model.to(device)
    model.eval()

    # Load dataset and prepare data loader
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = BirdSoup(root=args.image_root, meta_path=args.meta_path, use_meta_dir=args.use_abs_path, transform=processor, subset=args.subset)
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

            image_ids = [os.path.splitext(image_name)[0] for image_name in soup_names]

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
            # handle last batch
            if len(paths) < args.batch_size:
                text_input_tokens['input_ids'] = text_input_tokens['input_ids'][:len(paths)*len(prompts)]
                text_input_tokens['attention_mask'] = text_input_tokens['attention_mask'][:len(paths)*len(prompts)]
            
            owl_inputs = images | text_input_tokens
            owl_outputs = model(**owl_inputs)
            preds = processor.post_process_object_detection(outputs=owl_outputs, threshold=0, target_sizes=img_sizes.to(device))
            all_boxes = torch.stack([pred['boxes'] for pred in preds]).cpu()
            logits = owl_outputs.logits
            scores_ = torch.sigmoid(logits).cpu()

            top1_scores, top1_idxs = torch.topk(scores_, k=1, dim=1)

            if args.filter_size is not None:
                # the first box is the object
                object_boxes = all_boxes[torch.arange(len(all_boxes)), top1_idxs[:, 0, 0]]
                areas = (object_boxes[:, 2] - object_boxes[:, 0]) * (object_boxes[:, 3] - object_boxes[:, 1])
                # derive the boxes of the object
                object_box_lower_bound = args.filter_size ** 2 
                object_scores = top1_scores[:, 0, 0]
                object_idxs = top1_idxs[:, 0, 0]

                # filter out object boxes that are too small
                object_boxes = all_boxes[torch.arange(len(all_boxes)), object_idxs]

                score_keep = object_scores > args.owl_threshold
                box_keep = areas > object_box_lower_bound
                keep = score_keep & box_keep & torch.tensor(keep_missing)

                # drop the first box (object/target box)
                part_scores = top1_scores[:, 0, 1:]
                part_idxs = top1_idxs[:, 0, 1:]
                part_logits = scores_[keep][:, :, 1:].cpu()
                part_names = prompts[1:]
                object_boxes = object_boxes[keep]
                object_scores = object_scores[keep]

            else:
                keep = [True] * len(all_boxes)
                part_scores = top1_scores
                part_idxs = top1_idxs
                part_logits = scores_[keep].cpu()
                object_boxes = [None]*len(all_boxes)
                object_scores = [None]*len(all_boxes)
                part_names = prompts

            if args.owl_prompt_type == "stanforddog-6-parts-dog":
                part_names = [part_name.replace("dog ", "") for part_name in part_names]
            else:
                part_names = prompts

            # filter by keep
            part_scores = part_scores[keep]
            if len(part_idxs.shape) == 3:
                part_idxs = part_idxs[keep].squeeze(1)
            else:
                part_idxs = part_idxs[keep]


            # choose boxes per query and store class-wise per image
            for i, (boxes, part_idx, part_score, part_logit, object_box, object_score, image_name) in enumerate(zip(all_boxes, part_idxs, part_scores, part_logits, object_boxes, object_scores, soup_names)):

                output_image_path = os.path.join(args.image_output_dir, image_name) if args.save_img else paths[i]

                part_boxes = boxes[part_idx]
                assert len(part_boxes) == len(part_names)
                part_boxes = bound_boxes(boxes=part_boxes, img_sizes=img_sizes[i].repeat(len(part_boxes), 1))

                img_pred_dict = {
                    part_name: box.cpu().tolist()
                    for part_name, box in zip(part_names, part_boxes)
                }

                if args.update_logits:
                    output_dict = torch.load(f"{args.metadata_folder}/{image_ids[i]}.pth")
                    output_dict["part_logits"] = part_logit

                else:
                    output_dict = {"image_id": image_ids[i],
                                "image_path": output_image_path,
                                "org_image_path": paths[i],
                                "image_size": img_sizes[i].tolist()[::-1], # in (w, h)
                                "class_name": idx2name[class_labels[i].item()],
                                "boxes_info": img_pred_dict, # previous key should be 'part_boxes', keep it for consistency for Thang's code
                                "part_scores": part_score.tolist(),
                                "part_logits": part_logit if args.save_logits else None,}

                    if args.filter_size is not None:
                        output_dict |= {
                            "object_boxes": object_box.tolist(),
                            "object_score": object_score.item(),
                            "object_areas": areas[i].item(),
                            "object_ratio": (
                                areas[i].item()
                                / (img_sizes[i][0] * img_sizes[i][1])
                            ).item(),
                        }
                torch.save(output_dict, f"{boxes_dir}/{image_ids[i]}.pth")

                if args.save_img:
                    shutil.copy(paths[i], output_image_path)



