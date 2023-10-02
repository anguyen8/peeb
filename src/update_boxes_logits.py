import argparse
import os
from datetime import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from data_loader import DatasetWrapper
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------
    #   Must-check arguments for experiments but usually FIXED
    # ------------------------------------------------------------
    parser.add_argument('--model', help='select model', default="owlvit-large-patch14", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
    parser.add_argument('--dataset', help='select dataset', default="imagenet", choices=["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "places365", "cub", "nabirds", "bird_soup"])
    parser.add_argument('--distortion', help='select distortion type if using ImageNet-C', default="defocus_blur", choices=["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "shot_noise", "gaussian_noise", "impulse_noise"])
    parser.add_argument('--distortion_severity', type=int, help='select distortion severity if using ImageNet-C', default=1, choices=[1, 2, 3, 4, 5])

    parser.add_argument('--batch_size', type=int, help='num batch size', default=32)
    parser.add_argument('--num_workers', type=int, help='num workers for batch processing', default=16)
    parser.add_argument('--num_samples', type=int, help='num images per class', default=-1)
    parser.add_argument('--device', help='select device', default="cuda:0", type=str)
    parser.add_argument('--random_seed', help='random seed (for data subsampling only)', default=42, type=int)
    parser.add_argument('--image_type_cub', help='select image type for CUB only', type=str, default=None, choices=["synthetic", "upsampler"])
    parser.add_argument('--box_type_cub', help='select pred_box type for CUB only', type=str, default=None, choices=["with_bird_box", "synthetic", "upsampler"])

    parser.add_argument('--descriptors', help='select descriptors for OwlViT', default="sachit", choices=["sachit", "chatgpt"])
    parser.add_argument('--prompt_type', type=int, help='select prompt type', default=5)
    parser.add_argument('--owlvit_threshold', type=float, help='select threshold for owl_vit', default=-1)

    args = parser.parse_args()

    start_time = datetime.now()

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        check_device_availability(args.device)
        device = args.device

    boxes_dir = f"{PRECOMPUTED_DIR}/{args.dataset}/data"

    updated_boxes_dir = boxes_dir + "_updated_v2"
    if not os.path.exists(updated_boxes_dir):
        os.makedirs(updated_boxes_dir)

    if args.dataset == "cub":
        boxes_dir += "" if args.box_type_cub is None else f"_{args.box_type_cub}"
    elif args.dataset == "imagenet-c":
        boxes_dir = f"{boxes_dir}/distortion_{args.distortion}_severity_level_{args.distortion_severity}"

    # Prepare text embeddings
    descriptions_only, _ = load_descriptions(dataset_name=args.dataset, prompt_type=0, desc_type=args.descriptors)

    # Remove 'It has' and 'It is' from Sachit's CUB descriptors
    if args.dataset == "cub":
        descriptions_only = {key: [value.replace("It has", "").replace("It is", "").strip() for value in values] for key, values in descriptions_only.items()}

    class_list = list(descriptions_only.keys())
    all_descriptions = list(descriptions_only.values())
    if args.descriptors == "chatgpt":
        all_descriptions = [[descriptor.split(":")[0] for descriptor in descriptors if ":" in descriptor] for descriptors in descriptions_only.values()][0]

    processor = OwlViTProcessor.from_pretrained(f"google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained(f"google/owlvit-base-patch32")
    model.to(device)
    model.eval()

    # Load dataset and prepare data loader
    dataset_args = {"image_type_cub": args.image_type_cub}
    dataset = DatasetWrapper(dataset_name=args.dataset, transform=processor,
                             distortion=f"{args.distortion}:{args.distortion_severity}",
                             samples_per_class=args.num_samples, random_seed=args.random_seed, **dataset_args)

    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():
        text_inputs = processor(text=all_descriptions, padding="max_length", truncation=True, return_tensors="pt").to(device)
        total_descriptors = text_inputs['input_ids'].shape[0]
        text_inputs['input_ids'] = text_inputs['input_ids'].repeat(args.batch_size, 1)
        text_inputs['attention_mask'] = text_inputs['attention_mask'].repeat(args.batch_size, 1)

    for batch_idx, batch in tqdm(enumerate(dataloader), desc='Localizing descriptors', total=len(dataloader)):
        images, gt_labels, image_paths, image_ids, image_sizes = batch

        # Handle the last batch separately
        if batch_idx == len(dataloader) - 1:
            text_inputs['input_ids'] = text_inputs['input_ids'][:len(image_paths) * total_descriptors]
            text_inputs['attention_mask'] = text_inputs['attention_mask'][:len(image_paths) * total_descriptors]

        images['pixel_values'] = images['pixel_values'].squeeze(1).to(device)
        with torch.no_grad():
            owl_inputs = images | text_inputs
            owl_outputs = model(**owl_inputs)

            logits = torch.sigmoid(owl_outputs.logits).detach().cpu()

            for i, logit_scores in enumerate(logits):
                results = torch.load(f"{boxes_dir}/{image_ids[i]}.pth")

                # Do not overwrite if already exists
                if 'part_logits_owlvit_base' in results:
                    continue

                results['part_logits_owlvit_base'] = logit_scores
                torch.save(results, f"{updated_boxes_dir}/{image_ids[i]}.pth")



