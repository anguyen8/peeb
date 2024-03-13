import os
import json
import random
import warnings

import clip
import fire
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.prompt_engineering import GetPromptList
from train_owlvit import load_training_dataset
from utils import closest_string

def encode_descs(model: callable, descs: list[str], device: str, max_batch_size: int = 512):
    total_num_batches = len(descs) // max_batch_size + 1
    with torch.no_grad():
        text_embeds = []
        for batch_idx in range(total_num_batches):
            desc = descs[batch_idx*max_batch_size:(batch_idx+1)*max_batch_size]
            query_tokens = clip.tokenize(desc).to(device)
            text_embeds.append(model.encode_text(query_tokens).cpu().float())
    text_embeds = torch.cat(text_embeds, dim=0)
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
    return text_embeds
            

def sachit_eval(device: str, 
                model: callable, 
                class_list: list[str], 
                clip_descs: list[str], 
                sachit_descs: list[str], 
                sachit_desc_mapper: list[int], 
                dataloader: DataLoader, 
                is_birdsoup: bool = False,
                save_logits: bool = False,
                out_folder: str = None,
                descriptors: dict = None,
                dataset_name: str = None,
                comm2sci: dict = None,
                if_sci: bool = False,
                
                ):
    model.to(device)
    # # check if the order of class name is the same
    # dataset_name_list = list(dataset.class2idx.keys())
    # for desc_class_name, dataset_class_name in zip(class_list, dataset_name_list):
    #     if desc_class_name != dataset_class_name:
    #         raise ValueError(f"Class name mismatch: {desc_class_name} vs {dataset_class_name}")

    # create a sachit descriptor mask with shape (batch_size, n_classes*12):
    # 1 if sachit_desc_mapper is not -1, 0 otherwise
    sachit_mask = torch.tensor(sachit_desc_mapper) # convert to tensor
    sachit_mask = torch.where(sachit_mask == -1, 0, 1).unsqueeze(0) # convert -1 to 0, (1, n_classes*12)
    # get the number of descriptors for each class
    sachit_num_cls_desc = sachit_mask.view(1, -1, 12).sum(dim=-1) # (1, n_classes)
    
    # encode text embeddings
    clip_text_embeds = encode_descs(model, clip_descs, device)
    sachit_text_embeds = encode_descs(model, sachit_descs, device)
    
    with torch.no_grad():

        pred_clip = []
        pred_sachit = []
        pred_list = []
        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            if is_birdsoup:
                sample, target, size, image_path, soup_name = batch
            else:
                sample, target, image_path, sample_size = batch
            # img_tensor = preprocess(sample).unsqueeze(0).to(device)
            image_embed = model.encode_image(sample.to(device))
            image_embed = torch.nn.functional.normalize(image_embed, dim=-1).cpu().float()
            
            # clip prediction
            clip_sim = torch.matmul(image_embed, clip_text_embeds.T) # (batch_size, n_classes)
            clip_pred = torch.argmax(clip_sim, dim=-1)
            
            # sachit prediction
            batch_size = len(target)
            sachit_mask_batch = torch.repeat_interleave(sachit_mask, repeats=batch_size, dim=0) # repeat for each batch
            sachit_sim = torch.matmul(image_embed, sachit_text_embeds.T) # (batch_size, n_classes*12)
            sachit_sim_logits = (sachit_sim * sachit_mask_batch).view(batch_size, -1, 12) # (batch_size, n_classes, 12)
            sachit_sim = sachit_sim_logits.sum(dim=-1) / sachit_num_cls_desc # (batch_size, n_classes)
            sachit_pred = torch.argmax(sachit_sim, dim=-1)
            sachit_softmax = torch.nn.functional.softmax(sachit_sim, dim=-1)
            # sachit_softmax = torch.nn.functional.gumbel_softmax(sachit_sim, dim=-1, tau=0.5)
            sachit_softmax_top1 = torch.topk(sachit_softmax, k=1, dim=-1)[0].squeeze(-1)
            
            # get pred
            pred_clip.append(clip_pred == target)
            sachit_preds = sachit_pred == target
            pred_sachit.append(sachit_preds)
            
            if save_logits:
                for items in zip(image_path, target, clip_pred, clip_sim, sachit_pred, sachit_sim_logits, sachit_preds, sachit_softmax_top1):
                    image_file, class_id, clip_pred_id, clip_sim, sachit_pred_id, sachit_sim, sachit_is_correct, softmax_top1 = items
                    file_name = os.path.basename(items[0])
                    gt_name = dataloader.dataset.idx2class[class_id.item()]
                    # clip_pred_name = dataset.dataset.idx2class[clip_pred_id]
                    sachit_pred_name = dataloader.dataset.idx2class[sachit_pred_id.item()]
                    sachit_scores = sachit_sim[sachit_pred_id.item()]
                    # remove zeros in sachit_sim
                    sachit_scores = sachit_scores[sachit_scores.nonzero()].view(-1).tolist()
                    descriptions = descriptors[sachit_pred_name]
                    
                    if "birdsoup" in dataset_name:
                        bs_file_name = file_name
                        org_file_name = dataloader.dataset.meta_df[dataloader.dataset.meta_df['new_image_name'] == bs_file_name]['abs_path'].values[0]
                        org_file_name = os.path.basename(org_file_name)
                    else:
                        org_file_name = file_name
                        bs_file_name = ""
                    if len(sachit_scores) != len(descriptions):
                        warnings.warn(f"Number of descriptions and scores mismatch: {len(sachit_scores)} vs {len(descriptions)}")
                    
                    if if_sci:
                        gt_name = comm2sci[gt_name]
                        sachit_pred_name = comm2sci[sachit_pred_name]
                    pred_list.append({'file_name': org_file_name, 'bs_file_name': bs_file_name, 'prediction': sachit_is_correct.item(), 'ground_truth': gt_name, 'pred': sachit_pred_name, 'scores': sachit_scores, 'descriptions': descriptions, 'final_score': sum(sachit_scores), 'softmax_score': softmax_top1.item()})
                    
            if save_logits and out_folder is not None:
                json.dump(pred_list, open(os.path.join(out_folder, 'sachit_logits.json'), 'w'), indent=4)
            
    return torch.cat(pred_clip).float().mean().item(), torch.cat(pred_sachit).float().mean().item()


def get_class_names_in_descriptors(dataset_name: str, class_names: list[str], sachit_desc_file: str) -> dict:
    comm2sci = json.load(open(f"/home/peijie/workspace/xclip_ext/birdsoup/sci_mapping/{dataset_name.replace('birdsoup_', '')}_comm2sci.json", 'r'))
    if 'birdsoup' in dataset_name:
        # case sensitive names
        descriptor_names = list(json.load(open(sachit_desc_file, 'r')).keys())
        # common names
        if 'inaturalist' in dataset_name:
            sci2comm = json.load(open(f"/home/peijie/workspace/xclip_ext/birdsoup/sci_mapping/{dataset_name.replace('birdsoup_', '')}_sci2comm.json", 'r'))
            comm2sci = {v: k for k, v in sci2comm.items()}
            case_sensitive_names = [sci2comm[cls_name] for cls_name in descriptor_names]
        else:
            case_sensitive_names = descriptor_names
        # uncased common names
        comm_names_uncased = [comm_name.lower().replace("'s", "").replace("-", " ").replace("_", " ") for comm_name in case_sensitive_names]
        # check if names are matched with class_names, if not, use levenshtein distance to find the closest match (if distance <= 1)
        # Mostly due to inconsistent naming in different datasets of using single quatation mark or no quatation mark. e.g., Forster's tern vs Forsters tern
        all_matched = all(comm_name in class_names for comm_name in comm_names_uncased) # all class_names in birdsoup should be in lower case
        if not all_matched:
            comm_names_mapping = {}
            failed_names = []
            for comm_name in comm_names_uncased:
                if comm_name in class_names:
                    comm_names_mapping[comm_name] = comm_name
                else:
                    # find the closest match
                    closest_str, min_dist = closest_string(comm_name, class_names)
                    if min_dist <= 1:
                        comm_names_mapping[comm_name] = closest_str
                    else:
                        comm_names_mapping[comm_name] = comm_name
                        failed_names.append(comm_name)
            if failed_names:            
                warnings.warn(f"Failed to find the closest match for the following names: {failed_names}")
                # only evaluate on a subset of all classes
                if len(comm_names_mapping) < len(class_names):
                    warnings.warn(f"Only evaluate on a subset of all classes: {len(comm_names_mapping)} vs {len(class_names)}")
                    
        # dict: uncased common name -> original common name
        # uncased_name2org_name = dict(zip(comm_names_uncased, comm_names))
        uncased_name2org_name = {comm_names_mapping[uncased_name]: org_name for uncased_name, org_name in zip(comm_names_uncased, case_sensitive_names) if uncased_name in comm_names_mapping}
        # uncased name to original name mapping in descriptor's order
        class_name_in_descs = [uncased_name2org_name[comm_name] for comm_name in comm_names_uncased if comm_name in uncased_name2org_name]
        # convert to scientific name if 'inaturalist' in dataset_name
        if 'inaturalist' in dataset_name:
            class_name_in_descs = [comm2sci[cls_name] for cls_name in class_name_in_descs]
        return class_names, class_name_in_descs, comm2sci, uncased_name2org_name
    else:
        return class_names, class_names, comm2sci, dict(zip(class_names, class_names))


def eval_sachit_wrapper(dataset_name: str = 'birdsoup_cub',
                        device: str = 'cuda:4',
                        scientific_name: bool = False,
                        save_logits: bool = True,
                        out_folder: str = 'results/sachit_eval',
                        batch_size: int = 32,
                        descriptor_path: str = None, # Use: data/class_lists/descriptors_cub_sachit_gpt4_random_s42.json for random descriptors
                        no_class_names: bool = False,
                        zeroshot_split: bool = False,
                        is_birdsoup: bool = False,
                        ):
    os.makedirs(out_folder, exist_ok=True)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = load_training_dataset(dataset_name=dataset_name,
                                    sub_dataset_names=None,
                                    eval_size=0.2,
                                    transform=preprocess,
                                    split='test',
                                    zeroshot_split=zeroshot_split)
    class_names = list(dataset.class2idx.keys())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # get class names in descriptors
    birdsoup_classes, class_name_in_descs, comm2sci, uncased_name2org_name = get_class_names_in_descriptors(dataset_name, class_names, descriptor_path)
    name2idx = {uncased_name2org_name[name]: idx for name, idx in dataset.class2idx.items()}

    getprompt = GetPromptList(descriptor_path, class_names=class_name_in_descs, name2idx=name2idx)
    sachit_descs, class_idxs, class_mapping, org_desc_mapper, class_list = getprompt('Sachit-descriptors', max_len=12, pad=True)
    clip_descs = list(getprompt.name2idx.keys())

    # for saving logits
    if 'birdsoup' in dataset_name:
        if 'inaturalist' in dataset_name:
            bs_descriptors = {class_name: getprompt.desc[comm2sci[uncased_name2org_name[class_name]]] for class_name in birdsoup_classes}
        else:
            bs_descriptors = {class_name: getprompt.desc[uncased_name2org_name[class_name]] for class_name in birdsoup_classes}
        bs_comm2sci = {class_name: comm2sci[uncased_name2org_name[class_name]] for class_name in birdsoup_classes if class_name in uncased_name2org_name} if scientific_name else None
        
    # convert to scientific name if scientific_name is True
    if scientific_name:
        if dataset_name not in {'inaturalist', 'birdsoup_inaturalist'}:
            # comm2sci = json.load(open(f"/home/peijie/workspace/xclip_ext/birdsoup/sci_mapping/{dataset_name.replace('birdsoup_', '')}_comm2sci.json", 'r'))
            sachit_descs = [desc.replace(cls_name, comm2sci[cls_name]) for cls_name, desc in zip([cls for cls in class_list for _ in range(12)], sachit_descs)]
            clip_descs = [comm2sci[desc] for desc in clip_descs]
            class_mapping = {k: comm2sci[v] for k, v in class_mapping.items()}
    # inaturalist by default is in scientific name
    else:
        if dataset_name in {'inaturalist', 'birdsoup_inaturalist'}:
            sci2comm = json.load(open(f"/home/peijie/workspace/xclip_ext/birdsoup/sci_mapping/{dataset_name.replace('birdsoup_', '')}_sci2comm.json", 'r'))
            sachit_descs = [desc.replace(cls_name, sci2comm[cls_name]) for cls_name, desc in zip([cls for cls in class_list for _ in range(12)], sachit_descs)]
            clip_descs = [sci2comm[desc] for desc in clip_descs]

    print(f"CLIP descriptors: {clip_descs[0]}")
    if no_class_names:
        sachit_descs = [desc.replace(cls_name, "") for cls_name, desc in zip([cls for cls in class_list for _ in range(12)], sachit_descs)]

    clip_acc, sachit_acc = sachit_eval(device=device,
                                        model=clip_model,
                                        class_list=class_list if 'birdsoup' not in dataset_name else birdsoup_classes,
                                        clip_descs=clip_descs,
                                        sachit_descs=sachit_descs,
                                        sachit_desc_mapper=class_idxs,
                                        dataloader=dataloader,
                                        is_birdsoup=is_birdsoup,
                                        save_logits=save_logits,
                                        out_folder=out_folder,
                                        descriptors=getprompt.desc if 'birdsoup' not in dataset_name else bs_descriptors,
                                        dataset_name=dataset_name,
                                        comm2sci=comm2sci if 'birdsoup' not in dataset_name else bs_comm2sci,
                                        if_sci=scientific_name,
                                        )

    print(f"CLIP accuracy: {clip_acc:.4f}")
    print(f"Sachit accuracy: {sachit_acc:.4f}")

#%%

if __name__ == '__main__':
    fire.Fire({"eval": eval_sachit_wrapper})