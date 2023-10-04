import os
import shutil
import argparse

import pandas as pd
from tqdm import tqdm


def get_arg():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--meta_path", type=str)
    argparser.add_argument("--img_path", type=str)
    
    return argparser.parse_args()

def rename_images(meta_path: str, img_path: str, inplace: bool = True):
    if not inplace:
        # add suffix to image folder
        out_image_path = f"{img_path}_birdsoup"
    else:
        out_image_path = img_path
    os.makedirs(out_image_path, exist_ok=True)    
        
    meta = pd.read_hdf(meta_path)

    missing_images = []
    for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Rename images"):
        org_name = row["org_image_name"]
        new_image_name = row["image_name"]

        org_image_path = os.path.join(img_path, org_name)
        new_image_path = os.path.join(out_image_path, new_image_name)
        # skip and record missing images
        if not os.path.exists(org_image_path):
            data_source = row["data_source"]
            missing_images.append((org_image_path, data_source))
            continue
        # skip if image already exists (in case we want to run this script multiple times)
        if os.path.exists(new_image_path):
            continue
        
        shutil.move(org_image_path, new_image_path)
        
if __name__ == "__main__":
    args = get_arg()
    rename_images(args.meta_path, args.img_path, inplace=True)