"""
A meta date file is a pandas dataframe with at least the following columns:
['image_id', 'class_name', 'new_image_name', 'org_image_name', 'class_id', 'abs_path', 'data_source']
With possible additional columns:
['split', 'img_width', 'img_height', 'x_min', 'y_min', 'x_max', 'y_max', 'ann_path', 'is_train', 'wordnet_id', 'wordnet_lemma']

This script is written in a Jupiter notebook style, so it is not a standalone script. It serve as a skeleton for a script that can be used to create a metadata file for a dataset. 
The script is written in a modular way, so that it can be easily modified to suit the specific needs of a dataset.
"""
#%%
# prepare the environment and helper functions
import os
import glob
import shutil
import json

import pandas as pd
from tqdm import tqdm

class CreateMetadata(object):
    @staticmethod
    def get_all_image_paths(data_dir: str, exts: str | list[str] = None):
        if exts is None:
            exts = ['jpg', 'jpeg', 'png', 'gif', 'JPG', 'JPEG', 'PNG', 'GIF']
        all_image_paths = []
        if isinstance(exts, str):
            exts = [exts]
        for ext in exts:
            # Modify the pattern to search recursively in all subdirectories
            pattern = f"{data_dir}/**/*.{ext}"
            all_image_paths.extend(glob.glob(pattern, recursive=True))
        return all_image_paths
    
    @staticmethod
    def get_class_name_from_path(image_path: str, fn: callable) -> str:
        return fn(image_path)
    
    @staticmethod
    def get_parent_folder_name(image_path: str) -> str:
        # Get the folder name one level up from the image path
        return os.path.basename(os.path.dirname(image_path))
    
    @staticmethod
    def txt2list(file_path: str) -> list[str]:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
        

#%%
sd_dog = pd.read_hdf('/home/lab/datasets/stanford_dogs/metadata.h5')
# Create the metadata file for imagenet-21k
target_folder = '/home/lab/datasets/ImageNet21K/dogs'
wn_ids = CreateMetadata.txt2list('/home/peijie/workspace/xclip_ext/peeb/temp_data/imagenet21k_wn_id.txt')
wn_lemma = CreateMetadata.txt2list('/home/peijie/workspace/xclip_ext/peeb/temp_data/imagenet21k_wn_lemma.txt')
wnid2lemma = dict(zip(wn_ids, wn_lemma))
image_paths = CreateMetadata.get_all_image_paths(target_folder, exts='JPEG')
wnid2class_name = json.load(open('/home/peijie/workspace/xclip_ext/peeb/temp_data/imagenet21k_wnid_to_FCI_name_chatgpt.json', 'r'))
sd_dog_classes = json.load(open('/home/peijie/workspace/xclip_ext/peeb/temp_data/SD_to_FCI_name_chatgpt.json', 'r'))
# remove the classes that is a general term
wnid2class_name = {k: v for k, v in wnid2class_name.items() if 'General term' not in v}

sd_fci_names = list(sd_dog_classes.values())
sd_name2fci_name = dict(zip(sd_dog_classes.keys(), sd_fci_names))
fci_name2sd_name = dict(zip(sd_name2fci_name.values(), sd_dog_classes.keys()))
class_names = list(wnid2class_name.values())
all_classes = set(sd_fci_names + class_names)


classname2id = {}
current_max_class_id = sd_dog['class_id'].max()
for class_name in all_classes:
    if class_name not in fci_name2sd_name:
        current_max_class_id += 1
        classname2id[class_name] = current_max_class_id
    else:
        classname2id[class_name] = sd_dog[sd_dog['class_name'] == fci_name2sd_name[class_name]]['class_id'].values[0]


#%%
meta_dicts = []
counter = 0

for image_path in tqdm(image_paths):
    wn_id = CreateMetadata.get_class_name_from_path(image_path, CreateMetadata.get_parent_folder_name)
    if wn_id not in wnid2class_name:
        continue
    meta_dict = {
        'image_id': f'{counter:012d}',
        'wordnet_id': wn_id,
        'wordnet_lemma': json.dumps(wnid2lemma[wn_id]),
        'class_name': wnid2class_name[wn_id],
        'new_image_name': f'{counter:012d}.jpg',
        'org_image_name': os.path.basename(image_path),
        'class_id': classname2id[wnid2class_name[wn_id]],
        'abs_path': image_path,
        'is_train': 1,
        'data_source': 'ImageNet21K',
        
    }
    meta_dicts.append(meta_dict)
    counter += 1
df = pd.DataFrame(meta_dicts)

# %%
max_image_id = len(df)
sd_dog['data_source'] = 'StanfordDogs'
# 1. replace the class_name with the FCI name
# 2. start counting the image_id from the max_image_id
# 3. add 'new_image_name' and 'org_image_name' columns
# 4. add 'abs_path' column
# 5. concat to the df
new_sd_dicts = []
counter = max_image_id
image_root = '/home/lab/datasets/stanford_dogs/images/Images'
for _, row in sd_dog.iterrows():
    new_sd_dict = {
        'image_id': f'{counter:012d}',
        'wordnet_id': row.image_id.split('_')[0],
        'wordnet_lemma': row.class_name,
        'class_name': sd_name2fci_name[row['class_name']],
        'new_image_name': f'{counter:012d}.jpg',
        'org_image_name': row['image_path'],
        'class_id': classname2id[sd_name2fci_name[row['class_name']]],
        'abs_path': os.path.join(image_root, row['image_path']),
        'data_source': 'StanfordDogs',
        'is_train': row.is_train,
    }
    new_sd_dicts.append(new_sd_dict)
    counter += 1
sd_df = pd.DataFrame(new_sd_dicts)

df = pd.concat([df, sd_df], ignore_index=True)
# %%
df_train = df[df.is_train == 1]
# split the metadata file into train and val
from sklearn.model_selection import train_test_split
assert all(df_train['class_id'].value_counts() > 1), "All classes must have at least 2 samples."

# Custom split function
def custom_train_test_split(df, test_size, stratify_column, train_empty_ok=True, val_empty_ok=False):
    unique_classes = df[stratify_column].unique()
    train_idx, val_idx = [], []

    for cls in unique_classes:
        class_samples = df[df[stratify_column] == cls]

        # If there's only one sample in the class
        if len(class_samples) == 1:
            # If val_empty_ok is True, we might decide to not include this class in the validation set
            if val_empty_ok:
                continue  # Skip this class; it will neither be in train nor val set
            else:
                # If val_empty_ok is False, include this single sample in the val set
                val_idx.extend(class_samples.index.tolist())
        else:
            # For classes with 2 or more samples, proceed with the usual train-test split
            class_train, class_val = train_test_split(class_samples, test_size=test_size, stratify=class_samples[[stratify_column]])
            train_idx.extend(class_train.index.tolist())
            val_idx.extend(class_val.index.tolist())

    # Creating the train and validation DataFrames
    train_df = df.loc[train_idx].reset_index(drop=True) if train_idx else pd.DataFrame(columns=df.columns) if train_empty_ok else None
    val_df = df.loc[val_idx].reset_index(drop=True) if val_idx else pd.DataFrame(columns=df.columns) if val_empty_ok else None

    return train_df, val_df

# Use the custom split function
train_df, temp_df = custom_train_test_split(df_train, test_size=0.15, stratify_column='class_id')
val_df, test_df = custom_train_test_split(temp_df, test_size=0.5, stratify_column='class_id', train_empty_ok=False)
# %%
output_dir = '/home/lab/datasets/dogsoup'
# train_df.to_hdf(f'{output_dir}/v1_train.h5', key='dogsoup')
# val_df.to_hdf(f'{output_dir}/v1_val.h5', key='dogsoup')
# test_df.to_hdf(f'{output_dir}/v1_test.h5', key='dogsoup')

v1_class_list = df['class_name'].unique().tolist()
# # save to txt file with one line per class
# with open(f'{output_dir}/dogsoup_v1_class_list.txt', 'w') as file:
#     for class_name in v1_class_list:
#         file.write(f"{class_name}\n")