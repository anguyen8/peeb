# xclip

### Step 1: Data preparation

#### 1. Create a folder to store the data and set the `DATA_PATH` variable to this folder.
```bash
export DATA_PATH = PATH_TO_YOUR_DATA_FOLDER
```

#### 2. Download all the following files from [Box.com](https://auburn.app.box.com/s/owiwf73yxurz3r2k2i6x0r2cg9mgglcc) and store them in the `DATA_PATH` folder.
- birdsoup_* (`00` to `05`)
- inat_filtered_images.tar.gz
- boxes_no_logits.tar.gz
- metadata.tar.gz
- checksums_birdsoup_splits.txt

#### 3. Change directory and merge the splits into two files by running the following commands:
```bash
cd ${DATA_PATH}
sha256sum -c checksums_birdsoup_splits.txt
```
Once all the files check is okay, concat them with:
```
cat birdsoup_* > birdsoup.tar.gz
```
Otherwise, re-download the file that fails the check.

* Extract the data and rearrange sub-folders by running the following commands:
```bash
tar -xvf birdsoup.tar.gz
tar -xvf boxes_no_logits.tar.gz
tar -xvf inat_filtered_images.tar.gz

rm -r bird_soup/metadata                  # Remove the old metadata folder
tar -xvf metadata.tar.gz -C bird_soup/    # Use the latest metadata folder

# Move 42,691 missing images to the images folder to make it 440,934 in total
ls birdsoup_v2_missing_images/ | xargs -I {} mv birdsoup_v2_missing_images/{} bird_soup/images/   
```

After this step, the data folder should look like this:
```bash
DATA_PATH
└── bird_soup
    ├── data (total: 440,934)
    │   ├── 000000000000.pth
    │   ├── 000000000001.pth
    │   ├── ...
    ├── data_updated_v2 (total: 440,934)  # This folder is created after running update_boxes_logits.py in step 6
    │   ├── 000000000000.pth
    │   ├── 000000000001.pth
    │   ├── ...
    ├── images (total: 440,934)
    │   ├── 000000000000.jpg
    │   ├── 000000000001.jpg
    │   ├── ...
    └── metadata
        ├── bird_soup_uncased_v2.h5
        ├── level_1_exclude_cub_nabirds_inat
        │       ├── ...
        ├── level_3_exclude_cub
        │       ├── ...
        ├── level_3_exclude_cub_50_clore
        │       ├── ...
        ├── level_3_exclude_inaturalist
        │       ├── ...
        └── level_3_exclude_nabirds
                ├── train_keep_child_a100.h5
                └── ...
```

### Step 2: Clone the repository and change directory

```bash
git clone https://github.com/ThangPM/xclip && cd xclip
```

### Step 3: Set up the environment

```bash
conda create -n xclip python=3.9
conda activate xclip

pip install -r requirements.txt
conda install pycocotools -c conda-forge
python -m spacy download en_core_web_sm
```

### Step 4: Run experiments
* Before running experiments, we need to export the `PYTHONPATH` variable to the current directory, its sub-directories and change directory to the `src` folder as follows.
```bash
# Assume the current directory is xclip
export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/mmdetection
cd src/
```

### Step 5: Set up the file path in `config.py` file
Make the following changes to `src/config.py`:
1. Set `DATASET_DIR` to `{DATA_PATH}`
2. Set `PRECOMPUTED_DIR` to `{DATA_PATH}`


### Step 6: Compute teacher logits
Before the training, we need to compute the teacher logits. It takes around 60-90 minutes.
```bash
CUDA_VISIBLE_DEVICES=0 python update_boxes_logits.py --model "owlvit-large-patch14" --dataset "bird_soup" --descriptors "chatgpt" --batch_size 32 --num_workers 16 --prompt_type 0 --owlvit_threshold -1
```

### Step 7: Run experiments

#### Classification approach

1. Cross-entropy loss
 
```bash
torchrun --nproc_per_node=8 --rdzv-endpoint=0.0.0.0:29500 train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --save_freq 1 --num_workers 16 --devices 0,1,2,3,4,5,6,7 --epochs 32 --lr 0.00002 --project_name xclip_stage1_classification --loss_weights 0,0,0,0,1 --network_type classification --classification_loss ce_loss --freeze_box_heads --logits_from_teacher --fold 1 --early_stopping 5
```

2. Focal loss

```bash
torchrun --nproc_per_node=8 --rdzv-endpoint=0.0.0.0:29600 train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --save_freq 1 --num_workers 16 --devices 0,1,2,3,4,5,6,7 --epochs 32 --lr 0.00002 --project_name xclip_stage1_classification --loss_weights 0,0,0,0,1 --network_type classification --classification_loss focal_loss --freeze_box_heads --logits_from_teacher --fold 1 --early_stopping 5
```

#### Contrastive approach

1. Random sampler

```bash
torchrun --nproc_per_node=2 train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 48 --batch_size_val 50 --save_freq 1 --num_workers 16 --devices 2,3 --epochs 64 --lr 0.0002 --project_name xclip_stage1_contrastive --loss_weights 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 48 --num_negatives_val 50 --fold 1 --early_stopping 5
```

2. Removed sampler

```bash
torchrun --nproc_per_node=2 --rdzv-endpoint=0.0.0.0:29600 train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 48 --batch_size_val 50 --save_freq 1 --num_workers 16 --devices 4,5 --epochs 64 --lr 0.0002 --project_name xclip_stage1_contrastive --loss_weights 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --contrastive_sampler removed_empty_classes --num_negatives_train 48 --num_negatives_val 50 --fold 1 --early_stopping 5
```

3. Refilled sampler

```bash
torchrun --nproc_per_node=2 --rdzv-endpoint=0.0.0.0:29700 train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 48 --batch_size_val 50 --save_freq 1 --num_workers 16 --devices 6,7 --epochs 64 --lr 0.0002 --project_name xclip_stage1_contrastive --loss_weights 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --contrastive_sampler refilled_empty_classes --num_negatives_train 48 --num_negatives_val 50 --fold 1 --early_stopping 5
```

