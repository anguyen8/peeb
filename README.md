# PEEB: Part-based Bird Classifier with an Explanable and Editable Languange Bottlenect

_**TLDR:** We proposed a part-based bird classifier that makes predictions based on part-wise descriptions. Our method directly utilizes human-provided descriptions (in this work, from GPT4). It outperforms CLIP and M&V by 10 points in CUB and 28 points in NABirds._


### How is our method different from other methods?

[Comparison between M&V (demo)](http://x-clip.auburn.edu:8087)

[Editable Demo](http://x-clip.auburn.edu:7087)


### Model Card
We provide our pre-trained models as well as all fine-tuned models used in this work. See [here](./Model_card.md) for details.


### Prerequisite

#### Step 1: Clone this repo
```bash
git clone https://github.com/ThangPM/xclip && cd xclip
```

#### Step 2: Install environment

```bash
conda create -n xclip python=3.9
conda activate xclip

pip install -r requirements.txt
conda install pycocotools -c conda-forge
python -m spacy download en_core_web_sm
```


### Prepare Dataset

We do not redistribute the datasets; we provide a ```metadata``` of the combined dataset. Please follow [Data preparation](./Data_preparation.md) to prepare the data if you would like to train the model.


### Training

**Teacher logits** are required for the training. Please make sure to finish all steps in [Data preparation](./Data_preparation.md) before starting training. 

#### Step 1: Set up PYTHONPATH environment
* Before running experiments, we need to export the `PYTHONPATH` variable to the current directory and its sub-directories and change the directory to the `src` folder as follows.
```bash
# Assume the current directory is xclip
export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/mmdetection
cd src/
```

#### Step 2: Set up the file path in `config.py` file
Make the following changes to `src/config.py`:
1. Set `DATASET_DIR` to `{DATA_PATH}`
2. Set `PRECOMPUTED_DIR` to `{DATA_PATH}`

Our full training commands can be found [here](./scripts/).

#### Pre-training Example
We have two steps in our pre-training,

Step 1: **Part-text mapping**, e.g., for pre-training at excluding test sets of CUB, NABirds and iNaturalist
```bash
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --batch_size_val 50 --save_freq 1 --num_workers 8 --devices 0 --epochs 32 --lr 0.0002 --weight_decay 0.01 --project_name stage1_pretraining --loss_weights 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 48 --num_negatives_val 50 --early_stopping 5 --train_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/train_keep_child_a100_reindexed.h5" --val_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/val_keep_child_a100_reindexed.h5" --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_cub_reindexed.h5" --birdsoup_level 1 --note "stage1_pretraining_BIRD-11K_test"
```
See [1st_training.sh](./scripts/1st_training.sh) for other commands.

Step 2: Box prediction calibration, e.g.,
```bash
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --batch_size_val 50 --save_freq 1 --num_workers 8 --devices 0 --epochs 32 --lr 0.00002 --weight_decay 0.01 --project_name stage2_pretraining --loss_weights 0,1,1,2,0 --network_type contrastive --train_box_heads_only --num_negatives_train 48 --num_negatives_val 50 --early_stopping 5 --train_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/train_keep_child_a100_reindexed.h5" --val_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/val_keep_child_a100_reindexed.h5" --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_cub_reindexed.h5" --best_model "" --birdsoup_level 1 --note "stage2_pretraining_BIRD-11K_test"
```
See [2nd_training.sh](./scripts/2nd_training.sh) for other commands.


#### Fine-tuning Example
To fine-tune the model on a specific downstream dataset, e.g., CUB
```bash
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --save_freq 1 --num_workers 8 --devices 0 --epochs 30 --lr 0.00002 --weight_decay 0.001 --project_name finetuning --loss_weights 0,1,1,1,1 --network_type classification --classification_loss ce_loss --early_stopping 5 --train_file "../data/bird_11K/metadata/finetuning/cub_train_reindexed.h5" --val_file "../data/bird_11K/metadata/finetuning/cub_val_reindexed.h5" --test_file "../data/bird_11K/metadata/finetuning/cub_test_reindexed.h5" --best_model "" --birdsoup_level 1 --finetuning "vision_encoder_mlp" --note "all_components_cub_200"
```
See [3rd_training_cub_200.sh](./scripts/3rd_training_cub_200.sh) and [3rd_training_zeroshot.sh](./scripts/3rd_training_zeroshot.sh) for fine-tuning on CUB and other zeroshot dataset.


### Evaluation
For evaluation only, you may run the following script (for CUB):
```bash
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 0 --loss_weights 0,1,1,1,1 --network_type classification --eval_test --no_log --test_file "../data/bird_11K/metadata/finetuning/cub_test_reindexed.h5" --best_model "" --birdsoup_level 1
```
Evaluation commands of other test sets can be found in [here](./scripts/). 
