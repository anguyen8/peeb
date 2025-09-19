# PEEB: Part-based Image Classifier with an Explainable and Editable Language Bottleneck

`Official Implementation` for NAACL 2024 Findings paper [PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck
](https://arxiv.org/abs/2403.05297) by [Thang M. Pham](https://scholar.google.com/citations?user=eNrX3mYAAAAJ&hl=en), [Peijie Chen](https://chanfeechen.github.io/resume), [Tin Nguyen](https://ngthanhtin.github.io/), [Seunghyun Yoon](https://david-yoon.github.io/), [Trung Bui](https://research.adobe.com/person/trung-bui/), [Anh Totti Nguyen](https://anhnguyen.me/).

**TLDR:** We proposed a part-based fine-grained image classifier (identifying 🐦 or 🐕) that makes predictions by matching visual object parts detected in the input image with textual part-wise descriptions of each class. Our method directly utilizes human-provided descriptions (in this work, from GPT4). PEEB outperforms [CLIP](https://github.com/openai/CLIP) (2021) and [M&V](https://arxiv.org/abs/2210.07183) (2023) by +10 points on CUB and +28 points on NABirds in generalized zero-shot settings. While CLIP and its extensions depend heavily on the prompt to contain the known class name, PEEB relies solely on the textual descriptors and does not use class names and therefore generalizes to objects where class names or examplar photos are not unknown.

**If you use this software, please consider citing:**

    @article{auburn2024peeb,
      title={PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck},
      author={Pham, Thang M and Chen, Peijie and Nguyen, Tin and Yoon, Seunghyun and Bui, Trung and Nguyen, Anh},
      booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
      month = jul,
      year = "2024",
      address = "Mexico city, Mexico",
      publisher = "Association for Computational Linguistics"
    }
    
### 🌟 Interactive Demo 🌟

[Interactive Demo](https://huggingface.co/spaces/XAI/PEEB) that shows how one can edit a class' descriptors (during inference) to directly modify the classifier.


### Model Card
We provide our pre-trained models as well as all fine-tuned models used in this work. See [here](./Model_card.md) for details.


### Prerequisite

#### Step 1: Clone this repo
```bash
git clone https://github.com/{username}/xclip && cd xclip
```

#### Step 2: Install environment

```bash
conda create -n xclip python=3.10
conda activate xclip

pip install -r requirements.txt
conda install pycocotools -c conda-forge
python -m spacy download en_core_web_sm
```


### How to construct Bird-11K 🐦?

We do not redistribute the datasets; we provide a ```metadata``` of the combined dataset. To prepare the `Bird-11K` dataset, please follow [Data preparation](./Data_preparation.md) to prepare the data if you would like to train the model.


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
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 0 --loss_weights 0,1,1,1,1 --network_type classification --eval_test --no_log --test_file "../data/bird_11K/metadata/finetuning/cub_test_reindexed.h5" --best_model "PEEB_CUB.pt" --birdsoup_level 1
```
Evaluation commands of other test sets can be found in [here](./scripts/). 


## Troubleshooting

### Issue: Unable to reproduce zero-shot results on CUB
**Problem:** Users may experience difficulty reproducing the reported zero-shot results on the CUB dataset.

**Solution**: 

Download the test file here [cub_test_reindexed.h5](https://tigermailauburn-my.sharepoint.com/:u:/g/personal/ttn0011_auburn_edu/EbN9S7JXxy1MoRoa6ZopdJoBK1iDz3TywVQExK26XvAEsg?e=NXZRYP). 

Download the model here [PEEB_CUB.pt](https://drive.google.com/file/d/1IIGllKlCc8zgRVJiTXIB1CpbWRbwIhfU/view?pli=1).

And put these files to the arguments `--test_file` and `--best_model`, respectively, in the following **zero-shot evaluation command**.

**Zero-shot Evaluation Command:**
   For reproducing zero-shot results on CUB, use the following command:
   ```bash
   python src/train_owl_vit.py \
     --model owlvit-base-patch32 \
     --dataset bird_soup \
     --sub_datasets all \
     --descriptors chatgpt \
     --prompt_type 0 \
     --batch_size 32 \
     --num_workers 8 \
     --devices 0 \
     --loss_weights 0,1,1,1,1 \
     --network_type classification \
     --eval_test \
     --no_log \
     --test_file "cub_test_reindexed.h5" \
     --best_model "PEEB_CUB.pt" \
     --birdsoup_level 1
   ```

