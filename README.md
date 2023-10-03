# PEEB: Part-based Bird Classifier with an Explanable and Editable Languange Bottlenect

_**tldr:** We proposed a part-based bird classifier which make predictions based on part-wise descriptions. Our method directily utilize human-provided descriptions, in this work from GPT4. It outperform CLIP and M&V by 10 points in CUB and 28 points in NABirds._


### How our method is different from other methods?
:start2: [Comparison between M&V (demo)](x-clip:auburn.edu:8086)

:start2: [Editable Demo](x-clip:auburn.edu:7076)


### Model Card
We provide our pre-trained models as well as all fine-tuned models used in this work.

| Model                 | Fine-tune from   | Training Set      | Checkpoint       |
|-----------------------|------------------|-------------------|------------------|
| PEEB[-test]           | OWL-ViT[base]    | Bird-11K[-test]   | [Link]()|
| PEEB[-CUB]            | OWL-ViT[base]    | Bird-11K[-CUB]    | [Link]()|
| PEEB[-NAB]            | OWL-ViT[base]    | Bird-11K[-NAB]    | [Link]()|
| PEEB[-test]^CUB       | PEEB[-test]      | CUB               | [Link]()|
| PEEB[-cub]^Akata      | PEEB[-CUB]       | [akata2015label]  | [Link]()|
| PEEB[-cub]^SCS        | PEEB[-CUB]       | CUB-SCS           | [Link]()|
| PEEB[-cub]^SCE        | PEEB[-CUB]       | CUB-SCE           | [Link]()|
| PEEB[-nab]^SCS        | PEEB[-NAB]       | NABirds-SCS       | [Link]()|
| PEEB[-nab]^SCE        | PEEB[-NAB]       | NABirds-SCE       | [Link]()|


### Prerequisite

#### Step 1: Clone this repo
```bash
git clone https://github.com/ThangPM/xclip && cd xclip
```

#### Step 2: Install enviroment

```bash
conda create -n xclip python=3.9
conda activate xclip

pip install -r requirements.txt
conda install pycocotools -c conda-forge
python -m spacy download en_core_web_sm
```

#### Step 4: Set up PYTHONPATH enviroment
* Before running experiments, we need to export the `PYTHONPATH` variable to the current directory, its sub-directories and change directory to the `src` folder as follows.
```bash
# Assume the current directory is xclip
export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/mmdetection
cd src/
```

#### Step 5: Set up the file path in `config.py` file
Make the following changes to `src/config.py`:
1. Set `DATASET_DIR` to `{DATA_PATH}`
2. Set `PRECOMPUTED_DIR` to `{DATA_PATH}`


### Prepare Dataset

We do not redistribute the datasets, we provide a [meta data]() of the combined dataset. Please follow [Data preparation](./Data_preparation.md) to prepare for the data if you would like to train the model.


### Training

#### Teacher logits
Before the training, we need to compute the teacher logits. Please make sure to finish all steps in [Data preparation](./Data_preparation.md) before start training. 


#### Pre-training
Run the following script to start the pre-training:

```bash
torchrun --nproc_per_node=2 train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 48 --batch_size_val 50 --save_freq 1 --num_workers 16 --devices 2,3 --epochs 64 --lr 0.0002 --project_name xclip_stage1_contrastive --loss_weights 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 48 --num_negatives_val 50 --fold 1 --early_stopping 5
```

#### Fine-tuning
Run the following script to start fine-tuning:



### Evaluation
For evaluation only, consider run with the following scripts:
