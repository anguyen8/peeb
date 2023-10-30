## Prepare the `Bird-11K` Dataset.

### Step 1: Download Datasets
We use data from 7 different datasets plus 55k images for eBird. You may download the datasets from the corresponding resource:
- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [NABirds](https://dl.allaboutbirds.org/nabirds)
- [Indian Birds](https://www.kaggle.com/datasets/ichhadhari/indian-birds)
- [Birdnap v7](https://thomasberg.org/)
- [iNaturalist 2021-birds](https://www.kaggle.com/datasets/sharansmenon/inat2021birds) (birds only)
- [ImageNet-birds](https://www.image-net.org/) (birds only)
- [BIRDS 525](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
- [Macaulay Library at the Cornell Lab of Ornithology](https://www.birds.cornell.edu/home/) (You may need to request access to these images.)
*Note: The full image list from Macaulay Library at the Cornell Lab of Ornithology can be found in this [spreadsheet](./data/metadata/macaulay_library_image_list.xlsx). Or you can use the extracted assetID list from the JSON file [here](./data/macaulay_library_assetID.json).


Create a folder to store the data, and export the data path to the environment:
```bash
export DATA_PATH = PATH_TO_YOUR_DATA_FOLDER
```


### Step 2: Download metadata
Download the following files from [Box.com](https://auburn.app.box.com/s/owiwf73yxurz3r2k2i6x0r2cg9mgglcc) and store them in the `DATA_PATH` folder.
- boxes_no_logits.tar.gz
- metadata.tar.gz

 Extract the data and rearrange sub-folders by running the following commands:
```bash
cd ${DATA_PATH}
tar boxes_no_logits.tar.gz 
tar -xvf metadata.tar.gz -C bird_soup/    # Use the latest metadata folder
```

#### Images from Macaulay Library at the Cornell Lab of Ornithology.
Since the images from Macaulay Library at the Cornell Lab of Ornithology are not publicly available, you need to request from their [website](https://www.birds.cornell.edu/) for access to download the images.


### Step 3: Rename all files
Rename all images with the following script:
```bash
python src/rename_birdsoup_images.py --meta_path ${DATA_PATH}/metadata/bird_soup_uncased_v2.h5 --image_path ${DATA_PATH}/images
```

If one wants to skip the images from Macaulay Library at the Cornell Lab of Ornithology, we provide the metafile that excludes the corresponding images [here](https://auburn.box.com/s/9arivxicqkv52n9pajth3rbxyowuw0us). You can download the file and replace the original metafile. e.g.,
```bash
python src/rename_birdsoup_images.py --meta_path ${DATA_PATH}/metadata/bird_soup_uncased_v2_no_ebird.h5 --image_path ${DATA_PATH}/images 
```
Note that without the images from Macaulay Library at the Cornell Lab of Ornithology, the diversity of the dataset will be reduced, and the performance of the model will be lower. We use the column name `data_source` to indicate the source of the images. The value `ebird` indicates the images from Macaulay Library at the Cornell Lab of Ornithology; one can easily exclude these images by filtering the metafile with `'data_source' != 'ebird'` for the rest of the experiments. 


### Step 4: Compute teacher logits
It takes around 60-90 minutes for one single NVIDIA A100-SXM40 GPU.
```bash
CUDA_VISIBLE_DEVICES=0 python update_boxes_logits.py --model "owlvit-large-patch14" --dataset "bird_soup" --descriptors "chatgpt" --batch_size 32 --num_workers 16 --prompt_type 0 --owlvit_threshold -1
```

After the above steps, the data folder should look like this:
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

