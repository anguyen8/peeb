# TODO: 1. UPDATE PATH TO THE TRAIN/VAL/TEST METADATA FILES
# TODO: 2. REMEMBER TO UPDATE BEST MODEL GOT FROM STAGE 1

# BIRD-11K_test: Excluding test sets of CUB, NABirds and iNaturalist
# BS = 32 -- #NEGS = 48 -- DECAY 1e-2 -- EPOCHS = 32 -- EARLY STOPPING = 5
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --batch_size_val 50 --save_freq 1 --num_workers 8 --devices 0 --epochs 32 --lr 0.00002 --weight_decay 0.01 --project_name stage2_pretraining --loss_weights 0,1,1,2,0 --network_type contrastive --train_box_heads_only --num_negatives_train 48 --num_negatives_val 50 --early_stopping 5 --train_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/train_keep_child_a100_reindexed.h5" --val_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/val_keep_child_a100_reindexed.h5" --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_cub_reindexed.h5" --best_model "" --birdsoup_level 1 --note "stage2_pretraining_BIRD-11K_test"

# BIRD-11K_CUB: Excluding CUB classes
# BS = 32 -- #NEGS = 48 -- DECAY 1e-3 -- EPOCHS = 32 -- EARLY STOPPING = 5
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --batch_size_val 50 --save_freq 1 --num_workers 8 --devices 1 --epochs 32 --lr 0.00002 --weight_decay 0.001 --project_name stage2_pretraining --loss_weights 0,1,1,2,0 --network_type contrastive --train_box_heads_only --num_negatives_train 48 --num_negatives_val 50 --early_stopping 5 --train_file "../data/bird_11K/metadata/level_3_exclude_cub/train_keep_child_a100_reindexed.h5" --val_file "../data/bird_11K/metadata/level_3_exclude_cub/val_keep_child_a100_reindexed.h5" --test_file "/home/lab/datasets/bird_11K/metadata/level_3_exclude_cub/clore_split/test_unseen_reindexed.h5" --best_model "" --birdsoup_level 3 --note "stage2_pretraining_BIRD-11K_CUB"

# BIRD-11K_NABirds: Excluding NABirds classes
# BS = 32 -- #NEGS = 48 -- DECAY 1e-3 -- EPOCHS = 32 -- EARLY STOPPING = 5
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --batch_size_val 50 --save_freq 1 --num_workers 8 --devices 2 --epochs 32 --lr 0.00002 --weight_decay 0.001 --project_name stage2_pretraining --loss_weights 0,1,1,2,0 --network_type contrastive --train_box_heads_only --num_negatives_train 48 --num_negatives_val 50 --early_stopping 5 --train_file "../data/bird_11K/metadata/level_3_exclude_nabirds/train_keep_child_a100_reindexed.h5" --val_file "../data/bird_11K/metadata/level_3_exclude_nabirds/val_keep_child_a100_reindexed.h5" --test_file "../data/bird_11K/metadata/level_3_exclude_nabirds/super_category_split/easy_test_reindexed.h5" --best_model "" --birdsoup_level 3 --note "stage2_pretraining_BIRD-11K_NABirds"


