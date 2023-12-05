# TODO: 1. UPDATE PATH TO THE TEST METADATA FILES
# TODO: 2. REMEMBER TO UPDATE BEST MODEL GOT FROM STAGE-1 OR STAGE-2 PRE-TRAINING
# TODO: 3. ONLY FOR STAGE-1 PRE-TRAINING, SPECIFY --logits_from_teacher

# CUB-200
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 0 --loss_weights 0,1,1,1,1 --network_type contrastive --eval_test --no_log --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_cub_reindexed.h5" --best_model "" --birdsoup_level 1 # --logits_from_teacher
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 7 --loss_weights 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 32 --num_negatives_val 32 --birdsoup_level 1 --test_file "/home/lab/datasets/bird_soup/metadata/level_1_exclude_cub_nabirds_inat/test_cub_reindexed.h5" --val_file "/home/lab/datasets/bird_soup/metadata/level_1_exclude_cub_nabirds_inat/test_cub_reindexed.h5" --eval_test --no_log --best_model ""


# NABIRDS-555
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 1 --loss_weights 0,1,1,1,1 --network_type contrastive --eval_test --no_log --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_nabirds_reindexed.h5" --best_model "" --birdsoup_level 1 # --logits_from_teacher

# INATURALIST-1486
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 2 --loss_weights 0,1,1,1,1 --network_type contrastive --eval_test --no_log --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_inat_reindexed.h5" --best_model "" --birdsoup_level 1 # --logits_from_teacher

# INATURALIST-1486 (FILTERED BY A100, UNBALANCED)
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 3 --loss_weights 0,1,1,1,1 --network_type contrastive --eval_test --no_log --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_inat_a100_unbalanced_reindexed.h5" --best_model "" --birdsoup_level 1 # --logits_from_teacher

# INATURALIST-1486 (FILTERED BY A200, UNBALANCED)
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 4 --loss_weights 0,1,1,1,1 --network_type contrastive --eval_test --no_log --test_file "../data/bird_11K/metadata/level_1_exclude_cub_nabirds_inat/test_inat_a200_unbalanced_reindexed.h5" --best_model "" --birdsoup_level 1 # --logits_from_teacher


