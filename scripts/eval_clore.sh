# TODO: 1. UPDATE PATH TO THE TEST METADATA FILES
# TODO: 2. REMEMBER TO UPDATE BEST MODEL GOT FROM FINE-TUNING
# TODO: 3. SPECIFY --visualize + #batches TO VISUALIZE RESULTS IF NEEDED

# UNSEEN CLASSES
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 0 --loss_weights 0,1,1,1,1 --network_type classification --eval_test --no_log --test_file "../data/bird_11K/metadata/level_3_exclude_cub/clore_split/test_unseen_reindexed.h5" --best_model "" --birdsoup_level 3

# SEEN CLASSES
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --num_workers 8 --devices 1 --loss_weights 0,1,1,1,1 --network_type classification --eval_test --no_log --test_file "../data/bird_11K/metadata/level_3_exclude_cub/clore_split/test_seen_reindexed.h5" --best_model "" --birdsoup_level 3

