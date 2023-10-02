# TODO: 1. UPDATE PATH TO THE TRAIN/VAL/TEST METADATA FILES
# TODO: 2. REMEMBER TO UPDATE BEST MODEL GOT FROM STAGE-2 PRE-TRAINING

# CUB-200
python train_owl_vit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 32 --save_freq 1 --num_workers 8 --devices 0 --epochs 30 --lr 0.00002 --weight_decay 0.001 --project_name finetuning --loss_weights 0,1,1,1,1 --network_type classification --classification_loss ce_loss --early_stopping 5 --train_file "../data/bird_11K/metadata/finetuning/cub_train_reindexed.h5" --val_file "../data/bird_11K/metadata/finetuning/cub_val_reindexed.h5" --test_file "../data/bird_11K/metadata/finetuning/cub_test_reindexed.h5" --best_model "" --birdsoup_level 1 --finetuning "vision_encoder_mlp" --note "all_components_cub_200"


