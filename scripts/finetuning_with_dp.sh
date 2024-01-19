# --- Finetuning with DP --- 
# B32 with batch size 320, lr 2e-4, decay 1e-2, epochs 32, early stopping 5, patience 2
python train_owlvit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 320 --batch_size_val 240 --save_freq 1 --num_workers 8 --devices 0,1,2,3,4,5,6,7 --epochs 32 --lr 0.0002 --weight_decay 0.01 --project_name dp_peeb --loss_weight 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 320 --num_negatives_val 240 --early_stopping 5 --birdsoup_level 1 --note stage1_pretraining_lr2e-4_b320 --enable_dp --scheduler_patience 2 
# B32 with batch size 320, lr 2e-3, decay 1e-2, epochs 32, early stopping 5, patience 2
python train_owlvit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 320 --batch_size_val 240 --save_freq 1 --num_workers 8 --devices 0,1,2,3,4,5,6,7 --epochs 32 --lr 0.002 --weight_decay 0.01 --project_name dp_peeb --loss_weight 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 320 --num_negatives_val 240 --early_stopping 5 --birdsoup_level 1 --note stage1_pretraining_lr2e-4_b320 --enable_dp --scheduler_patience 2
# B32 with batch size 320, lr 1e-2, decay 1e-2, epochs 32, early stopping 5, patience 2, reduce factor 0.7
python train_owlvit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 320 --batch_size_val 240 --save_freq 1 --num_workers 8 --devices 0,1,2,3,4,5,6,7 --epochs 32 --lr 0.01 --weight_decay 0.01 --project_name dp_peeb --loss_weight 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 320 --num_negatives_val 240 --early_stopping 5 --birdsoup_level 1 --note stage1_pretraining_lr1e-2_b320 --enable_dp --scheduler_patience 2 --scheduler_factor 0.7 --scheduler_verbose

# B32 with batch size 320, decay 1e-2, epochs 32, early stopping 7, patience 4, continue trianing starting with lr = 6.25e-4
python train_owlvit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 320 --batch_size_val 240 --save_freq 1 --num_workers 8 --devices 0,1,2,3,4,5,6,7 --epochs 32 --lr 6.25e-4 --weight_decay 0.01 --project_name dp_peeb --loss_weight 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 320 --num_negatives_val 240 --early_stopping 6 --birdsoup_level 1 --note stage1_pretraining_lr6.25e-4_b320_stop7_patience4_continue --enable_dp --scheduler_patience 4 --scheduler_verbose --best_model "/home/peijie/workspace/xclip_ext/peeb/results/bird_soup/level_1/training_contrastive/chatgpt-owlvit-base-patch32/_stage1_pretraining_lr1e-2_b320_stop7_patience4/wandb/latest-run/files/last.pt"

# B32 with batch size 320, lr 1e-2, decay 1e-2, epochs 60, early stopping 7, patience 4, reduce factor 0.7
python train_owlvit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 320 --batch_size_val 240 --save_freq 1 --num_workers 8 --devices 0,1,2,3,4,5,6,7 --epochs 60 --lr 0.01 --weight_decay 0.01 --project_name dp_peeb --loss_weight 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 320 --num_negatives_val 240 --early_stopping 7 --birdsoup_level 1 --note stage1_pretraining_lr1e-2_b320_stop7_patience4_e60_sf1e-7 --enable_dp --scheduler_patience 4 --scheduler_factor 0.7 --scheduler_verbose

# B32 with batch size 320, lr 1e-2, decay 1e-2, epochs 60, early stopping 7, patience 4, reduce factor 0.7
python train_owlvit.py --model owlvit-base-patch32 --dataset bird_soup --sub_datasets all --descriptors chatgpt --prompt_type 0 --batch_size 320 --batch_size_val 240 --save_freq 1 --num_workers 8 --devices 0,1,2,3,4,5,6,7 --epochs 64 --lr 0.01 --weight_decay 0.02 --project_name dp_peeb --loss_weight 0,0,0,0,1 --network_type contrastive --freeze_box_heads --logits_from_teacher --num_negatives_train 320 --num_negatives_val 240 --early_stopping 10 --birdsoup_level 1 --note stage1_pretraining_lr1e-2_b320_stop10_patience4_e64_wd2e-3 --enable_dp --scheduler_patience 4 --scheduler_verbose