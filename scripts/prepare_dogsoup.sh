# compute dogsoup boxes
python src/precompute_boxes.py --dataset dogsoup_v1 --owl_prompt_type stanforddog-6-parts-dog --batch_size 12 --meta_path /home/lab/datasets/dogsoup/metadata_v1.h5 --image_root /home/lab/datasets/dogsoup --device cuda:7 --filter_size 100
