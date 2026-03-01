# Baseline
# CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack." \
#                 --target_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack and holding a hiking stick." \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/hiking.jpg' \
#                 --num_steps 15  \
#                 --inject 2 \
#                 --offload \
#                 --name 'flux-dev' \
#                 --output_prefix 'rf_solver' \
#                 --output_dir 'examples/edit-result/hiking/' 

# Fast Editing
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack." \
                --target_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack and holding a hiking stick." \
                --guidance 2 \
                --source_img_dir 'examples/source/hiking.jpg' \
                --num_steps 8  \
                --inject 1 \
                --offload \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --output_dir 'examples/edit-result/hiking/' 

# Replace
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "A woman hiking on a trail with mountains in the distance." \
                --target_prompt "A man hiking on a trail with mountains in the distance." \
                --guidance 2 \
                --source_img_dir 'examples/source/hiking.jpg' \
                --num_steps 8  \
                --inject 1 \
                --offload \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --output_dir 'examples/edit-result/hiking/' 

