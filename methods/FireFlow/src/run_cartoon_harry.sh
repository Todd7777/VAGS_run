# Baseline
# CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
#                 --target_prompt "a cartoon style Harry Potter raising his left hand " \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/cartoon.jpg' \
#                 --num_steps 25 \
#                 --inject 2 \
#                 --name 'flux-dev' \
#                 --output_prefix 'rf_solver' \
#                 --offload \
#                 --output_dir 'examples/edit-result/cartoon/' 

# Fast Editing
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
                --target_prompt "a cartoon style Harry Potter raising his left hand " \
                --guidance 2 \
                --source_img_dir 'examples/source/cartoon.jpg' \
                --num_steps 8 \
                --inject 1 \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --offload \
                --output_dir 'examples/edit-result/cartoon/' 

# Better Instruction Following
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
                --target_prompt "a cartoon style Harry Potter raising his left hand " \
                --guidance 2 \
                --source_img_dir 'examples/source/cartoon.jpg' \
                --num_steps 10 \
                --inject 1 \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --offload \
                --output_dir 'examples/edit-result/cartoon/'  