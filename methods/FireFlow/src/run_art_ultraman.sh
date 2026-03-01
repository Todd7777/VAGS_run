# # Fast Editing
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the Ultraman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 8  \
                --inject 1 \
                --name 'flux-dev'  \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --output_dir 'examples/edit-result/art/' 

# Better Instruction Following
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the Ultraman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 8  \
                --inject 1 \
                --name 'flux-dev'  \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --reuse_v 0 \
                --editing_strategy 'add_q' \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_add_q' \
                --output_dir 'examples/edit-result/art/' 
