# Baseline
# CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
#                 --target_prompt "A minimalistic line-drawing portrait of Joe Biden with black lines and light brown shadow" \
#                 --guidance 2.5 \
#                 --source_img_dir 'examples/source/nobel.jpg' \
#                 --num_steps 25  \
#                 --inject 2 \
#                 --name 'flux-dev' \
#                 --offload \
#                 --output_prefix 'rf_solver' \
#                 --output_dir 'examples/edit-result/nobel/'  

# Fast Editing
CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
                --target_prompt "A minimalistic line-drawing portrait of Joe Biden with black lines and light brown shadow" \
                --guidance 3 \
                --source_img_dir 'examples/source/nobel.jpg' \
                --num_steps 15  \
                --inject 1 \
                --name 'flux-dev' \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --output_dir 'examples/edit-result/nobel/'  
