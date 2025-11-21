python train.py --task_names 'edge_existence' 'node_degree' 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 'disconnected_nodes' 'reachability' 'shortest_path' 'maximum_flow' 'triangle_counting' 'node_classification'\
    --prompt_styles "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot" "zero_shot"\
    --text_encoders "incident" "incident" "incident" "incident" "incident" "incident" "incident" "incident" "incident" "incident" "incident" "incident" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name mtl --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
