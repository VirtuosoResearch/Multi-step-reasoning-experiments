for task_name in  'connected_nodes' 'reachability' 'shortest_path' 'maximum_flow' 'triangle_counting' # 'edge_existence' 'node_degree' 'node_count' 'edge_count'  'cycle_check' 'disconnected_nodes'  'node_classification'
do
for prompt_style in "er" # "ba" "path" "sbm" "sfn" "star" "complete"
do
for samples in 1000 2000 4000 8000 
do
python train.py --task_names $task_name --prompt_styles "zero_shot_${prompt_style}" --text_encoders "incident" --min_nodes 20 --max_nodes 40\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 2 --inference_batch_size 2 --max_length 4096 --max_output_length 32 --generate_output --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name stl_length_20_40 --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000 --accumulate 4
done
done
done