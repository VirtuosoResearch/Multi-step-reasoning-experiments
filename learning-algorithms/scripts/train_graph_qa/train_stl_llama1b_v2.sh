# 'edge_existence' 'node_degree' 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 
# 'disconnected_nodes' 'reachability' 'shortest_path' 'maximum_flow' 'triangle_counting' 'node_classification'

for lora_rank in 4 8 16 32 64 128
do
for task_name in 'edge_existence'
do
python train.py --task_names $task_name --prompt_styles "zero_shot" --text_encoders "adjacency"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 2 --lr 5e-5 \
    --train_lora --lora_rank $lora_rank --lora_alpha $((lora_rank*8)) \
    --save_name test --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
done
done