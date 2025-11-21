styles=("zero_shot" "zero_shot_ba" "zero_shot_sbm" "zero_shot_sfn")
length=${#styles[@]}

for task_name in 'triangle_counting' 'edge_existence' 'node_degree' 'shortest_path' 'maximum_flow' # 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 'disconnected_nodes' 'reachability' 'triangle_counting' 'node_classification'
do
for ((i = 0; i < $length; i++)); do
  for ((j = i + 1; j < $length; j++)); do
    python train.py --task_names "${task_name}" "${task_name}" --prompt_styles "${styles[$i]}" "${styles[$j]}" --text_encoders "incident" "incident"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name pariwise_graph_types --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
  done
done
done