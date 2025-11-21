task_names=("bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")

length=${#task_names[@]}

for ((i = 4; i < $length; i++)); do
    python train_clrs_text.py --task_names "${task_names[$i]}" \
        --model_key "Qwen/Qwen2.5-1.5B" \
        --devices 1 --batch_size 2 --inference_batch_size 2 --max_length 512 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --accumulate 2\
        --save_name clrs_pair --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
        --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 500 --minimum_samples_validation 50 --write_results
  
  for ((j = i + 1; j < $length; j++)); do
    python train_clrs_text.py --task_names "${task_names[$i]}" "${task_names[$j]}" \
        --model_key "Qwen/Qwen2.5-1.5B" \
        --devices 1 --batch_size 2 --inference_batch_size 2 --max_length 512 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --accumulate 2\
        --save_name clrs_pair --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
        --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 500 --minimum_samples_validation 50 --write_results
  done
done