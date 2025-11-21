# python train_clrs_text.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" \
#     --model_key "Qwen/Qwen2.5-14B" \
#     --devices 1 --batch_size 2 --inference_batch_size 4 --max_length 200 --max_output_length 300 --train_lengths 5 --test_lengths 5 --generate_output --runs 1 --lr 2e-5 --accumulate 2\
#     --save_name clrs_all_algorithms --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 0 --downsample_ratio 0.001 --minimum_samples 80 --minimum_samples_validation 20

python train_clrs_text.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 2 --inference_batch_size 4 --max_length 200 --max_output_length 300 --train_lengths 5 --test_lengths 5 --generate_output --runs 1 --lr 2e-5 --accumulate 2\
    --save_name clrs_all_algorithms --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.001 --minimum_samples 80 --minimum_samples_validation 10

python train_clrs_text.py --task_names "dag_shortest_paths" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 2 --inference_batch_size 2 --max_length 512 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 --accumulate 2\
    --save_name clrs_single --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 1000 --minimum_samples_validation 50 --write_results

python train_clrs_text.py --task_names "floyd_warshall" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 2 --inference_batch_size 2 --max_length 632 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 --accumulate 2\
    --save_name clrs_single --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 1000 --minimum_samples_validation 50 --write_results

