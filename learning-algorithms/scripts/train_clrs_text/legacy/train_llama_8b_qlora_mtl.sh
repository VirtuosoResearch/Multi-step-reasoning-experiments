python train_clrs_text.py --task_names 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_adapter --reduction_factor 128 --use_qadapter \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15
