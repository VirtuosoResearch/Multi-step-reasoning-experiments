python train_clrs_text.py --task_names "dfs" "articulation_points" "strongly_connected_components" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 --accumulate 1\
    --save_name clrs_grouping --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 1000 --minimum_samples_validation 50 

python train_clrs_text.py --task_names "topological_sort" "mst_kruskal" "mst_prim" "dijkstra" "dag_shortest_paths" "floyd_warshall" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 632 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 --accumulate 1\
    --save_name clrs_grouping --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 1000 --minimum_samples_validation 50 

