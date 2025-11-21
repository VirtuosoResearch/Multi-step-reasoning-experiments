for scale in 0.1 0.08 0.06 0.04 0.02; do
for seed in 0 1 2 3 4 5 6 7 8 9 
do
python fast_estimate_eval_approximation.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
    --model_key "Qwen/Qwen2.5-14B"\
    --devices 0 --batch_size 2 --inference_batch_size 2 --max_length 200 --max_output_length 300 --train_lengths 5 --test_lengths 5 --generate_output --runs 1 --lr 2e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision "bf16-true"\
    --few_shot_k 0 --downsample_ratio 0.001 --minimum_samples 80 --minimum_samples_validation 20 \
    --save_name qwen14b_12_graph_algorithms --epochs 0 \
    --compute_gradients_seed 0 --project_gradients_dim 400 \
    --load_model_dir "Qwen-Qwen2.5-14B_12_tasks_lora_r_16_clrs_all_algorithms_run_0/epoch_epoch=6.pt" \
    --number_of_subsets 1 --subset_seed $seed --subset_size 0.3 --scale $scale --lr_regularization_lambda 5e3
done
done

for scale in 0.1 0.08 0.06 0.04 0.02; do
for seed in 0 1 2 3 4 5 6 7 8 9 
do
python fast_estimate_eval_approximation.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 2 --inference_batch_size 2 --max_length 200 --max_output_length 300 --train_lengths 5 --test_lengths 5 --generate_output --runs 1 --lr 2e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision "bf16-true"\
    --few_shot_k 0 --downsample_ratio 0.001 --minimum_samples 80 --minimum_samples_validation 20 \
    --save_name llama8b_12_graph_algorithms --epochs 0 \
    --compute_gradients_seed 0 --project_gradients_dim 400 \
    --load_model_dir "meta-llama-Llama-3.1-8B_12_tasks_lora_r_16_clrs_all_algorithms_run_0/epoch_epoch=9.pt" \
    --number_of_subsets 1 --subset_seed $seed --subset_size 0.3 --scale $scale --lr_regularization_lambda 5e3
done
done

# for scale in 0.0025 0.0020 0.0015 0.0010 0.0005 0.0001; do
# for seed in 0 1 2 3 4 5 6 7 8 9 
# do
# python fast_estimate_eval_approximation.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
#     --model_key "meta-llama/Llama-3.2-1B"\
#     --devices 2 --batch_size 2 --inference_batch_size 2 --max_length 632 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 200 --minimum_samples_validation 50 \
#     --save_name llama1b_12_graph_algorithms --epochs 0 \
#     --compute_gradients_seed 0 --project_gradients_dim 400 \
#     --load_model_dir "meta-llama-Llama-3.2-1B_12_tasks_lora_r_16_clrs_all_algorithms_run_0/epoch_epoch=6.pt" \
#     --number_of_subsets 1 --subset_seed $seed --subset_size 0.3 --scale $scale --lr_regularization_lambda 5e3
# done
# done

# python fast_estimate_eval_approximation.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
#     --model_key "Qwen/Qwen2.5-1.5B"\
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 632 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 200 --minimum_samples_validation 50 \
#     --save_name qwen_12_graph_algorithms --epochs 0 \
#     --compute_gradients_seed 0 --project_gradients_dim 400 \
#     --load_model_dir "Qwen-Qwen2.5-1.5B_12_tasks_lora_r_16_clrs_all_algorithms_run_0/epoch_epoch=7.pt" \
#     --number_of_subsets 1 --subset_size 0.3 --scale 0.0025 --lr_regularization_lambda 5e3