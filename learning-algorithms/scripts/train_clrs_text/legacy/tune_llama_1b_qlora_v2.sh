
for task in 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"
do
python train_clrs_text.py --task_names $task\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name test_stl --epochs 10 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 1000 --minimum_samples_validation 1000\
    --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora
done