
for task in 'bfs' "dfs" "topological_sort" "articulation_points" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"
do
for shot in 40 100 200 400
do
python train_clrs_text.py --task_names $task \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length $((shot * 200)) --max_output_length 150 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_20 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k $shot --only_evaluate_test_set
done
done

for shot in 40 100 200 400
do
python train_clrs_text.py --task_names "strongly_connected_components" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length $((shot * 280)) --max_output_length 210 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_20 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k $shot --only_evaluate_test_set
done

for shot in 40 100 200 400
do
python train_clrs_text.py --task_names "bridges" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length $((shot * 530)) --max_output_length 510 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_20 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k $shot --only_evaluate_test_set
done