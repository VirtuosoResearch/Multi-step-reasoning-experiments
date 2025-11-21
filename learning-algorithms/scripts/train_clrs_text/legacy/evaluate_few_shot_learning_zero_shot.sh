
python train_clrs_text.py --task_names 'bfs' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 2 --batch_size 1 --inference_batch_size 1 --max_length 700 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_zero --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'bellman_ford' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 2 --batch_size 1 --inference_batch_size 1 --max_length 700 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_zero --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dfs' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 2 --batch_size 1 --inference_batch_size 1 --max_length 512 --max_output_length 1400 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_zero --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dijkstra' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 2 --batch_size 1 --inference_batch_size 1 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_zero --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'mst_prim' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 2 --batch_size 1 --inference_batch_size 1 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_zero --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'topological_sort' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 2 --batch_size 1 --inference_batch_size 1 --max_length 520 --max_output_length 1300 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_zero --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 0 --only_evaluate_test_set --train_lengths 15 --test_lengths 15


# for task in 'bfs' "dfs" "topological_sort" "articulation_points" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"
# do
# python train_clrs_text.py --task_names $task \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 200 --max_output_length 150 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 0 --only_evaluate_test_set
# done

# for task in "strongly_connected_components"
# do
# python train_clrs_text.py --task_names $task \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 280 --max_output_length 210 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 0 --only_evaluate_test_set
# done

# for task in "bridges"
# do
# python train_clrs_text.py --task_names $task \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 530 --max_output_length 510 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 0 --only_evaluate_test_set
# done