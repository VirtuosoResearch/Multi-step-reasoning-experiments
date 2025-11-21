# python train_clrs_text.py --task_names 'bfs' \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 3800 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_5 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 5 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

# python train_clrs_text.py --task_names 'bfs' \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 6900 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 10 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

# python train_clrs_text.py --task_names 'bfs' \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 13200 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_20 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 20 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

# python train_clrs_text.py --task_names 'bellman_ford' \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 4800 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_5 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 5 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

# python train_clrs_text.py --task_names 'bellman_ford' \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 8600 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 10 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

# python train_clrs_text.py --task_names 'bellman_ford' \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 16300 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name prompt_20 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
#     --few_shot_k 20 --only_evaluate_test_set --train_lengths 15 --test_lengths 15


python train_clrs_text.py --task_names 'dfs' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 9868 --max_output_length 1400 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_5 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 5 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dfs' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 19228 --max_output_length 1400 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 10 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dijkstra' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 6237 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_5 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 5 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dijkstra' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 11735 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 10 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'mst_prim' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 6233 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_5 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 5 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'mst_prim' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 6233 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 10 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'topological_sort' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 9143 --max_output_length 1300 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_5 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 5 --only_evaluate_test_set --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'topological_sort' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 17464 --max_output_length 1300 --generate_output --runs 1 --lr 5e-5 \
    --save_name prompt_10 --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora \
    --few_shot_k 10 --only_evaluate_test_set --train_lengths 15 --test_lengths 15