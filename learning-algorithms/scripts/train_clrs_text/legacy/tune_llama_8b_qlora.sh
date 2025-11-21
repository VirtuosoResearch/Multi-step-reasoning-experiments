# for task in "bellman_ford" # bfs
# do
# python train_clrs_text.py --task_names $task \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
#     --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 1 --lora_alpha 8 --use_qlora \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15
# done

python train_clrs_text.py --task_names 'dfs' \
        --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 1400 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 1 --lora_alpha 8 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dijkstra' \
        --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 1 --lora_alpha 8 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'mst_prim' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 1 --lora_alpha 8 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'topological_sort' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 520 --max_output_length 1300 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 1 --lora_alpha 8 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15