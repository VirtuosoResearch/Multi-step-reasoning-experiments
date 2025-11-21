for task in "bfs" "bellman_ford"
do
python train_clrs_text.py --task_names $task \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 8 --lora_alpha 64 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15
done

python train_clrs_text.py --task_names 'dfs' \
        --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 1400 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 8 --lora_alpha 64 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'dijkstra' \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 8 --lora_alpha 64 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'mst_prim' \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 700 --max_output_length 512 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 8 --lora_alpha 64 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

python train_clrs_text.py --task_names 'topological_sort' \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 520 --max_output_length 1300 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 10 --precision "bf16-true" --train_lora --lora_rank 8 --lora_alpha 64 --use_qlora \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15


python train_clrs_text.py --task_names bfs \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 5e-4 \
    --save_name tuning_stl --epochs 10 --use_graph_llama \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

# --train_lora --lora_rank 16 --lora_alpha 128