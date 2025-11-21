python train_clrs_text.py --task_names "bfs" "bridges" "bellman_ford" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 2 --batch_size 2 --inference_batch_size 2 --max_length 512 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 --accumulate 2\
    --save_name clrs_grouping --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 1000 --minimum_samples_validation 50 
