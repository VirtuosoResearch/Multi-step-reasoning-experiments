python train_clrs_text.py --task_names "dag_shortest_paths" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 500 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 5000 --minimum_samples_validation 79 --only_answer_output


python train_clrs_text.py --task_names "dag_shortest_paths" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 500 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 5000 --minimum_samples_validation 79 --only_answer_output --load_model_dir Qwen-Qwen2.5-1.5B_dag_shortest_paths_5000_use_only_answer_output_lora_r_16_clrs_bfs_run_0 --eval_step_num 15

python train_clrs_text.py --task_names "dag_shortest_paths" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 500 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 5000 --minimum_samples_validation 79 --only_answer_output --load_model_dir Qwen-Qwen2.5-1.5B_dag_shortest_paths_5000_use_only_answer_output_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005