
for sample in 2000 5000 10000
do
python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 260 --max_output_length 190 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --eval_step_num 15

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 260 --max_output_length 190 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_bfs_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 260 --max_output_length 190 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_bfs_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.008
done