for sample in 500 1000 2000 5000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 16 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_non_cot_dfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 90 --only_answer_output \
    --load_model_dir Qwen-Qwen2.5-1.5B_dfs_${sample}_use_only_answer_output_lora_r_16_clrs_non_cot_dfs_run_0


python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 16 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_non_cot_dfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 90 --only_answer_output \
    --load_model_dir Qwen-Qwen2.5-1.5B_dfs_${sample}_use_only_answer_output_lora_r_16_clrs_non_cot_dfs_run_0 \
    --eval_step_num 15 --add_weight_perturb --perturb_std 0.005

done
