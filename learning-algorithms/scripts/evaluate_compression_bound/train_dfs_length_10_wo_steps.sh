# for dim in 5000 1000
# do
# python train_clrs_text.py --task_names "dfs" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 16 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
#     --save_name "clrs_dfs_wo_steps_${dim}" --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 5000 --minimum_samples_validation 100 --only_answer_output \
#     --intrinsic_dim $dim --intrinsic_mode "rdkronqr"
# done

python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 16 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name "clrs_dfs_wo_steps_only_lora" --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 5000 --minimum_samples_validation 100 --only_answer_output