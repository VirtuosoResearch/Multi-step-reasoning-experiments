for sample in 5000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 600 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 145 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_run_1/epoch_epoch=9.pt --eval_step_num 15
done


for sample in 5000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 600 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 145 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_run_1/epoch_epoch=9.pt --eval_step_num 15 --add_weight_perturb --perturb_std 0.005
done


for sample in 5000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 600 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 145 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_run_1/epoch_epoch=9.pt --eval_step_num 15 --add_weight_perturb --perturb_std 0.008
done


for sample in 5000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 600 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 145 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_run_1/epoch_epoch=9.pt --eval_step_num 15 --add_weight_perturb --perturb_std 0.01
done

