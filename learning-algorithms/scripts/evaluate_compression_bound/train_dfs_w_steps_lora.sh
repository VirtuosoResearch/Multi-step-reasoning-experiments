for sample in 100 200 500 1000 2000 5000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 80 --max_output_length 170 --train_lengths 5 --test_lengths 5 --generate_output --runs 2 --lr 2e-5 \
    --save_name lora_clrs_dfs_length_5 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step
done

for sample in 1000 2000 5000 10000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 520 --max_output_length 795 --train_lengths 15 --test_lengths 15 --generate_output --runs 2 --lr 2e-5 \
    --save_name lora_clrs_dfs_length_15 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step
done

for sample in 1000 2000 5000 10000
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 2 --inference_batch_size 2 --accumlate 2 --max_length 805 --max_output_length 1035 --train_lengths 19 --test_lengths 19 --generate_output --runs 2 --lr 2e-5 \
    --save_name lora_clrs_dfs_length_19 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step
done