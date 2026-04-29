sample=2000
device=0


python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 180 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output --reduce_steps_equally_spaced

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 100 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0.5 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5 --generate_output --reduce_steps_equally_spaced

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 60 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0.25 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.25 --generate_output --reduce_steps_equally_spaced

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 50 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0.1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.1 --generate_output --reduce_steps_equally_spaced

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 50 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --only_answer_output --generate_output


python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 180 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 100 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0.5 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5 --generate_output

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 60 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0.25 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.25 --generate_output

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 50 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0.1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.1 --generate_output

python train_clrs_text.py --task_names "bfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 50 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bfs_v2_ratio_0 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --only_answer_output --generate_output
