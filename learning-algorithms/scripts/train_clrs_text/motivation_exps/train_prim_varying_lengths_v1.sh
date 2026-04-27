sample=2000
device=1
python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 230 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_mst_prim_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 120 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_mst_prim_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5 --generate_output

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 80 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_mst_prim_v2 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.25 --generate_output

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 60 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_mst_prim_v2 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.1 --generate_output

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 40 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_mst_prim_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --only_answer_output --generate_output

# for sample in 500 1000 2000
# do
# python train_clrs_text.py --task_names "mst_prim" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 230 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
#     --save_name clrs_mst_prim_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0
# done

# for sample in 3000 4000 5000
# do
# python train_clrs_text.py --task_names "mst_prim" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 120 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
#     --save_name clrs_mst_prim_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5
# done

# for sample in 8000 9000 10000
# do
# python train_clrs_text.py --task_names "mst_prim" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 40 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
#     --save_name clrs_mst_prim_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --only_answer_output
# done


# 457 219
# 457 109
# 457 65
# 457 43
# 457 21