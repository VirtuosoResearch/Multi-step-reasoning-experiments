
# for sample in 2000 3000 4000
# do
# python train_clrs_text.py --task_names "dijkstra" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 240 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
#     --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0
# done

# for sample in 4000 5000 6000
# do
# done

sample=5000
device=0
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 220 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 120 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5 --generate_output

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 70 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.25 --generate_output

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 45 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.1 --generate_output

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 30 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v2 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --only_answer_output --generate_output

# 456 219
# 456 109
# 456 65
# 456 43
# 456 21