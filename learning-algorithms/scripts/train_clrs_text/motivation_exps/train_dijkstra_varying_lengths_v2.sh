
for sample in 5000 6000 7000 8000 9000 10000
do
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 70 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v2 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.25
done

for sample in 5000 6000 7000 8000 9000 10000
do
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 50 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v2 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.1
done

for sample in 5000 6000 7000 8000 9000 10000
do
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 30 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v2 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --only_answer_output
done

# 456 219
# 456 109
# 456 65
# 456 43
# 456 21