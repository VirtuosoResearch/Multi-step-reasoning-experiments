sample=2000
device=1

# for noise in 0.002 0.001 0.0005
# do
# python train_clrs_text.py --task_names "mst_prim" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 230 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
#     --save_name clrs_mst_prim_noise_${noise} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
#     --use_forward_noise_injection --forward_noise_lora_only --forward_noise_std ${noise}
# done

for noise in 0.0005
do
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 220 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_noise_${noise} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
    --use_forward_noise_injection --forward_noise_lora_only --forward_noise_std ${noise}
done

for noise in 0.0005
do
python train_clrs_text.py --task_names "bellman_ford" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 180 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_bellmanford_noise_${noise} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
    --use_forward_noise_injection --forward_noise_lora_only --forward_noise_std ${noise}
done

for noise in 0.0005
do
python train_clrs_text.py --task_names "dfs" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 660 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
    --save_name clrs_dfs_noise_${noise} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 \
    --use_forward_noise_injection --forward_noise_lora_only --forward_noise_std ${noise}
done