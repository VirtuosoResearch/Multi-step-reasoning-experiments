# for sample in 100 200 500 1000 2000 5000
# do
# python train_clrs_text.py --task_names "bfs" "bellman_ford"\
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 180 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 2e-5 \
#     --save_name clrs_bfs --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step
# done

for sample in 100 200 500 1000 2000 5000
do
python train_clrs_text.py --task_names "bellman_ford" "dfs"\
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 2 --inference_batch_size 4 --max_lengt 460 --max_output_length 660 --train_lengths 10 --test_lengths 10 --generate_output --runs 2 --lr 1e-5 --accumulate 2\
    --save_name clrs_bellman_ford --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step
done