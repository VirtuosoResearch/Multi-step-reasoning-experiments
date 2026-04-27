# for sample in 500 1000 2000
# do
# python train_clrs_text.py --task_names "mst_kruskal" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 510 --max_output_length 440 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
#     --save_name clrs_mst_kruskal_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0
# done

for sample in 1000 2000 3000
do
python train_clrs_text.py --task_names "mst_kruskal" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 510 --max_output_length 330 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_mst_kruskal_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5
done


# 509 430
# 509 320
# 509 254
# 509 232
# 509 210