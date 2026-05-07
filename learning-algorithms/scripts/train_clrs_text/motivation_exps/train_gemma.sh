sample=1000
device=0

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "google/gemma-2-2b-it" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 462 --max_output_length 230 --train_lengths 10 --test_lengths 10 --runs 1 --lr 2e-5 \
    --save_name clrs_mst_prim --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output


python train_clrs_text.py --task_names "bellman_ford" \
    --model_key "google/gemma-2-2b-it" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 463 --max_output_length 198 --train_lengths 10 --test_lengths 10 --runs 1 --lr 2e-5 \
    --save_name clrs_bellmanford --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "google/gemma-2-2b-it" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 461 --max_output_length 220 --train_lengths 10 --test_lengths 10 --runs 1 --lr 2e-5 \
    --save_name clrs_dijkstra --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output

python train_clrs_text.py --task_names "bfs" \
    --model_key "google/gemma-2-2b-it" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 180 --train_lengths 10 --test_lengths 10 --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output --reduce_steps_equally_spaced
