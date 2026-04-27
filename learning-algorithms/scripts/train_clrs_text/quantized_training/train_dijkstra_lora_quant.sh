#!/usr/bin/env bash
# set -euo pipefail

# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo_root="$(cd "${script_dir}/../../.." && pwd)"
# cd "${repo_root}"

device="${1:-1}"
sample="${2:-2000}"
# quant_bits="${3:-2}"

for quant_bits in 1
do
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices "${device}" --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 220 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}" \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples "${sample}" --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices "${device}" --batch_size 4 --inference_batch_size 4 --max_length 460 --max_output_length 120 --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
    --save_name clrs_dijkstra_v1 --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}" \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples "${sample}" --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 0.5 --generate_output
done