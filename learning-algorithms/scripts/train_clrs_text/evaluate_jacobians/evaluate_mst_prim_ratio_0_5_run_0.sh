#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

device="${1:-2}"
max_examples="${2:-10}"
max_jacobian_steps="${3:-0}"
power_iters="${4:-10}"

# checkpoint_path="external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_v1_ratio_0.5_run_0/epoch_epoch=9.pt"
# output_dir="jacobian_results/mst_prim_v1_ratio_0.5_run_0_max${max_examples}_steps${max_jacobian_steps}"

# extra_args=()
# if [[ "${FULL_JACOBIAN:-0}" == "1" ]]; then
#     extra_args+=(--full_jacobian)
#     output_dir="${output_dir}_full"
# else
#     output_dir="${output_dir}_adjacent"
# fi

# finite_difference_checks="${FINITE_DIFFERENCE_CHECKS:-2}"
# generation_max_new_tokens="${GENERATION_MAX_NEW_TOKENS:-120}"

# python evaluate_cot_jacobian.py \
#     --task_names "mst_prim" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --checkpoint_path "${checkpoint_path}" \
#     --device "${device}" \
#     --max_length 460 \
#     --max_output_length 120 \
#     --train_lengths 10 \
#     --test_lengths 10 \
#     --precision "bf16-true" \
#     --train_lora \
#     --lora_rank 16 \
#     --lora_alpha 128 \
#     --few_shot_k 0 \
#     --downsample_ratio 0.01 \
#     --minimum_samples 2000 \
#     --minimum_samples_validation 100 \
#     --reduce_steps_ratio 0.5 \
#     --eval_batch_size 1 \
#     --max_examples "${max_examples}" \
#     --max_jacobian_steps "${max_jacobian_steps}" \
#     --power_iters "${power_iters}" \
#     --max_target_tokens_for_jacobian 8 \
#     --jacobian_method jvp \
#     --attention_backend math \
#     --finite_difference_checks "${finite_difference_checks}" \
#     --generation_max_new_tokens "${generation_max_new_tokens}" \
#     --output_dir "${output_dir}" \
#     "${extra_args[@]}"


# Ratio 0.1

# checkpoint_path="external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_v2_ratio_0.1_run_0/epoch_epoch=9.pt"
# output_dir="jacobian_results/mst_prim_v2_ratio_0.1_run_0_max${max_examples}_steps${max_jacobian_steps}"

# extra_args=()
# if [[ "${FULL_JACOBIAN:-0}" == "1" ]]; then
#     extra_args+=(--full_jacobian)
#     output_dir="${output_dir}_full"
# else
#     output_dir="${output_dir}_adjacent"
# fi

# finite_difference_checks="${FINITE_DIFFERENCE_CHECKS:-2}"
# generation_max_new_tokens="${GENERATION_MAX_NEW_TOKENS:-60}"

# python evaluate_cot_jacobian.py \
#     --task_names "mst_prim" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --checkpoint_path "${checkpoint_path}" \
#     --device "${device}" \
#     --max_length 460 \
#     --max_output_length 60 \
#     --train_lengths 10 \
#     --test_lengths 10 \
#     --precision "bf16-true" \
#     --train_lora \
#     --lora_rank 16 \
#     --lora_alpha 128 \
#     --few_shot_k 0 \
#     --downsample_ratio 0.01 \
#     --minimum_samples 2000 \
#     --minimum_samples_validation 100 \
#     --reduce_steps_ratio 0.1 \
#     --eval_batch_size 1 \
#     --max_examples "${max_examples}" \
#     --max_jacobian_steps "${max_jacobian_steps}" \
#     --power_iters "${power_iters}" \
#     --max_target_tokens_for_jacobian 8 \
#     --jacobian_method jvp \
#     --attention_backend math \
#     --finite_difference_checks "${finite_difference_checks}" \
#     --generation_max_new_tokens "${generation_max_new_tokens}" \
#     --output_dir "${output_dir}" \
#     "${extra_args[@]}"


checkpoint_path="external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_use_only_answer_output_lora_r_16_clrs_mst_prim_v1_run_0/epoch_epoch=9.pt"
output_dir="jacobian_results/mst_prim_v1_only_answer_run_0_max${max_examples}_steps${max_jacobian_steps}"

extra_args=()
if [[ "${FULL_JACOBIAN:-0}" == "1" ]]; then
    extra_args+=(--full_jacobian)
    output_dir="${output_dir}_full"
else
    output_dir="${output_dir}_adjacent"
fi

finite_difference_checks="${FINITE_DIFFERENCE_CHECKS:-2}"
generation_max_new_tokens="${GENERATION_MAX_NEW_TOKENS:-40}"

python evaluate_cot_jacobian.py \
    --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --checkpoint_path "${checkpoint_path}" \
    --device "${device}" \
    --max_length 460 \
    --max_output_length 40 \
    --train_lengths 10 \
    --test_lengths 10 \
    --precision "bf16-true" \
    --train_lora \
    --lora_rank 16 \
    --lora_alpha 128 \
    --few_shot_k 0 \
    --downsample_ratio 0.01 \
    --minimum_samples 2000 \
    --minimum_samples_validation 100 \
    --only_answer_output --include_input_jacobian \
    --eval_batch_size 1 \
    --max_examples "${max_examples}" \
    --max_jacobian_steps "${max_jacobian_steps}" \
    --power_iters "${power_iters}" \
    --max_target_tokens_for_jacobian 8 \
    --jacobian_method jvp \
    --attention_backend math \
    --include_input_jacobian \
    --finite_difference_checks "${finite_difference_checks}" \
    --generation_max_new_tokens "${generation_max_new_tokens}" \
    --output_dir "${output_dir}" \
    "${extra_args[@]}"
