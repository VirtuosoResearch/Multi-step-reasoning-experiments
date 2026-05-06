#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

device="${1:-1}"
test_length="${2:-11}"
sample="${3:-2000}"
minimum_samples_validation="${4:-100}"

model_key="Qwen/Qwen2.5-1.5B"
task_name="dfs"
train_length=11
batch_size=4
inference_batch_size=4
max_length=288
max_output_length=549
lora_rank=16
lora_alpha=128
lr=2e-5
precision="bf16-true"
downsample_ratio=0.01

common_args=(
  --task_names "${task_name}"
  --model_key "${model_key}"
  --devices "${device}"
  --batch_size "${batch_size}"
  --inference_batch_size "${inference_batch_size}"
  --max_length "${max_length}"
  --max_output_length "${max_output_length}"
  --train_lengths "${train_length}"
  --test_lengths "${test_length}"
  --runs 1
  --lr "${lr}"
  --save_name "eval_length_generalization_${task_name}_len_${test_length}"
  --epochs 0
  --precision "${precision}"
  --train_lora
  --lora_rank "${lora_rank}"
  --lora_alpha "${lora_alpha}"
  --few_shot_k 0
  --downsample_ratio "${downsample_ratio}"
  --minimum_samples "${sample}"
  --minimum_samples_validation "${minimum_samples_validation}"
  --generate_output
  --eval_last_step
  --only_evaluate_test_set
)

run_eval() {
  local label="$1"
  local checkpoint="$2"
  shift 2

  if [[ "${checkpoint}" == learning-algorithms/external_lightning_logs/* ]]; then
    checkpoint="${checkpoint#learning-algorithms/external_lightning_logs/}"
  fi

  if [[ "${checkpoint}" == external_lightning_logs/* ]]; then
    checkpoint="${checkpoint#external_lightning_logs/}"
  fi

  if [[ ! -f "external_lightning_logs/${checkpoint}" ]]; then
    echo "Checkpoint not found: external_lightning_logs/${checkpoint}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Evaluating ${label}"
  echo "Checkpoint: external_lightning_logs/${checkpoint}"
  echo "Test length: ${test_length}"
  echo "============================================================"

  python train_clrs_text.py \
    "${common_args[@]}" \
    "$@" \
    --load_model_dir "${checkpoint}"
}

# Ours: quantized LoRA.
run_eval "ours_quant_3bit_answer_only_run_0" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_use_only_answer_output_lora_r_16_quant_lora_3bit_clrs_dfs_quant_ratio_0_run_0/epoch_epoch=9.pt" \
  --use_quant --quant_training_mode lora --quant_w_bits 3 --only_answer_output

run_eval "ours_quant_3bit_answer_only_run_1" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_use_only_answer_output_lora_r_16_quant_lora_3bit_clrs_dfs_quant_ratio_0_run_1/epoch_epoch=8.pt" \
  --use_quant --quant_training_mode lora --quant_w_bits 3 --only_answer_output

# No CoT: answer-only supervision/evaluation.
run_eval "no_cot_run_0" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_use_only_answer_output_lora_r_16_clrs_bfs_ratio_0_run_0/epoch_epoch=9.pt" 

run_eval "no_cot_run_1" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_use_only_answer_output_lora_r_16_clrs_bfs_ratio_0_run_1/epoch_epoch=8.pt" 

# SFT CoT.
run_eval "sft_cot_run_0" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_lora_r_16_clrs_dfs_v2_ratio_1_run_0/epoch_epoch=9.pt"

run_eval "sft_cot_run_1" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_lora_r_16_clrs_dfs_v2_ratio_1_run_1/epoch_epoch=9.pt"

# Reduced CoT.
run_eval "reduced_cot_ratio_0.5_run_0" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_lora_r_16_clrs_dfs_v2_ratio_0.5_run_0/epoch_epoch=9.pt" 

run_eval "reduced_cot_ratio_0.5_run_1" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_lora_r_16_clrs_dfs_v2_ratio_0.5_run_1/epoch_epoch=9.pt" 

# Implicit CoT: validation/test targets are answer-only in train_clrs_text.py.
run_eval "implicit_cot_steps_2000_run_0" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_implicit_cot_lora_r_16_clrs_dfs_implicit_cot_steps_2000_run_0/epoch_epoch=8.pt" 

run_eval "implicit_cot_steps_2000_run_1" \
  "Qwen-Qwen2.5-1.5B_dfs_2000_len_[10]_implicit_cot_lora_r_16_clrs_dfs_implicit_cot_steps_2000_run_1/epoch_epoch=7.pt" 
