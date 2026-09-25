#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

device="0"
task_name="bfs"
ratio="1"
split="test"
max_jacobian_steps="5" # 0 means no limit 
min_jacobian_steps="0"
max_length="256"
max_output_length="230"
max_target_tokens_for_jacobian="8"
finite_difference_checks="0"
ce_lipschitz_noise_checks="${CE_LIPSCHITZ_NOISE_CHECKS:-8}"
ce_lipschitz_noise_epsilon="${CE_LIPSCHITZ_NOISE_EPSILON:-1e-4}"
generation_max_new_tokens="$max_output_length"
max_examples="10"
power_iters="10"
model_key="meta-llama/Llama-3.2-1B"

baseline_name=""
checkpoint_path="meta-llama-Llama-3.2-1B_bfs_1000_len_[10]_lora_r_16_clrs_bfs_run_0/epoch_epoch=9.pt"
# "meta-llama-Llama-3.2-1B_dijkstra_1000_len_[10]_lora_r_16_clrs_dijkstra_run_0/epoch_epoch=9.pt"
# "meta-llama-Llama-3.2-1B_bellman_ford_1000_len_[10]_lora_r_16_clrs_bellmanford_run_0/epoch_epoch=8.pt"
# "meta-llama-Llama-3.2-1B_mst_prim_1000_len_[10]_lora_r_16_clrs_mst_prim_run_0/epoch_epoch=9.pt"
# "google-gemma-2-2b-it_bfs_1000_len_[10]_lora_r_16_clrs_bfs_run_0/epoch_epoch=9.pt"
# "google-gemma-2-2b-it_dijkstra_1000_len_[10]_lora_r_16_clrs_dijkstra_run_0/epoch_epoch=8.pt"
# "google-gemma-2-2b-it_mst_prim_1000_len_[10]_lora_r_16_clrs_mst_prim_run_0/epoch_epoch=9.pt"
# "google-gemma-2-2b-it_bellman_ford_1000_len_[10]_lora_r_16_clrs_bellmanford_run_0/epoch_epoch=9.pt"
# "Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_lora_r_16_quant_lora_4bit_clrs_bellmanford_quant_run_0/epoch_epoch=8.pt"
# "Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_implicit_cot_lora_r_16_clrs_bellmanford_implicit_cot_steps_2000_run_0/epoch_epoch=9.pt" 460 20
# "Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_lora_r_16_clrs_bellmanford_noise_0.002_run_0/epoch_epoch=9.pt" 460 180
# "Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_implicit_cot_lora_r_16_clrs_dijkstra_implicit_cot_steps_2000_run_0/epoch_epoch=8.pt" 460 30
# "Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_quant_lora_1bit_clrs_dijkstra_v1_run_0/epoch_epoch=9.pt" 460 30
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_clrs_dijkstra_v1_ratio_0.1_run_0/epoch_epoch=9.pt" 460 45
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_implicit_cot_lora_r_16_clrs_mst_prim_implicit_cot_steps_2000_run_0/epoch_epoch=8.pt" 460 40
# "Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_noise_0.001_run_0/epoch_epoch=9.pt" 460 230
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_quant_lora_4bit_clrs_mst_prim_quant_run_0/epoch_epoch=9.pt" 460 60
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_clrs_dijkstra_ratio_1.0_run_0/epoch_epoch=9.pt" 460 220 
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_ratio_1_run_0/epoch_epoch=9.pt" 460 230
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_bfs_2000_len_[10]_lora_r_16_clrs_bfs_ratio_1_run_0/epoch_epoch=7.pt" 256 180
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_lora_r_16_clrs_bellmanford_ratio_1_run_0/epoch_epoch=9.pt" 460 180

quant_bits=""
quant_suffix=""
quant_args=()
if [[ -n "${quant_bits}" ]]; then
    quant_suffix="_quant_${quant_bits}bit"
    quant_args+=(--use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}")
fi

output_dir="jacobian_results/${model_key}_${task_name}_${baseline_name}_ratio_${ratio}_run_0_scaling_expected_embedding_hidden_final_max${max_examples}_steps${max_jacobian_steps}_min_jacobian_steps_${min_jacobian_steps}_split_${split}${quant_suffix}_full"

python evaluate_cot_jacobian.py \
    --task_names "${task_name}" \
    --model_key "${model_key}" \
    --checkpoint_path "${checkpoint_path}" \
    --device "${device}" \
    --max_length "${max_length}" \
    --max_output_length "${max_output_length}" \
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
    --reduce_steps_ratio "${ratio}" \
    --eval_batch_size 1 \
    --max_examples "${max_examples}" \
    --max_jacobian_steps "${max_jacobian_steps}" \
    --min_jacobian_steps "${min_jacobian_steps}" \
    --power_iters "${power_iters}" \
    --max_target_tokens_for_jacobian "${max_target_tokens_for_jacobian}" \
    --split "${split}" \
    --jacobian_method jvp \
    --jacobian_output expected_embedding \
    --embedding_final_state hidden_state \
    --attention_backend math \
    --finite_difference_checks "${finite_difference_checks}" \
    --ce_lipschitz_noise_checks "${ce_lipschitz_noise_checks}" \
    --ce_lipschitz_noise_epsilon "${ce_lipschitz_noise_epsilon}" \
    --generation_max_new_tokens "${generation_max_new_tokens}" \
    --output_dir "${output_dir}" \
    --full_jacobian \
    "${quant_args[@]}"

python evaluate_cot_jacobian_post_processing.py "${output_dir}"
