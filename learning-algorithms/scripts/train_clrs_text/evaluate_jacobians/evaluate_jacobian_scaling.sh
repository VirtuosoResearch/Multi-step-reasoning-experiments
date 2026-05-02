#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

device="${1:-1}"
max_examples="${2:-10}"
max_jacobian_steps="${3:-0}"
power_iters="${4:-10}"
task_name="dijkstra" # "mst_prim" # "dijkstra" # "bellman_ford" # "bfs" # "mst_prim" # "bellman_ford" #"mst_prim" #
max_length=460 # 460 # 460 # 460 # 460 # 460 # 460 # 256 # 460 # 460 # 460
max_output_length=70 # 45 # 30 # 70 # 120 # 220 # 180 # 180 # 230 # 100 120 
ratio=0.25
min_jacobian_steps=1
split="test"
quant_bits=4

checkpoint_path="external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_quant_lora_4bit_clrs_dijkstra_v1_run_0/epoch_epoch=6.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_quant_lora_2bit_clrs_dijkstra_v1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_quant_lora_3bit_clrs_dijkstra_v1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_quant_lora_1bit_clrs_dijkstra_v1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_quant_lora_3bit_clrs_mst_prim_quant_run_0/epoch_epoch=8.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_quant_lora_1bit_clrs_mst_prim_quant_run_0/epoch_epoch=7.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_quant_lora_2bit_clrs_mst_prim_quant_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_quant_lora_4bit_clrs_mst_prim_quant_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_use_only_answer_output_lora_r_16_clrs_mst_prim_v1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_v2_ratio_0.1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_ratio_0.25_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_ratio_1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_v1_ratio_0.5_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_clrs_dijkstra_ratio_0.1_run_0/epoch_epoch=8.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_use_only_answer_output_lora_r_16_clrs_dijkstra_v2_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_clrs_dijkstra_ratio_0.25_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_clrs_dijkstra_ratio_0.5_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_clrs_dijkstra_ratio_1.0_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_lora_r_16_clrs_bellmanford_ratio_1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_bfs_2000_len_[10]_lora_r_16_clrs_bfs_ratio_1_run_0/epoch_epoch=7.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_ratio_1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_lora_r_16_clrs_bellmanford_v1_ratio_0.5_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_mst_prim_2000_len_[10]_lora_r_16_clrs_mst_prim_v1_ratio_0.5_run_0/epoch_epoch=9.pt"
output_dir="jacobian_results/${task_name}_ratio_${ratio}_run_0_scaling_max${max_examples}_steps${max_jacobian_steps}_min_jacobian_steps_${min_jacobian_steps}_split_${split}_quant_${quant_bits}bit"
# FULL_JACOBIAN=1
# extra_args=()
# if [[ "${FULL_JACOBIAN:-0}" == "1" ]]; then
#     extra_args+=(--full_jacobian)
#     output_dir="${output_dir}_full"
# else
#     output_dir="${output_dir}_adjacent"
# fi

finite_difference_checks="${FINITE_DIFFERENCE_CHECKS:-2}"
generation_max_new_tokens="${GENERATION_MAX_NEW_TOKENS:-120}"

python evaluate_cot_jacobian.py \
    --task_names "${task_name}" \
    --model_key "Qwen/Qwen2.5-1.5B" \
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
    --reduce_steps_ratio ${ratio} \
    --eval_batch_size 1 \
    --max_examples "${max_examples}" \
    --max_jacobian_steps "${max_jacobian_steps}" \
    --min_jacobian_steps "${min_jacobian_steps}" \
    --power_iters "${power_iters}" \
    --max_target_tokens_for_jacobian 8 \
    --split "${split}" \
    --jacobian_method jvp \
    --attention_backend math \
    --finite_difference_checks "${finite_difference_checks}" \
    --generation_max_new_tokens "${generation_max_new_tokens}" \
    --output_dir "${output_dir}" \
    --full_jacobian \
    --use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}"
    # \
    # --only_answer_output 

python - "${output_dir}" <<'PY'
import csv
import math
import os
import sys

output_dir = sys.argv[1]
loss_path = os.path.join(output_dir, "generated_step_loss_rows.csv")
rho_path = os.path.join(output_dir, "rho_prefix_rows.csv")
out_path = os.path.join(output_dir, "test_loss_scaling_rows.csv")
summary_path = os.path.join(output_dir, "test_loss_scaling_summary.csv")

def key(row):
    return (row["task"], row["split"], row["length"], row["sample_idx"], row["t"])

with open(rho_path, newline="") as f:
    rho_by_key = {key(row): row for row in csv.DictReader(f)}

rows = []
with open(loss_path, newline="") as f:
    for row in csv.DictReader(f):
        rho_row = rho_by_key.get(key(row), {})
        teacher_loss = float(row["teacher_step_loss"])
        generated_loss = float(row["generated_prefix_step_loss"])
        loss_gap = generated_loss - teacher_loss
        adjacent_log_rho = float(rho_row.get("adjacent_log_rho_t_1", "nan"))
        adjacent_rho = float(rho_row.get("adjacent_rho_t_1", "nan"))
        recursive_log_rho = float(rho_row.get("recursive_log_rho_t_1", "nan"))
        recursive_rho = float(rho_row.get("recursive_rho_t_1", "nan"))
        rows.append(
            {
                "task": row["task"],
                "split": row["split"],
                "length": row["length"],
                "sample_idx": row["sample_idx"],
                "t": row["t"],
                "num_steps": row["num_steps"],
                "L_t_CoT": teacher_loss,
                "L_t_generated": generated_loss,
                "L_t_gap": loss_gap,
                "adjacent_rho_t_1": adjacent_rho,
                "adjacent_log_rho_t_1": adjacent_log_rho,
                "recursive_rho_t_1": recursive_rho,
                "recursive_log_rho_t_1": recursive_log_rho,
                "generated_step_correct": row["generated_step_correct"],
                "generated_step_edit_distance": row["generated_step_edit_distance"],
            }
        )

fieldnames = [
    "task",
    "split",
    "length",
    "sample_idx",
    "t",
    "num_steps",
    "L_t_CoT",
    "L_t_generated",
    "L_t_gap",
    "adjacent_rho_t_1",
    "adjacent_log_rho_t_1",
    "recursive_rho_t_1",
    "recursive_log_rho_t_1",
    "generated_step_correct",
    "generated_step_edit_distance",
]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = {}
for row in rows:
    groups.setdefault(int(row["t"]), []).append(row)

summary_rows = []
for t, group in sorted(groups.items()):
    def mean(name):
        vals = [float(r[name]) for r in group if math.isfinite(float(r[name]))]
        return sum(vals) / len(vals) if vals else float("nan")

    summary_rows.append(
        {
            "t": t,
            "count": len(group),
            "mean_L_t_CoT": mean("L_t_CoT"),
            "mean_L_t_generated": mean("L_t_generated"),
            "mean_L_t_gap": mean("L_t_gap"),
            "mean_adjacent_log_rho_t_1": mean("adjacent_log_rho_t_1"),
            "mean_recursive_log_rho_t_1": mean("recursive_log_rho_t_1"),
            "mean_generated_step_correct": mean("generated_step_correct"),
            "mean_generated_step_edit_distance": mean("generated_step_edit_distance"),
        }
    )

summary_fields = [
    "t",
    "count",
    "mean_L_t_CoT",
    "mean_L_t_generated",
    "mean_L_t_gap",
    "mean_adjacent_log_rho_t_1",
    "mean_recursive_log_rho_t_1",
    "mean_generated_step_correct",
    "mean_generated_step_edit_distance",
]
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Wrote {out_path}")
print(f"Wrote {summary_path}")
PY
