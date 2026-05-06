#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

device="0"
task_name="bellman_ford"
ratio="1.0"
split="test"
max_jacobian_steps="0" # 0 means no limit 
min_jacobian_steps="5"
max_length="460"
max_output_length="180"
max_target_tokens_for_jacobian="8"
finite_difference_checks="0"
ce_lipschitz_noise_checks="${CE_LIPSCHITZ_NOISE_CHECKS:-8}"
ce_lipschitz_noise_epsilon="${CE_LIPSCHITZ_NOISE_EPSILON:-1e-4}"
generation_max_new_tokens="$max_output_length"
max_examples="10"
power_iters="10"

checkpoint_path="external_lightning_logs/Qwen-Qwen2.5-1.5B_bellman_ford_2000_len_[10]_lora_r_16_clrs_bellmanford_ratio_1_run_0/epoch_epoch=9.pt"
# "external_lightning_logs/Qwen-Qwen2.5-1.5B_dijkstra_2000_len_[10]_lora_r_16_quant_lora_4bit_clrs_dijkstra_v1_run_0/epoch_epoch=6.pt"

quant_bits=""
quant_suffix=""
quant_args=()
if [[ -n "${quant_bits}" ]]; then
    quant_suffix="_quant_${quant_bits}bit"
    quant_args+=(--use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}")
fi

output_dir="jacobian_results/${task_name}_ratio_${ratio}_run_0_scaling_expected_embedding_max${max_examples}_steps${max_jacobian_steps}_min_jacobian_steps_${min_jacobian_steps}_split_${split}${quant_suffix}_full"

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
    --attention_backend math \
    --finite_difference_checks "${finite_difference_checks}" \
    --ce_lipschitz_noise_checks "${ce_lipschitz_noise_checks}" \
    --ce_lipschitz_noise_epsilon "${ce_lipschitz_noise_epsilon}" \
    --generation_max_new_tokens "${generation_max_new_tokens}" \
    --output_dir "${output_dir}" \
    --full_jacobian \
    "${quant_args[@]}"

python - "${output_dir}" <<'PY'
import csv
import math
import os
import sys

output_dir = sys.argv[1]
loss_path = os.path.join(output_dir, "generated_step_loss_rows.csv")
rho_path = os.path.join(output_dir, "rho_prefix_rows.csv")
jacobian_path = os.path.join(output_dir, "jacobian_rows.csv")
out_path = os.path.join(output_dir, "test_loss_scaling_rows.csv")
summary_path = os.path.join(output_dir, "test_loss_scaling_summary.csv")

def key(row):
    return (row["task"], row["split"], row["length"], row["sample_idx"], row["t"])

def sample_key(row):
    return (row["task"], row["split"], row["length"], row["sample_idx"])

def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

with open(rho_path, newline="") as f:
    rho_by_key = {key(row): row for row in csv.DictReader(f)}

jacobian_norms = {}
train_errors = {}
max_target_step_by_sample = {}
with open(jacobian_path, newline="") as f:
    for row in csv.DictReader(f):
        skey = sample_key(row)
        target_step = int(row["i"])
        source_step = int(row["j"])
        jac_norm = safe_float(row.get("jac_norm"))
        train_error = safe_float(row.get("train_error_i"))
        if math.isfinite(jac_norm):
            jacobian_norms[(skey, target_step, source_step)] = jac_norm
            max_target_step_by_sample[skey] = max(
                max_target_step_by_sample.get(skey, 0), target_step
            )
        if math.isfinite(train_error):
            train_errors.setdefault((skey, source_step), train_error)

prefix_bound_by_key = {}
for skey, max_target_step in max_target_step_by_sample.items():
    recursive_rho = {}
    for target_step in range(1, max_target_step + 1):
        if target_step == 1:
            prefix_bound_by_key[(*skey, str(target_step))] = {
                "sum_rho_t_i": 0.0,
                "bound_proxy_t": 0.0,
            }
            continue

        sum_rho_t_i = 0.0
        bound_proxy_t = 0.0
        for source_step in range(1, target_step):
            direct_norm = jacobian_norms.get((skey, target_step, source_step), float("nan"))
            if not math.isfinite(direct_norm):
                continue

            rho_value = direct_norm
            for middle_step in range(source_step + 1, target_step):
                target_to_middle = jacobian_norms.get(
                    (skey, target_step, middle_step), float("nan")
                )
                middle_to_source = recursive_rho.get(
                    (middle_step, source_step), float("nan")
                )
                if math.isfinite(target_to_middle) and math.isfinite(middle_to_source):
                    rho_value += target_to_middle * middle_to_source

            recursive_rho[(target_step, source_step)] = rho_value
            sum_rho_t_i += rho_value

            train_error = train_errors.get((skey, source_step), float("nan"))
            if math.isfinite(train_error):
                bound_proxy_t += rho_value * train_error

        prefix_bound_by_key[(*skey, str(target_step))] = {
            "sum_rho_t_i": sum_rho_t_i,
            "bound_proxy_t": bound_proxy_t,
        }

rows = []
with open(loss_path, newline="") as f:
    for row in csv.DictReader(f):
        rho_row = rho_by_key.get(key(row), {})
        prefix_stats = prefix_bound_by_key.get(key(row), {})
        teacher_loss = float(row["teacher_step_loss"])
        generated_loss = float(row["generated_prefix_step_loss"])
        bound_proxy_t = safe_float(prefix_stats.get("bound_proxy_t"))
        ce_lipschitz_noise_max = safe_float(row.get("ce_lipschitz_noise_max"))
        noise_max_calibrated_bound_t = (
            ce_lipschitz_noise_max * bound_proxy_t
            if math.isfinite(ce_lipschitz_noise_max) and math.isfinite(bound_proxy_t)
            else float("nan")
        )
        rows.append(
            {
                "task": row["task"],
                "jacobian_output": row.get("jacobian_output", "expected_embedding"),
                "split": row["split"],
                "length": row["length"],
                "sample_idx": row["sample_idx"],
                "t": row["t"],
                "num_steps": row["num_steps"],
                "L_t_CoT": teacher_loss,
                "L_t_generated": generated_loss,
                "L_t_gap": generated_loss - teacher_loss,
                "adjacent_rho_t_1": float(rho_row.get("adjacent_rho_t_1", "nan")),
                "adjacent_log_rho_t_1": float(rho_row.get("adjacent_log_rho_t_1", "nan")),
                "recursive_rho_t_1": float(rho_row.get("recursive_rho_t_1", "nan")),
                "recursive_log_rho_t_1": float(rho_row.get("recursive_log_rho_t_1", "nan")),
                "sum_rho_t_i": safe_float(prefix_stats.get("sum_rho_t_i")),
                "bound_proxy_t": bound_proxy_t,
                "noise_max_calibrated_bound_t": noise_max_calibrated_bound_t,
                "expected_embedding_delta_norm": float(row.get("expected_embedding_delta_norm", "nan")),
                "empirical_lipschitz_C": float(row.get("empirical_lipschitz_C", "nan")),
                "ce_lipschitz_noise_max": ce_lipschitz_noise_max,
                "ce_lipschitz_noise_mean": float(row.get("ce_lipschitz_noise_mean", "nan")),
                "generated_step_correct": row["generated_step_correct"],
                "generated_step_edit_distance": row["generated_step_edit_distance"],
            }
        )

fieldnames = [
    "task",
    "jacobian_output",
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
    "sum_rho_t_i",
    "bound_proxy_t",
    "noise_max_calibrated_bound_t",
    "expected_embedding_delta_norm",
    "empirical_lipschitz_C",
    "ce_lipschitz_noise_max",
    "ce_lipschitz_noise_mean",
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
            "mean_sum_rho_t_i": mean("sum_rho_t_i"),
            "mean_bound_proxy_t": mean("bound_proxy_t"),
            "mean_noise_max_calibrated_bound_t": mean("noise_max_calibrated_bound_t"),
            "mean_expected_embedding_delta_norm": mean("expected_embedding_delta_norm"),
            "mean_empirical_lipschitz_C": mean("empirical_lipschitz_C"),
            "mean_ce_lipschitz_noise_max": mean("ce_lipschitz_noise_max"),
            "mean_ce_lipschitz_noise_mean": mean("ce_lipschitz_noise_mean"),
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
    "mean_sum_rho_t_i",
    "mean_bound_proxy_t",
    "mean_noise_max_calibrated_bound_t",
    "mean_expected_embedding_delta_norm",
    "mean_empirical_lipschitz_C",
    "mean_ce_lipschitz_noise_max",
    "mean_ce_lipschitz_noise_mean",
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
