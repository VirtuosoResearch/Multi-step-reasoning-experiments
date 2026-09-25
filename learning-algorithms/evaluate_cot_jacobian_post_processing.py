import csv
import math
import os
import sys

output_dir = sys.argv[1]
loss_path = os.path.join(output_dir, "generated_step_loss_rows.csv")
rho_path = os.path.join(output_dir, "rho_prefix_rows.csv")
jacobian_path = os.path.join(output_dir, "jacobian_rows.csv")
terminal_jacobian_path = os.path.join(output_dir, "terminal_jacobian_rows.csv")
second_order_path = os.path.join(output_dir, "second_order_rows.csv")
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

def preferred_loss(row, capped_name, full_name):
    capped = safe_float(row.get(capped_name))
    if math.isfinite(capped):
        return capped
    return safe_float(row.get(full_name))

def compute_gamma(intermediate_rho, max_target_step):
    gamma = {1: 1.0}
    for k in range(2, max_target_step + 1):
        candidates = []
        for j in range(1, k):
            candidates.append(
                1.0 + sum(intermediate_rho.get((i, j), 0.0) for i in range(j + 1, k))
            )
        gamma[k] = max(candidates) if candidates else 1.0
    return gamma

with open(rho_path, newline="") as f:
    rho_by_key = {key(row): row for row in csv.DictReader(f)}

jacobian_norms = {}
terminal_jacobian_norms = {}
second_order_values_by_sample_target = {}
train_errors = {}
max_target_step_by_sample = {}
embedding_final_state_by_sample = {}
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
        embedding_final_state_by_sample.setdefault(
            skey, row.get("embedding_final_state", "embedding")
        )

if os.path.exists(terminal_jacobian_path):
    with open(terminal_jacobian_path, newline="") as f:
        for row in csv.DictReader(f):
            skey = sample_key(row)
            target_step = int(row["i"])
            source_step = int(row["j"])
            terminal_jac_norm = safe_float(row.get("terminal_jac_norm"))
            train_error = safe_float(row.get("train_error_i"))
            if math.isfinite(terminal_jac_norm):
                terminal_jacobian_norms[(skey, target_step, source_step)] = terminal_jac_norm
                max_target_step_by_sample[skey] = max(
                    max_target_step_by_sample.get(skey, 0), target_step
                )
            if math.isfinite(train_error):
                train_errors.setdefault((skey, source_step), train_error)
            embedding_final_state_by_sample.setdefault(
                skey, row.get("embedding_final_state", "hidden_state")
            )

if os.path.exists(second_order_path):
    with open(second_order_path, newline="") as f:
        for row in csv.DictReader(f):
            skey = sample_key(row)
            target_step = int(row["i"])
            M_random_max = safe_float(row.get("M_random_max"))
            M_random_mean = safe_float(row.get("M_random_mean"))
            if math.isfinite(M_random_max):
                second_order_values_by_sample_target.setdefault((skey, target_step), []).append(
                    (M_random_max, M_random_mean)
                )
                max_target_step_by_sample[skey] = max(
                    max_target_step_by_sample.get(skey, 0), target_step
                )
            embedding_final_state_by_sample.setdefault(
                skey, row.get("embedding_final_state", "embedding")
            )

test_error_squares = {}
with open(loss_path, newline="") as f:
    for row in csv.DictReader(f):
        test_error_square = safe_float(row.get("test_error_square"))
        if not math.isfinite(test_error_square):
            test_error_norm = safe_float(row.get("test_error_norm"))
            if math.isfinite(test_error_norm):
                test_error_square = test_error_norm ** 2
        if math.isfinite(test_error_square):
            test_error_squares[(sample_key(row), int(row["t"]))] = test_error_square

prefix_bound_by_key = {}
embedding_prefix_bound_by_key = {}
second_order_prefix_by_key = {}
for skey, max_target_step in max_target_step_by_sample.items():
    recursive_rho = {}
    for target_step in range(1, max_target_step + 1):
        if target_step == 1:
            embedding_prefix_bound_by_key[(*skey, str(target_step))] = {
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

        embedding_prefix_bound_by_key[(*skey, str(target_step))] = {
            "sum_rho_t_i": sum_rho_t_i,
            "bound_proxy_t": bound_proxy_t,
        }

    # Terminal rows describe a different output space and apply only to the
    # target node named by that row.  In particular, Coconut has ordinary latent
    # nodes 1..L and a single hidden-state answer node T=L+1.  Treating the mere
    # presence of a terminal row as a sample-wide switch would incorrectly erase
    # every latent-prefix rho.
    effective_rho = dict(recursive_rho)
    for target_step in range(1, max_target_step + 1):
        has_terminal_for_target = any(
            key_s == skey and key_t == target_step
            for key_s, key_t, _ in terminal_jacobian_norms
        )
        if not has_terminal_for_target:
            prefix_bound_by_key[(*skey, str(target_step))] = embedding_prefix_bound_by_key[
                (*skey, str(target_step))
            ]
            continue

        sum_rho_t_i = 0.0
        bound_proxy_t = 0.0
        for source_step in range(1, target_step):
            direct_norm = terminal_jacobian_norms.get(
                (skey, target_step, source_step), float("nan")
            )
            if not math.isfinite(direct_norm):
                continue

            rho_value = direct_norm
            for middle_step in range(source_step + 1, target_step):
                terminal_to_middle = terminal_jacobian_norms.get(
                    (skey, target_step, middle_step), float("nan")
                )
                middle_to_source = recursive_rho.get(
                    (middle_step, source_step), float("nan")
                )
                if math.isfinite(terminal_to_middle) and math.isfinite(middle_to_source):
                    rho_value += terminal_to_middle * middle_to_source

            effective_rho[(target_step, source_step)] = rho_value
            sum_rho_t_i += rho_value
            train_error = train_errors.get((skey, source_step), float("nan"))
            if math.isfinite(train_error):
                bound_proxy_t += rho_value * train_error

        prefix_bound_by_key[(*skey, str(target_step))] = {
            "sum_rho_t_i": sum_rho_t_i,
            "bound_proxy_t": bound_proxy_t,
        }

    for target_step in range(1, max_target_step + 1):
        M_values = []
        M_mean_values = []
        for pair_target in range(1, target_step + 1):
            for M_random_max, M_random_mean in second_order_values_by_sample_target.get(
                (skey, pair_target),
                [],
            ):
                if math.isfinite(M_random_max):
                    M_values.append(M_random_max)
                if math.isfinite(M_random_mean):
                    M_mean_values.append(M_random_mean)
        M_random_max_t = max(M_values) if M_values else float("nan")
        M_random_mean_t = sum(M_mean_values) / len(M_mean_values) if M_mean_values else float("nan")
        test_error_square_sum_t = sum(
            test_error_squares.get((skey, source_step), 0.0)
            for source_step in range(1, target_step)
            if math.isfinite(test_error_squares.get((skey, source_step), float("nan")))
        )
        second_order_weighted_test_error_square_sum_t = 0.0
        for source_step in range(1, target_step):
            test_error_square = test_error_squares.get((skey, source_step), float("nan"))
            if not math.isfinite(test_error_square):
                continue
            remainder_weight = 1.0 + sum(
                effective_rho.get((target_step, middle_step), 0.0)
                for middle_step in range(source_step + 1, target_step)
            )
            second_order_weighted_test_error_square_sum_t += remainder_weight * test_error_square
        rho_remainder_multiplier_t = (
            second_order_weighted_test_error_square_sum_t / test_error_square_sum_t
            if test_error_square_sum_t > 1e-12
            else float("nan")
        )
        has_test_error_squares = any(
            math.isfinite(test_error_squares.get((skey, source_step), float("nan")))
            for source_step in range(1, target_step)
        )
        if math.isfinite(M_random_max_t) and (target_step == 1 or has_test_error_squares):
            second_order_geometry_amount_t = (
                0.5 * M_random_max_t * second_order_weighted_test_error_square_sum_t
            )
            second_order_geometry_coefficient_t = (
                second_order_geometry_amount_t / test_error_square_sum_t
                if test_error_square_sum_t > 1e-12
                else float("nan")
            )
        else:
            second_order_geometry_coefficient_t = float("nan")
            second_order_geometry_amount_t = float("nan")
        second_order_prefix_by_key[(*skey, str(target_step))] = {
            "M_random_max_t": M_random_max_t,
            "M_random_mean_t": M_random_mean_t,
            "gamma_t": float("nan"),
            "rho_remainder_multiplier_t": rho_remainder_multiplier_t,
            "test_error_square_sum_t": test_error_square_sum_t if target_step == 1 or has_test_error_squares else float("nan"),
            "second_order_weighted_test_error_square_sum_t": (
                second_order_weighted_test_error_square_sum_t
                if target_step == 1 or has_test_error_squares
                else float("nan")
            ),
            "second_order_geometry_coefficient_t": second_order_geometry_coefficient_t,
            "second_order_geometry_amount_t": second_order_geometry_amount_t,
        }

rows = []
with open(loss_path, newline="") as f:
    for row in csv.DictReader(f):
        rho_row = rho_by_key.get(key(row), {})
        prefix_stats = prefix_bound_by_key.get(key(row), {})
        embedding_prefix_stats = embedding_prefix_bound_by_key.get(key(row), {})
        teacher_loss = preferred_loss(row, "teacher_step_loss_capped", "teacher_step_loss")
        generated_loss = preferred_loss(
            row,
            "generated_step_loss_capped",
            "generated_prefix_step_loss",
        )
        bound_proxy_t = safe_float(prefix_stats.get("bound_proxy_t"))
        # C from Equation 4 is the Lipschitz constant of the loss in its first
        # argument. C_segment is the analytic bound over the teacher->generated
        # segment, valid for exactly the pair Equation 6 applies to. The
        # input-sensitivity number already carries a ||J|| factor and would
        # double-count against rho.
        C_segment = safe_float(row.get("C_segment"))
        segment_calibrated_bound_t = (
            C_segment * bound_proxy_t
            if math.isfinite(C_segment) and math.isfinite(bound_proxy_t)
            else float("nan")
        )
        C_random_max = safe_float(row.get("C_random_max"))
        random_max_calibrated_bound_t = (
            C_random_max * bound_proxy_t
            if math.isfinite(C_random_max) and math.isfinite(bound_proxy_t)
            else float("nan")
        )
        C_global = safe_float(row.get("C_global"))
        global_calibrated_bound_t = (
            C_global * bound_proxy_t
            if math.isfinite(C_global) and math.isfinite(bound_proxy_t)
            else float("nan")
        )
        pred_calibrated_bound_t = (
            global_calibrated_bound_t
            if row.get("evaluation_mode") == "coconut_text_proxy"
            else segment_calibrated_bound_t
        )
        empirical_lipschitz_C = safe_float(row.get("empirical_lipschitz_C"))
        empirical_calibrated_bound_t = (
            empirical_lipschitz_C * bound_proxy_t
            if math.isfinite(empirical_lipschitz_C) and math.isfinite(bound_proxy_t)
            else float("nan")
        )
        second_stats = second_order_prefix_by_key.get(key(row), {})
        second_order_geometry_amount_t = safe_float(
            second_stats.get("second_order_geometry_amount_t")
        )
        segment_second_order_bound_t = (
            C_segment * second_order_geometry_amount_t
            if math.isfinite(C_segment) and math.isfinite(second_order_geometry_amount_t)
            else float("nan")
        )
        random_max_second_order_bound_t = (
            C_random_max * second_order_geometry_amount_t
            if math.isfinite(C_random_max) and math.isfinite(second_order_geometry_amount_t)
            else float("nan")
        )
        global_second_order_bound_t = (
            C_global * second_order_geometry_amount_t
            if math.isfinite(C_global) and math.isfinite(second_order_geometry_amount_t)
            else float("nan")
        )
        rows.append(
            {
                "task": row["task"],
                "jacobian_output": row.get("jacobian_output", "expected_embedding"),
                "jacobian_granularity": row.get("jacobian_granularity", "block"),
                "embedding_final_state": row.get("embedding_final_state", "embedding"),
                "evaluation_mode": row.get("evaluation_mode", "text_cot"),
                "intermediate_state_space": row.get("intermediate_state_space", ""),
                "terminal_state_space": row.get("terminal_state_space", ""),
                "proxy_target": row.get("proxy_target", ""),
                "num_latent_steps": row.get("num_latent_steps", ""),
                "split": row["split"],
                "length": row["length"],
                "sample_idx": row["sample_idx"],
                "t": row["t"],
                "num_steps": row["num_steps"],
                "num_gold_steps": row.get("num_gold_steps", ""),
                "L_t_CoT": teacher_loss,
                "L_t_generated": generated_loss,
                "L_t_gap": (
                    generated_loss - teacher_loss
                    if math.isfinite(generated_loss) and math.isfinite(teacher_loss)
                    else float("nan")
                ),
                "adjacent_rho_t_1": float(rho_row.get("adjacent_rho_t_1", "nan")),
                "adjacent_log_rho_t_1": float(rho_row.get("adjacent_log_rho_t_1", "nan")),
                "recursive_rho_t_1": float(rho_row.get("recursive_rho_t_1", "nan")),
                "recursive_log_rho_t_1": float(rho_row.get("recursive_log_rho_t_1", "nan")),
                "embedding_adjacent_rho_t_1": safe_float(rho_row.get("embedding_adjacent_rho_t_1")),
                "embedding_adjacent_log_rho_t_1": safe_float(rho_row.get("embedding_adjacent_log_rho_t_1")),
                "embedding_recursive_rho_t_1": safe_float(rho_row.get("embedding_recursive_rho_t_1")),
                "embedding_recursive_log_rho_t_1": safe_float(rho_row.get("embedding_recursive_log_rho_t_1")),
                "sum_rho_t_i": safe_float(prefix_stats.get("sum_rho_t_i")),
                "bound_proxy_t": bound_proxy_t,
                "embedding_sum_rho_t_i": safe_float(embedding_prefix_stats.get("sum_rho_t_i")),
                "embedding_bound_proxy_t": safe_float(embedding_prefix_stats.get("bound_proxy_t")),
                "segment_calibrated_bound_t": segment_calibrated_bound_t,
                "random_max_calibrated_bound_t": random_max_calibrated_bound_t,
                "global_calibrated_bound_t": global_calibrated_bound_t,
                "pred_calibrated_bound_t": pred_calibrated_bound_t,
                "empirical_calibrated_bound_t": empirical_calibrated_bound_t,
                "M_random_max_t": safe_float(second_stats.get("M_random_max_t")),
                "M_random_mean_t": safe_float(second_stats.get("M_random_mean_t")),
                "gamma_t": safe_float(second_stats.get("gamma_t")),
                "rho_remainder_multiplier_t": safe_float(
                    second_stats.get("rho_remainder_multiplier_t")
                ),
                "test_error_square_sum_t": safe_float(second_stats.get("test_error_square_sum_t")),
                "second_order_weighted_test_error_square_sum_t": safe_float(
                    second_stats.get("second_order_weighted_test_error_square_sum_t")
                ),
                "second_order_geometry_coefficient_t": safe_float(
                    second_stats.get("second_order_geometry_coefficient_t")
                ),
                "second_order_geometry_amount_t": second_order_geometry_amount_t,
                "segment_second_order_bound_t": segment_second_order_bound_t,
                "random_max_second_order_bound_t": random_max_second_order_bound_t,
                "global_second_order_bound_t": global_second_order_bound_t,
                "segment_total_bound_t": (
                    segment_calibrated_bound_t + segment_second_order_bound_t
                    if math.isfinite(segment_calibrated_bound_t)
                    and math.isfinite(segment_second_order_bound_t)
                    else float("nan")
                ),
                "random_max_total_bound_t": (
                    random_max_calibrated_bound_t + random_max_second_order_bound_t
                    if math.isfinite(random_max_calibrated_bound_t)
                    and math.isfinite(random_max_second_order_bound_t)
                    else float("nan")
                ),
                "global_total_bound_t": (
                    global_calibrated_bound_t + global_second_order_bound_t
                    if math.isfinite(global_calibrated_bound_t)
                    and math.isfinite(global_second_order_bound_t)
                    else float("nan")
                ),
                "state_space": row.get("state_space", ""),
                "state_delta_norm": safe_float(row.get("state_delta_norm")),
                "test_error_norm": safe_float(row.get("test_error_norm")),
                "test_error_square": safe_float(row.get("test_error_square")),
                "empirical_lipschitz_C": empirical_lipschitz_C,
                "C_segment": C_segment,
                "C_point_teacher": safe_float(row.get("C_point_teacher")),
                "C_random_max": C_random_max,
                "C_random_mean": safe_float(row.get("C_random_mean")),
                "C_global": C_global,
                "input_sensitivity_noise_max": safe_float(row.get("input_sensitivity_noise_max")),
                "input_sensitivity_noise_mean": safe_float(row.get("input_sensitivity_noise_mean")),
                "generated_step_correct": row["generated_step_correct"],
                "generated_step_edit_distance": row["generated_step_edit_distance"],
            }
        )

fieldnames = [
    "task",
    "jacobian_output",
    "jacobian_granularity",
    "embedding_final_state",
    "evaluation_mode",
    "intermediate_state_space",
    "terminal_state_space",
    "proxy_target",
    "num_latent_steps",
    "split",
    "length",
    "sample_idx",
    "t",
    "num_steps",
    "num_gold_steps",
    "L_t_CoT",
    "L_t_generated",
    "L_t_gap",
    "adjacent_rho_t_1",
    "adjacent_log_rho_t_1",
    "recursive_rho_t_1",
    "recursive_log_rho_t_1",
    "embedding_adjacent_rho_t_1",
    "embedding_adjacent_log_rho_t_1",
    "embedding_recursive_rho_t_1",
    "embedding_recursive_log_rho_t_1",
    "sum_rho_t_i",
    "bound_proxy_t",
    "embedding_sum_rho_t_i",
    "embedding_bound_proxy_t",
    "segment_calibrated_bound_t",
    "random_max_calibrated_bound_t",
    "global_calibrated_bound_t",
    "pred_calibrated_bound_t",
    "empirical_calibrated_bound_t",
    "M_random_max_t",
    "M_random_mean_t",
    "gamma_t",
    "rho_remainder_multiplier_t",
    "test_error_square_sum_t",
    "second_order_weighted_test_error_square_sum_t",
    "second_order_geometry_coefficient_t",
    "second_order_geometry_amount_t",
    "segment_second_order_bound_t",
    "random_max_second_order_bound_t",
    "global_second_order_bound_t",
    "segment_total_bound_t",
    "random_max_total_bound_t",
    "global_total_bound_t",
    "state_space",
    "state_delta_norm",
    "test_error_norm",
    "test_error_square",
    "empirical_lipschitz_C",
    "C_segment",
    "C_point_teacher",
    "C_random_max",
    "C_random_mean",
    "C_global",
    "input_sensitivity_noise_max",
    "input_sensitivity_noise_mean",
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
            "evaluation_mode": group[0].get("evaluation_mode", "text_cot"),
            "intermediate_state_space": group[0].get("intermediate_state_space", ""),
            "terminal_state_space": group[0].get("terminal_state_space", ""),
            "proxy_target": group[0].get("proxy_target", ""),
            "num_latent_steps": group[0].get("num_latent_steps", ""),
            "mean_L_t_CoT": mean("L_t_CoT"),
            "mean_L_t_generated": mean("L_t_generated"),
            "mean_L_t_gap": mean("L_t_gap"),
            "mean_adjacent_log_rho_t_1": mean("adjacent_log_rho_t_1"),
            "mean_recursive_log_rho_t_1": mean("recursive_log_rho_t_1"),
            "mean_sum_rho_t_i": mean("sum_rho_t_i"),
            "mean_bound_proxy_t": mean("bound_proxy_t"),
            "mean_embedding_sum_rho_t_i": mean("embedding_sum_rho_t_i"),
            "mean_embedding_bound_proxy_t": mean("embedding_bound_proxy_t"),
            "mean_segment_calibrated_bound_t": mean("segment_calibrated_bound_t"),
            "mean_random_max_calibrated_bound_t": mean("random_max_calibrated_bound_t"),
            "mean_global_calibrated_bound_t": mean("global_calibrated_bound_t"),
            "mean_pred_calibrated_bound_t": mean("pred_calibrated_bound_t"),
            "mean_empirical_calibrated_bound_t": mean("empirical_calibrated_bound_t"),
            "mean_M_random_max_t": mean("M_random_max_t"),
            "mean_M_random_mean_t": mean("M_random_mean_t"),
            "mean_gamma_t": mean("gamma_t"),
            "mean_rho_remainder_multiplier_t": mean("rho_remainder_multiplier_t"),
            "mean_test_error_square_sum_t": mean("test_error_square_sum_t"),
            "mean_second_order_weighted_test_error_square_sum_t": mean(
                "second_order_weighted_test_error_square_sum_t"
            ),
            "mean_second_order_geometry_coefficient_t": mean("second_order_geometry_coefficient_t"),
            "mean_second_order_geometry_amount_t": mean("second_order_geometry_amount_t"),
            "mean_segment_second_order_bound_t": mean("segment_second_order_bound_t"),
            "mean_random_max_second_order_bound_t": mean("random_max_second_order_bound_t"),
            "mean_global_second_order_bound_t": mean("global_second_order_bound_t"),
            "mean_segment_total_bound_t": mean("segment_total_bound_t"),
            "mean_random_max_total_bound_t": mean("random_max_total_bound_t"),
            "mean_global_total_bound_t": mean("global_total_bound_t"),
            "mean_state_delta_norm": mean("state_delta_norm"),
            "mean_test_error_norm": mean("test_error_norm"),
            "mean_test_error_square": mean("test_error_square"),
            "mean_empirical_lipschitz_C": mean("empirical_lipschitz_C"),
            "mean_C_segment": mean("C_segment"),
            "mean_C_point_teacher": mean("C_point_teacher"),
            "mean_C_random_max": mean("C_random_max"),
            "mean_C_random_mean": mean("C_random_mean"),
            "mean_C_global": mean("C_global"),
            "mean_input_sensitivity_noise_max": mean("input_sensitivity_noise_max"),
            "mean_generated_step_correct": mean("generated_step_correct"),
            "mean_generated_step_edit_distance": mean("generated_step_edit_distance"),
        }
    )

summary_fields = [
    "t",
    "count",
    "evaluation_mode",
    "intermediate_state_space",
    "terminal_state_space",
    "proxy_target",
    "num_latent_steps",
    "mean_L_t_CoT",
    "mean_L_t_generated",
    "mean_L_t_gap",
    "mean_adjacent_log_rho_t_1",
    "mean_recursive_log_rho_t_1",
    "mean_sum_rho_t_i",
    "mean_bound_proxy_t",
    "mean_embedding_sum_rho_t_i",
    "mean_embedding_bound_proxy_t",
    "mean_segment_calibrated_bound_t",
    "mean_random_max_calibrated_bound_t",
    "mean_global_calibrated_bound_t",
    "mean_pred_calibrated_bound_t",
    "mean_empirical_calibrated_bound_t",
    "mean_M_random_max_t",
    "mean_M_random_mean_t",
    "mean_gamma_t",
    "mean_rho_remainder_multiplier_t",
    "mean_test_error_square_sum_t",
    "mean_second_order_weighted_test_error_square_sum_t",
    "mean_second_order_geometry_coefficient_t",
    "mean_second_order_geometry_amount_t",
    "mean_segment_second_order_bound_t",
    "mean_random_max_second_order_bound_t",
    "mean_global_second_order_bound_t",
    "mean_segment_total_bound_t",
    "mean_random_max_total_bound_t",
    "mean_global_total_bound_t",
    "mean_state_delta_norm",
    "mean_test_error_norm",
    "mean_test_error_square",
    "mean_empirical_lipschitz_C",
    "mean_C_segment",
    "mean_C_point_teacher",
    "mean_C_random_max",
    "mean_C_random_mean",
    "mean_C_global",
    "mean_input_sensitivity_noise_max",
    "mean_generated_step_correct",
    "mean_generated_step_edit_distance",
]
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Wrote {out_path}")
print(f"Wrote {summary_path}")
