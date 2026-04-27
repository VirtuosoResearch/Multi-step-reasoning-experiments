#!/usr/bin/env bash

set -euo pipefail
task1="${1:-}" # default to "bfs" if not provided
task2="${2:-}" # optional second task for multi-task training, default to empty string if not provided
device="${3:-}" # default to "0" if not provided
if [ -z "$task1" ]; then
  task1="bfs"
fi
if [ -z "$task2" ]; then
  task2=""
fi
if [ -z "$device" ]; then
  device="0"
fi
# Length generalization setup:
# train on lengths 4 and 5, evaluate on lengths 10, 11, and 12.
TRAIN_LENGTHS=(4 5)
TEST_LENGTHS=(10)

# Number of intermediate steps to evaluate.
# Metrics will include: <task>_step_0_accuracy ... <task>_step_N_accuracy and <task>_step_final_accuracy.
EVAL_STEP_NUM=20

MODEL_KEY="Qwen/Qwen2.5-1.5B"
DEVICES=("$device")
BATCH_SIZE=4
INFERENCE_BATCH_SIZE=4
RUNS=2
LR=2e-5
EPOCHS=10
PRECISION="bf16-true"
LORA_RANK=16
LORA_ALPHA=128
DOWNSAMPLE_RATIO=0.01
MINIMUM_SAMPLES_VALIDATION=100

# Use task-specific max lengths to avoid truncation for longer reasoning traces.
for task in "$task1" "$task2"
do
  # Skip the optional second task when it is not provided.
  if [ -z "$task" ]; then
    continue
  fi

  case "$task" in
    bfs)
      MAX_LENGTH=345
      MAX_OUTPUT_LENGTH=260
      ;;
    dfs)
      MAX_LENGTH=338
      MAX_OUTPUT_LENGTH=610
      ;;
    mst_kruskal)
      MAX_LENGTH=509
      MAX_OUTPUT_LENGTH=210
      ;;
    mst_prim)
      MAX_LENGTH=615
      MAX_OUTPUT_LENGTH=405
      ;;
    bellman_ford)
      MAX_LENGTH=458
      MAX_OUTPUT_LENGTH=200
      ;;
    dijkstra)
      MAX_LENGTH=456
      MAX_OUTPUT_LENGTH=219
      ;;
    *)
      echo "Unknown task: $task"
      exit 1
      ;;
  esac

  for sample in 5000
  do
    python train_clrs_text.py \
      --task_names "$task" \
      --model_key "$MODEL_KEY" \
      --devices "${DEVICES[@]}" \
      --batch_size "$BATCH_SIZE" \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      --max_output_length "$MAX_OUTPUT_LENGTH" \
      --train_lengths "${TRAIN_LENGTHS[@]}" \
      --test_lengths "${TEST_LENGTHS[@]}" \
      --generate_output \
      --eval_last_step \
      --eval_step_num "$EVAL_STEP_NUM" \
      --eval_test_during_fit \
      --runs "$RUNS" \
      --lr "$LR" \
      --save_name "clrs_${task}_len_gen_45_to_101112" \
      --epochs "$EPOCHS" \
      --precision "$PRECISION" \
      --train_lora \
      --lora_rank "$LORA_RANK" \
      --lora_alpha "$LORA_ALPHA" \
      --few_shot_k 0 \
      --downsample_ratio "$DOWNSAMPLE_RATIO" \
      --minimum_samples "$sample" \
      --minimum_samples_validation "$MINIMUM_SAMPLES_VALIDATION"
  done
done
