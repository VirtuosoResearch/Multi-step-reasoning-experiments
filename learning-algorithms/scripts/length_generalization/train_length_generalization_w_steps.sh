#!/usr/bin/env bash

set -euo pipefail
task="$1" # default to "bfs" if not provided
device="$2" # default to "0" if not provided
if [ -z "$task" ]; then
  task="bfs"
fi
if [ -z "$device" ]; then
  device="0"
fi
# Length generalization setup:
# train on lengths 4 and 5, evaluate on lengths 10, 11, and 12.
TRAIN_LENGTHS=(4 5)
TEST_LENGTHS=(10 11 12)

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
for task in "$task"
do
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
    mst_kruskal)
      MAX_LENGTH=615
      MAX_OUTPUT_LENGTH=405
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
