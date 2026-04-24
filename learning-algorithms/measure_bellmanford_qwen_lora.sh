#!/usr/bin/env bash

set -euo pipefail

# Optional positional args:
#   1) num_examples
#   2) num_perturbations
#   3) noise_std
#   4) batch_size
#   5) device
#   6) seed

NUM_EXAMPLES=20 #"${1:-8}"
NUM_PERTURBATIONS=50 #"${2:-20}"
NOISE_STD=0.02 #"${3:-1e-3}"
BATCH_SIZE=4 # "${4:-4}"
DEVICE=0 #"${5:-0}"
for SEED in 0 1 2 4 5 6 7 8 9 10; do

CHECKPOINT="./external_lightning_logs/Qwen-Qwen2.5-1.5B_bellman_ford_5000_lora_r_16_clrs_bellmanford_v1_run_0/epoch_epoch=9.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

python gradient_noise_correlation.py \
  --task_names bellman_ford \
  --model_key Qwen/Qwen2.5-1.5B \
  --train_lora \
  --lora_rank 16 \
  --load_model_path "$CHECKPOINT" \
  --num_examples "$NUM_EXAMPLES" \
  --num_perturbations "$NUM_PERTURBATIONS" \
  --noise_std "$NOISE_STD" \
  --batch_size "$BATCH_SIZE" \
  --devices "$DEVICE" \
  --seed "$SEED"
done