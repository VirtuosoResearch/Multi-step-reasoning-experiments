#!/bin/bash
# Example script to compute bounds for CLRS text models

# Configuration
MODEL_KEY="Qwen/Qwen2.5-1.5B"
TASK_NAME="dfs"
TRAIN_LENGTH=10

# Find the best checkpoint
CHECKPOINT_PATH="./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_10000_run_0/epoch_epoch=6.pt"
if [ ! -f ${CHECKPOINT_PATH} ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Please specify the correct checkpoint path"
    exit 1
fi

# Output directory
OUTPUT_DIR="./outputs/bounds_results/${MODEL_KEY}_${TASK_NAME}_length_${TRAIN_LENGTH}"
mkdir -p ${OUTPUT_DIR}

echo "Computing bounds for CLRS text model"
echo "Task: ${TASK_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run bound computation
python compute_bounds.py \
    --model_key ${MODEL_KEY} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --task_names ${TASK_NAME} \
    --train_lengths ${TRAIN_LENGTH} \
    --test_lengths ${TRAIN_LENGTH} \
    --max_length 256 --max_output_length 660 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --intrinsic_dim 10000 \
    --eval_batch_size 8 \
    --bound_samples 5000 \
    --levels 11 \
    --misc_extra_bits 5 \
    --output_dir ${OUTPUT_DIR} \
    --device 0

echo ""
echo "Bound computation complete!"
echo "Results saved to: ${OUTPUT_DIR}/bounds.yml"
