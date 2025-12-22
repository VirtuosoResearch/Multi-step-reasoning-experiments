#!/bin/bash
# Example script to compute bounds for CLRS text models

CHECKPOINT_PATH_LIST=(
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_10000_run_0/epoch_epoch=6.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_10000_run_1/epoch_epoch=7.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_50000_run_0/epoch_epoch=9.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_50000_run_1/epoch_epoch=5.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_100000_run_0/epoch_epoch=6.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_100000_run_1/epoch_epoch=5.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_500000_run_0/epoch_epoch=5.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_500000_run_1/epoch_epoch=8.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_1000000_run_0/epoch_epoch=9.pt"
    "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_1000000_run_1/epoch_epoch=9.pt"
)

dims=(10000 10000 50000 50000 100000 100000 500000 500000 1000000 1000000)
# CHECKPOINT_PATH="./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_10000_run_0/epoch_epoch=6.pt"
# intrinsic_dim=10000

for i in "${!CHECKPOINT_PATH_LIST[@]}"; do
    CHECKPOINT_PATH=${CHECKPOINT_PATH_LIST[$i]}
    intrinsic_dim=${dims[$i]}
    

    echo "Processing checkpoint: ${CHECKPOINT_PATH} with intrinsic dim: ${intrinsic_dim}"
    # Find the best checkpoint
    if [ ! -f ${CHECKPOINT_PATH} ]; then
        echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
        echo "Please specify the correct checkpoint path"
        exit 1
    fi

    # Configuration
    MODEL_KEY="Qwen/Qwen2.5-1.5B"
    TASK_NAME="dfs"
    TRAIN_LENGTH=10

    # Output directory
    OUTPUT_DIR="./outputs/bounds_results/$(basename $(dirname ${CHECKPOINT_PATH}))"
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
        --intrinsic_dim ${intrinsic_dim} \
        --eval_batch_size 8 \
        --bound_samples 5000 \
        --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 5000 --minimum_samples_validation 100 \
        --levels 11 \
        --misc_extra_bits 5 \
        --output_dir ${OUTPUT_DIR} \
        --device 0

    echo ""
    echo "Bound computation complete!"
    echo "Results saved to: ${OUTPUT_DIR}/bounds.yml"
done