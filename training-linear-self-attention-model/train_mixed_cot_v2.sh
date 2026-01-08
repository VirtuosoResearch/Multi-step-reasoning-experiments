#!/bin/bash

# Example script for training with mixed CoT generation
# - 1% of data has ground-truth CoT
# - 99% of data has no CoT initially
# - Every 100 steps, generate CoT for no-CoT data and filter by error threshold

for interval in 10 20 30 40 50
do
python train.py \
    --input_dim 10 \
    --n_examples 20 \
    --n_train_tasks 1000000 \
    --n_test_tasks 1000 \
    --noise_std 0.0 \
    --gd_lr 0.4 \
    --T 20 \
    --batch_size 1000 \
    --epochs 1000 \
    --lr 0.001 \
    --lr_min 1e-5 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --device 0 \
    --eval_interval 10 \
    --sigma 0.002 \
    --use_wandb \
    --wandb_project linear-self-attention-weight-prediction \
    --wandb_run_name "input10_examples20_T${T}_seed${seed}_training_tasks1e6_cot_ratio0.001_interval${interval}_noise" \
    --seed 0 \
    --cot_ratio 0.001 \
    --regen_interval $interval \
    --cot_error_threshold 0.1 \
    --cot_inject_noise \
    --cot_noise_sigma 0.001 
done

    # --use_scheduler \