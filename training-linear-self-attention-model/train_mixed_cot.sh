#!/bin/bash

# Example script for training with mixed CoT generation
# - 1% of data has ground-truth CoT
# - 99% of data has no CoT initially
# - Every 100 steps, generate CoT for no-CoT data and filter by error threshold
T=20
for interval in 10 20
do
python train.py \
    --input_dim 10 \
    --n_examples 20 \
    --n_train_tasks 1000000 \
    --n_test_tasks 1000 \
    --noise_std 0.0 \
    --gd_lr 0.4 \
    --T $T \
    --batch_size 1000 \
    --epochs 1000 \
    --lr 0.001 \
    --lr_min 1e-5 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --device 1 \
    --eval_interval 10 \
    --sigma 0.002 \
    --use_wandb \
    --save_dir "checkpoints/linear_attention_model_input10_examples20_lr0.001_bs1000_T${T}_mixed_cot_ratio0.0001_interval${interval}_0.5_with_noise_no_gen" \
    --wandb_project linear-self-attention-weight-prediction \
    --wandb_run_name "input10_examples20_T${T}_seed${seed}_training_tasks1e6_cot_ratio0.0001_interval${interval}_0.5_with_noise_no_gen" \
    --seed 0 \
    --cot_ratio 0.0001 \
    --regen_interval $interval \
    --cot_error_threshold 0.1 \
    --use_noise_injection --train_noise_sigma 0.001 
    # --cot_generate_inject_noise --cot_generate_noise_sigma 0.001
done

    # --use_scheduler \