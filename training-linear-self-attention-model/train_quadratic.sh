# Training script for linear attention model on weight prediction
# Input dimension: 10
# Number of examples: 20
# Learning rate: 0.001
# Batch size: 1000

# Usage: sh train_quadratic.sh [T_value]
# Example: sh train_quadratic.sh 40

# Set T from command line argument, default to 20 if not provided
T=${1:-200}
device=${2:-0}

for seed in 0
do
python train.py \
  --input_dim 10 \
  --n_examples 200 \
  --n_train_tasks 1000000 \
  --n_test_tasks 1000 \
  --noise_std 0.0 \
  --gd_lr 0.001 \
  --T $T \
  --batch_size 1000 \
  --epochs 1 \
  --lr 0.001 \
  --lr_min 1e-5 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --num_workers 4 \
  --save_dir "checkpoints/nonlinear_attention_model_input10_examples200_lr0.001_bs1000_T${T}_quad_v2" \
  --device $device \
  --eval_interval 10 \
  --sigma 0.002 \
  --seed $seed \
  --use_quadratic_functions \
  --use_wandb \
  --wandb_project linear-self-attention-weight-prediction \
  --wandb_run_name "input10_examples200_T${T}_seed${seed}_training_tasks1e6_quad"
done

# python train.py \
#   --input_dim 10 \
#   --n_examples 200 \
#   --n_train_tasks 1000000 \
#   --n_test_tasks 1000 \
#   --noise_std 0.0 \
#   --gd_lr 0.001 \
#   --T 100 \
#   --batch_size 1000 \
#   --epochs 0 \
#   --lr 0.001 \
#   --lr_min 1e-5 \
#   --weight_decay 0.0 \
#   --grad_clip 1.0 \
#   --num_workers 4 \
#   --save_dir "checkpoints/nonlinear_attention_model_input10_examples200_lr0.001_bs1000_T${T}_quad_v5" \
#   --device 1 \
#   --eval_interval 10 \
#   --sigma 0.002 \
#   --seed 0 \
#   --use_quadratic_functions