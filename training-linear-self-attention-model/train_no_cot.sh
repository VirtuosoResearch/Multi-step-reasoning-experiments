# Training script for linear attention model on weight prediction
# Input dimension: 10
# Number of examples: 20
# Learning rate: 0.001
# Batch size: 1000

python train.py \
  --input_dim 10 \
  --n_examples 20 \
  --n_train_tasks 1000000 \
  --n_test_tasks 1000 \
  --noise_std 0.0 \
  --gd_lr 0.4 \
  --T 20 \
  --batch_size 1000 \
  --epochs 2 \
  --lr 0.001 \
  --lr_min 1e-5 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --seed 42 \
  --num_workers 4 \
  --save_dir checkpoints/linear_attention_model_input10_examples20_lr0.001_bs1000_T20_no_cot \
  --device 0 \
  --eval_interval 50 \
  --no_cot --sigma 0.002

#   --use_wandb \
#   --wandb_project linear-attention-weight-prediction \
#   --wandb_run_name input10_examples20_lr0.001_bs1000
#   --use_scheduler \
