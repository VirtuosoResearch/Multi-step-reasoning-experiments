# Training script for linear attention model on weight prediction
# Input dimension: 10
# Number of examples: 20
# Learning rate: 0.001
# Batch size: 1000

for seed in 0
do
for T in 20
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
  --epochs 1 \
  --lr 0.001 \
  --lr_min 1e-5 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --num_workers 4 \
  --save_dir "checkpoints/linear_attention_model_input10_examples20_lr0.001_bs1000_T${T}" \
  --device 1 \
  --eval_interval 10 \
  --sigma 0.002 \
  --use_wandb \
  --wandb_project linear-self-attention-weight-prediction \
  --wandb_run_name "input10_examples20_T${T}_seed${seed}_training_tasks1e6" \
  --seed $seed 
  # --use_softmax
  # --use_noise_injection --train_noise_sigma 0.001
done
done