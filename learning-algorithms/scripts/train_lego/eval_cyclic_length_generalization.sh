device=1
sample=2000
quant_bits=1

for length in 8 9
do
python train_clrs_text.py --task_names "cyclic" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 150 --train_lengths $length --test_lengths $length --runs 1 --lr 2e-5 \
    --save_name cyclic_output_length_${length} --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
    --use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}" --load_model_dir "Qwen-Qwen2.5-1.5B_cyclic_500_len_[8]_lora_r_16_quant_lora_1bit_cyclic_quant_1_run_0/epoch_epoch=8.pt"
done