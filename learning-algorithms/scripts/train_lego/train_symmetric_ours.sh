sample=2000
device=0


# Cyclic task - Full steps (ratio 1.0)
# Dataset has 5000 training examples, all with length 8
for quant_bits in 1 2 3 4
do
python train_clrs_text.py --task_names "symmetric" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 150 --train_lengths 5 --test_lengths 5 --runs 2 --lr 2e-5 \
    --save_name symmetric_quant_${quant_bits} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
    --use_quant --quant_training_mode lora --quant_w_bits "${quant_bits}" 
done

