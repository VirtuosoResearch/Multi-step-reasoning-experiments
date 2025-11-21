for samples in 2000 4000 6000 8000
do
python train_math.py --task_names "gsm8k" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 32 --generate_output --runs 2 --lr 2e-5 --only_answer_output \
    --save_name gsm8k --epochs 5 --precision "bf16-true" \
    --eval_split 0.02 --downsample_ratio 0.001 --minimum_samples $samples --minimum_samples_validation 1400 \
    --train_lora --lora_rank 16 --lora_alpha 128
done

# python train_math.py --task_names "gsm8k" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 32 --generate_output --runs 2 --lr 2e-5 --only_answer_output \
#     --save_name gsm8k --epochs 10 --precision "bf16-true" \
#     --eval_split 0.02 --downsample_ratio 0.001 --minimum_samples 2000 --minimum_samples_validation 1400 