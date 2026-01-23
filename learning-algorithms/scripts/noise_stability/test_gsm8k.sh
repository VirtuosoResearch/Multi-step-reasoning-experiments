model_keys=("Qwen/Qwen2.5-1.5B" "Qwen/Qwen3-4B-Instruct-2507" "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B")

for model_key in ${model_keys[@]}; do
python train_gsm8k.py \
    --model_key $model_key \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 512 --generate_output --runs 1 --lr 2e-5 \
    --save_name gsm8k --epochs 0 --precision "bf16-true" \
    --eval_split 0.02 --downsample_ratio 0.001 --minimum_samples 500 --minimum_samples_validation 89 \
    --train_lora --lora_rank 16 --lora_alpha 128 --eval_step_num 15


python train_gsm8k.py \
    --model_key $model_key \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 512 --generate_output --runs 1 --lr 2e-5 \
    --save_name gsm8k --epochs 0 --precision "bf16-true" \
    --eval_split 0.02 --downsample_ratio 0.001 --minimum_samples 500 --minimum_samples_validation 89 \
    --train_lora --lora_rank 16 --lora_alpha 128 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005
done


python train_gsm8k.py \
    --model_key "deepseek-ai/deepseek-math-7b-instruct" \
    --devices 0 --batch_size 2 --inference_batch_size 2 --max_length 256 --max_output_length 512 --generate_output --runs 1 --lr 2e-5 \
    --save_name gsm8k --epochs 0 --precision "bf16-true" \
    --eval_split 0.02 --downsample_ratio 0.001 --minimum_samples 500 --minimum_samples_validation 89 \
    --train_lora --lora_rank 16 --lora_alpha 128 --eval_step_num 15


python train_gsm8k.py \
    --model_key "deepseek-ai/deepseek-math-7b-instruct" \
    --devices 0 --batch_size 2 --inference_batch_size 2 --max_length 256 --max_output_length 512 --generate_output --runs 1 --lr 2e-5 \
    --save_name gsm8k --epochs 0 --precision "bf16-true" \
    --eval_split 0.02 --downsample_ratio 0.001 --minimum_samples 500 --minimum_samples_validation 89 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --add_weight_perturb --perturb_std 0.005 --eval_step_num 15