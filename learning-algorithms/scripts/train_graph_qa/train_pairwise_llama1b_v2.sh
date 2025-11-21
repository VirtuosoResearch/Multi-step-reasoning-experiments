task_names=('maximum_flow' 'triangle_counting' 'node_classification')
length=${#task_names[@]}

python train.py --task_names 'cycle_check' 'triangle_counting' --prompt_styles "zero_shot" "zero_shot" --text_encoders "incident" "incident"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name pariwise --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
  
python train.py --task_names 'cycle_check' 'node_classification' --prompt_styles "zero_shot" "zero_shot" --text_encoders "incident" "incident"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name pariwise --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000

for ((i = 0; i < $length; i++)); do
  for ((j = i + 1; j < $length; j++)); do
    python train.py --task_names "${task_names[$i]}" "${task_names[$j]}" --prompt_styles "zero_shot" "zero_shot" --text_encoders "incident" "incident"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name pariwise --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
  done
done