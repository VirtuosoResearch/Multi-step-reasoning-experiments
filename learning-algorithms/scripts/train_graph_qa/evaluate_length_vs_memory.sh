python train.py --task_names "edge_existence" --prompt_styles "zero_shot" --text_encoders "incident"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 16384 --max_output_length 64 --generate_output --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name test --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 2000