python train_graphwiz.py --task_names "flow"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_stl --epochs 12 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" 

python train_graphwiz.py --task_names "shortest"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_stl --epochs 12 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" 

python train_graphwiz.py --task_names "topology" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_stl --epochs 12 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" 

python train_graphwiz.py --task_names "triangle" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_stl --epochs 12 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" 
