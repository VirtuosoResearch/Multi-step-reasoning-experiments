# for task_name in 'maximum_flow' # 'edge_existence' 'node_degree' 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 'disconnected_nodes' 'reachability' 'triangle_counting' 'node_classification'
# do
# for prompt_style in "er" "ba" "path" "sbm" "sfn" "star" "complete"
# do
# python train.py --task_names $task_name --prompt_styles "zero_shot_${prompt_style}" --text_encoders "incident"\
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 32 --generate_output --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name stl_vary_generators --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
# done
# done

for task_name in 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 'disconnected_nodes' 'reachability' 'node_classification'
do
for prompt_style in "ba" "sbm" "sfn" 
do
python train.py --task_names $task_name --prompt_styles "zero_shot_${prompt_style}" --text_encoders "incident"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 32 --generate_output --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name stl_vary_generators --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.1 --minimum_samples 1000
done
done