for model in "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B"
do

for task_name in 'edge_existence' 'node_degree' 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 'disconnected_nodes' 'reachability' 'shortest_path' 'maximum_flow' 'triangle_counting' 'node_classification'
do
python train.py --task_names $task_name --prompt_styles "zero_shot" --text_encoders "adjacency"\
    --model_key $model \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name test --epochs 0 --precision "bf16-true"  --write_results
done

for task_name in 'edge_existence' 'node_degree' 'node_count' 'edge_count' 'connected_nodes' 'cycle_check' 'disconnected_nodes' 'reachability' 'shortest_path' 'maximum_flow' 'triangle_counting' 'node_classification'
do
python train.py --task_names $task_name --prompt_styles "zero_shot" --text_encoders "incident"\
    --model_key $model \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name test --epochs 0 --precision "bf16-true"  --write_results
done

done

python train_graphqa.py --task_names 'triangle_counting' --prompt_styles "zero_shot" --text_encoders "incident"\
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 64 --generate_output --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name test --epochs 0 --precision "bf16-true"  --write_results --min_nodes 10 --max_nodes 11