# python train_graphqa.py --task_names "node_degree" --graph_types "er" --text_encoders "adjacency" --min_nodes 15 --max_nodes 16\
#     --model_key "Qwen/Qwen2.5-1.5B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 820 --max_output_length 4 --runs 2 --lr 2e-5 \
#     --save_name ood --epochs 10 --precision "bf16-true" --downsample_ratio 0.01 --minimum_samples 500 \
#     --remove_checkpoint \
#     --eval_task_names "node_degree" "node_degree" "node_degree" "node_degree" \
#     --eval_graph_types "er" "ba" "sbm" "sfn" --eval_text_encoders "adjacency" "adjacency" "adjacency" "adjacency" \
#     --eval_min_nodes 20 15 15 15 --eval_max_nodes 21 16 16 16 \
#     --eval_max_length 1580 --eval_max_output_length 4 

for alpha in 0.01 0.02 0.05 0.1 0.2 0.4 0.6 0.8 1
do
python train_graphqa.py --task_names "node_degree" --graph_types "er" --text_encoders "adjacency" --min_nodes 15 --max_nodes 16\
    --model_key "Qwen/Qwen2.5-1.5B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 820 --max_output_length 4 --runs 2 --lr 2e-5 \
    --save_name ood --epochs 10 --precision "bf16-true" --downsample_ratio 0.01 --minimum_samples 500 \
    --train_invariant_mix --invariant_mix_alpha $alpha --remove_checkpoint \
    --eval_task_names "node_degree" "node_degree" "node_degree" "node_degree" \
    --eval_graph_types "er" "ba" "sbm" "sfn" --eval_text_encoders "adjacency" "adjacency" "adjacency" "adjacency" \
    --eval_min_nodes 20 15 15 15 --eval_max_nodes 21 16 16 16 \
    --eval_max_length 1580 --eval_max_output_length 4 
done

# --train_invariant_mix --invariant_mix_alpha 0.1