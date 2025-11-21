for size in 1000 2000 5000 10000
do
python train_graphqa.py --task_names "triangle_counting" --prompt_styles "zero_shot" --text_encoders "adjacency" --min_nodes 20 --max_nodes 21\
    --model_key "meta-llama/Llama-3.2-1B" --use_graph_llama \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 64 --max_output_length 2 --generate_output --runs 1 --lr 2e-5 \
    --save_name test --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.01 --minimum_samples $size 
done

# --use_graph_llama
python train_graphqa.py --task_names "triangle_counting" --prompt_styles "zero_shot" --text_encoders "adjacency" --min_nodes 20 --max_nodes 21\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --max_output_length 2 --generate_output --runs 1 --lr 2e-5 \
    --save_name test --epochs 10 --precision "bf16-true" --write_results --downsample_ratio 0.01 --minimum_samples 1000