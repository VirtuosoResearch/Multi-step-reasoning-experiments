for size in 100 200 500 1000
do
python train_graphqa.py --task_names "maximum_flow" --prompt_styles "zero_shot" --text_encoders "adjacency" --min_nodes 15 --max_nodes 16\
    --model_key "Qwen/Qwen2.5-1.5B" --disable_checkpointing \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 820 --max_output_length 4 --runs 3 --lr 2e-5 \
    --save_name scaling --epochs 20 --precision "bf16-true" --downsample_ratio 0.01 --minimum_samples $size 
done

for size in 2000 5000 10000 
do
python train_graphqa.py --task_names "maximum_flow" --prompt_styles "zero_shot" --text_encoders "adjacency" --min_nodes 15 --max_nodes 16\
    --model_key "Qwen/Qwen2.5-1.5B" --disable_checkpointing \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 820 --max_output_length 4 --runs 1 --lr 2e-5 \
    --save_name scaling --epochs 10 --precision "bf16-true" --downsample_ratio 0.01 --minimum_samples $size 
done
