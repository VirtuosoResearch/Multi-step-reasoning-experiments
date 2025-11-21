for alpha in 0.02 0.01 0.005
do
python train_graphqa.py --task_names "node_degree" --prompt_styles "zero_shot" --text_encoders "adjacency" --min_nodes 15 --max_nodes 16\
    --model_key "Qwen/Qwen2.5-1.5B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 820 --max_output_length 4 --runs 2 --lr 2e-5 \
    --save_name scaling --epochs 10 --precision "bf16-true" --downsample_ratio 0.01 --minimum_samples 500 \
    --train_invariant_mix --invariant_mix_alpha 0.1 --remove_checkpoint
done

for alpha in 0.02 0.01 0.005
do
python train_graphqa.py --task_names "node_degree" --prompt_styles "zero_shot" --text_encoders "adjacency" --min_nodes 15 --max_nodes 16\
    --model_key "Qwen/Qwen2.5-1.5B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 820 --max_output_length 4 --runs 2 --lr 2e-5 \
    --save_name scaling --epochs 10 --precision "bf16-true" --downsample_ratio 0.01 --minimum_samples 1000 \
    --train_invariant_mix --invariant_mix_alpha 0.1 --remove_checkpoint
done