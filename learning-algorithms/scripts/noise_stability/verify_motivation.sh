# sample=10000

# # ------------- Dijkstra -------------
# python train_clrs_text.py --task_names "dijkstra" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
#     --save_name clrs_bfs --epochs 6 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step 

# python train_clrs_text.py --task_names "dijkstra" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
#     --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dijkstra_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15

# python train_clrs_text.py --task_names "dijkstra" \
#     --model_key "Qwen/Qwen2.5-1.5B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
#     --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
#     --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dijkstra_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
#     --add_weight_perturb --perturb_std 0.005


sample=5000

# ------------- Dijkstra -------------
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 8 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step 

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dijkstra_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dijkstra_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005


sample=2000

# ------------- Dijkstra -------------
python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 8 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step 

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dijkstra_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15

python train_clrs_text.py --task_names "dijkstra" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_dijkstra_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005


for sample in 2000 5000 10000
do

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 470 --max_output_length 230 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 8 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step 

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 470 --max_output_length 230 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_mst_prim_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15

python train_clrs_text.py --task_names "mst_prim" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 470 --max_output_length 230 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_mst_prim_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005

done




for sample in 2000 5000 10000
do

python train_clrs_text.py --task_names "mst_kruskal" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 520 --max_output_length 440 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 8 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step 

python train_clrs_text.py --task_names "mst_kruskal" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 520 --max_output_length 440 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_mst_kruskal_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15

python train_clrs_text.py --task_names "mst_kruskal" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 520 --max_output_length 440 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_bfs --epochs 0 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --load_model_dir Qwen-Qwen2.5-1.5B_mst_kruskal_${sample}_len_[10]_lora_r_16_clrs_bfs_run_0 --eval_step_num 15 \
    --add_weight_perturb --perturb_std 0.005

done