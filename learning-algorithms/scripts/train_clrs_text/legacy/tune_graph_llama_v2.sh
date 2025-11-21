# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 32 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl_lora --epochs 10 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --only_load_last_output

# for lr in 1e-4 5e-5
# do
# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl_lora --epochs 10 --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" \
#     --use_graph_llama

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl_lora --epochs 10 --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" \
#     --use_graph_llama --use_cross_attn 

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl_lora --epochs 10 --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" \
#     --use_graph_llama --use_cross_attn --add_output_projection
# done

for weight in 0.1
do
# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-4\
#     --save_name tuning_stl_lora --epochs 10 --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" \
#     --use_graph_llama --alignment_loss_weight $weight

python train_clrs_text.py --task_names 'mst_prim' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 20 --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --use_graph_llama --alignment_loss_weight $weight

python train_clrs_text.py --task_names 'topological_sort' \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name tuning_stl --epochs 20  --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 --use_graph_llama --alignment_loss_weight $weight
done

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-4\
#     --save_name tuning_stl_lora --epochs 10 --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" \
#     --use_graph_llama