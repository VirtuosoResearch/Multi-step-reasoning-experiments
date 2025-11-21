python train_clrs_text.py --task_names bfs \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-3\
    --save_name test_pretraining_new --epochs 20 --use_graph_llama \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --only_train_graph --test_classifier_before_cross_attn
# Accuracy: 0.9333333802223206

python train_clrs_text.py --task_names bfs \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-4\
    --save_name test_pretraining_cross_attn --epochs 20 --use_graph_llama \
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --load_model_dir "meta-llama-Llama-3.2-1B_bfs_test_pretraining_run_0/epoch_epoch=18.pt" --only_train_graph --use_cross_attn --freeze_graph_tower
# accuracy: 0.9380

python train_clrs_text.py --task_names 'dijkstra' \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-4\
    --save_name test_pretraining_new --epochs 40 --use_graph_llama \
    --few_shot_k 20 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --load_model_dir "/meta-llama-Llama-3.2-1B_dijkstra_test_pretraining_new_run_0/epoch_epoch=20.pt"
    --only_train_graph