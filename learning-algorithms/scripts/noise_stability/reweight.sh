sample=2000

# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_wandb

# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.01

python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bellman --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.1


python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bellman --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.2

# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bellman --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.1



# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.03



# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.04


# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.06



# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.07

# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.1


# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.2


# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.3


# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 0.5

# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 1


# python train_clrs_text.py --task_names bellman_ford --model_key Qwen/Qwen2.5-1.5B --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 500 --max_output_length 240 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 --save_name clrs_bfs --epochs 8 --precision bf16-true --train_lora --lora_rank 16 --lora_alpha 128 --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 79 --eval_last_step --use_reweight --use_wandb --reweight_eta 5