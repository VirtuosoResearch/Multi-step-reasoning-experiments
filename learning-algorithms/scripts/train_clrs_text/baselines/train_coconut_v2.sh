#!/usr/bin/env bash

sample=2000
device=1
batch_size=4
accumulate=1
latents_per_step=1
max_train_latent_thoughts=4
eval_latent_thoughts=4

run_coconut() {
    local task_name=$1
    local save_task_name=$2
    local max_length=$3
    local max_output_length=$4

    for steps_to_full_latent in 2000
    do
    python train_clrs_text.py --task_names "${task_name}" \
        --model_key "Qwen/Qwen2.5-1.5B" \
        --devices $device --batch_size ${batch_size} --inference_batch_size 1 --accumulate ${accumulate} --max_length ${max_length} --max_output_length ${max_output_length} --train_lengths 10 --test_lengths 10 --runs 2 --lr 2e-5 \
        --save_name clrs_${save_task_name}_coconut_steps_${steps_to_full_latent}_train_latents_${max_train_latent_thoughts}_eval_latents_${eval_latent_thoughts} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
        --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
        --use_coconut --coconut_steps_to_full_latent ${steps_to_full_latent} --coconut_latents_per_step ${latents_per_step} --coconut_max_train_latent_thoughts ${max_train_latent_thoughts} --coconut_eval_latent_thoughts ${eval_latent_thoughts} --coconut_gradient_checkpointing
    done
}

# run_coconut "mst_prim" "mst_prim" 460 230
# run_coconut "dijkstra" "dijkstra" 460 220
run_coconut "bellman_ford" "bellmanford" 460 180
run_coconut "bfs" "bfs" 256 180
run_coconut "dfs" "dfs" 256 660
