sample=2000
device=1

# symmetric task - Full steps (ratio 1.0)
# Dataset has 5000 training examples, all with length 8

for steps_reduction_to_zero in 2000 4000
do
python train_clrs_text.py --task_names "symmetric" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 150 --train_lengths 5 --test_lengths 5 --runs 2 --lr 2e-5 \
    --save_name symmetric_implicit_cot_steps_${steps_reduction_to_zero} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
    --implicit_cot --steps_reduction_to_zero ${steps_reduction_to_zero}
done


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

    for steps_to_full_latent in 2000 4000
    do
    python train_clrs_text.py --task_names "symmetric" \
    --model_key "Qwen/Qwen2.5-1.5B" \
    --devices $device --batch_size 4 --inference_batch_size 4 --max_length 256 --max_output_length 150 --train_lengths 5 --test_lengths 5 --runs 2 --lr 2e-5 \
    --save_name symmetric_coconut_steps_${steps_to_full_latent} --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples $sample --minimum_samples_validation 100 --eval_last_step --reduce_steps_ratio 1.0 --generate_output \
    --use_coconut --coconut_steps_to_full_latent ${steps_to_full_latent} --coconut_latents_per_step ${latents_per_step} --coconut_max_train_latent_thoughts ${max_train_latent_thoughts} --coconut_eval_latent_thoughts ${eval_latent_thoughts} --coconut_gradient_checkpointing
    done
}