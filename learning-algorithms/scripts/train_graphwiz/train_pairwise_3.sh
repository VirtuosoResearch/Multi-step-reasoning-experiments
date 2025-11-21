task_names=("connectivity" "bipartite" "cycle" "flow" "hamilton" "shortest" "substructure" "topology" "triangle")

length=${#task_names[@]}

for ((i = 2; i < 3; i++)); do
  for ((j = i + 1; j < $length; j++)); do
    python train_graphwiz.py --task_names "${task_names[$i]}" "${task_names[$j]}"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 512 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_pair --epochs 3 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" --write_results 
  done
done