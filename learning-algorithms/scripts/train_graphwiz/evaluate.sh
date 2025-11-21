# CUDA_VISIBLE_DEVICES=0,1 python evaluate_graphwiz.py

names=('cycle' 'connectivity' 'hamilton' 'substructure' 'bipartite' 'flow' 'shortest' 'triplet' 'topology')

for name in ${names[@]}
do
python train_graphwiz.py --task_names "$name" \
    --model_key "GraphWiz/LLaMA2-7B-DPO" \
    --devices 0 --batch_size 2 --inference_batch_size 2 --max_length 1025 --max_output_length 1025 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_evaluate --epochs 0 --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" --write_results
done 