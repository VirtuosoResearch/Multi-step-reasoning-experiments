for dim in 50000 100000 200000 25000
do
OS_CUDA_VISIBLE_DEVICES=0 python experiments/train.py --config-file=config/sublora_train.yaml \
                            --data.dataset_dir="./data" \
                            --login.out_dir="./checkpoints" \
                            --sublora.intrinsic_dim=$dim 
done