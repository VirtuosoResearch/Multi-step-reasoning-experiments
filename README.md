# Quantifying the Stability of Multi-Step Reasoning via Error Amplification

This repository contains the code for the paper "Quantifying the Stability of Multi-Step Reasoning via Error Amplification". 

We study how inference errors accumulate during multi-step reasoning. Our analysis uses the spectral norms of Jacobians taken through the input space to measure error amplification. This motivates a training method to regularizaing the norms based on (1) chain-of-thought length compression, and (2) quantization-aware training.

The repository also includes synthetic experiments with one-layer transformers trained on linear and quadratic weight-prediction tasks.

## Setup

The language-model experiments use Python 3.10 and a CUDA-enabled GPU. From the repository root, run:

```bash
conda env create -f learning-algorithms/environment.yml
conda activate llama-env
```

Install the additional dependencies for the synthetic transformer experiments with:

```bash
pip install -r training-linear-self-attention-model/requirements.txt
```

## Evaluate the inference error bound

The evaluation has three steps: train a checkpoint, estimate its input Jacobians, and aggregate the resulting bound.

### 1. Train a checkpoint

We provide launchers for the models used in the bound evaluation. Edit the `device` variable and keep the task configurations you need, then run one of the following scripts:

```bash
cd learning-algorithms

# Llama-3.2-1B checkpoints
bash scripts/train_clrs_text/motivation_exps/train_llama.sh

# Gemma-2-2B checkpoints
bash scripts/train_clrs_text/motivation_exps/train_gemma.sh

# Qwen2.5-1.5B example: Dijkstra checkpoints at different reasoning lengths
bash scripts/train_clrs_text/motivation_exps/train_dijkstra_varying_lengths_v1.sh
```

Checkpoints are written to:

```text
learning-algorithms/external_lightning_logs/<experiment_name>/epoch_epoch=<N>.pt
```

Other task-specific launchers are available in `scripts/train_clrs_text/motivation_exps/`.

### 2. Evaluate the Jacobians

Edit `model_key`, `checkpoint_path`, `task_name`, and `device` near the top of:

```text
learning-algorithms/scripts/train_clrs_text/evaluate_jacobians/evaluate_jacobian_scaling_expected_embedding.sh
```

Then run:

```bash
cd learning-algorithms
bash scripts/train_clrs_text/evaluate_jacobians/evaluate_jacobian_scaling_expected_embedding.sh
```

The evaluator estimates the spectral norms of all step-to-step input Jacobians with power iteration. It then computes the error amplification factors and compares the bound with the autoregressive loss gap.

### 3. Inspect the results

Results are saved under `learning-algorithms/jacobian_results/`.

| File | Description |
| --- | --- |
| `jacobian_rows.csv` | Estimated Jacobian norm for each source-target step pair |
| `test_loss_scaling_rows.csv` | Per-example loss gaps and bound components |
| `test_loss_scaling_summary.csv` | Results averaged by reasoning length |

## Run the proposed algorithm

The main arguments of our method are:

- `--reduce_steps_ratio`: fraction of intermediate reasoning steps retained;
- `--use_quant --quant_training_mode lora --quant_w_bits B`: enable `B`-bit quantization-aware training.

For example, run the Dijkstra configuration with:

```bash
cd learning-algorithms
python train_clrs_text.py \
  --task_names dijkstra \
  --model_key Qwen/Qwen2.5-1.5B \
  --devices 0 --precision bf16-true \
  --batch_size 4 --inference_batch_size 4 \
  --train_lengths 10 --test_lengths 10 \
  --max_length 460 --max_output_length 45 \
  --downsample_ratio 0.01 \
  --minimum_samples 2000 --minimum_samples_validation 100 \
  --epochs 10 --runs 3 --lr 2e-5 \
  --train_lora --lora_rank 16 --lora_alpha 128 \
  --reduce_steps_ratio 0.1 \
  --use_quant --quant_training_mode lora --quant_w_bits 1 \
  --eval_last_step --generate_output \
  --save_name clrs_dijkstra_ours
```

Change the task, sampling ratio, and quantization bit width to run other graph experiments. Trained checkpoints are saved under `learning-algorithms/external_lightning_logs/`.

The LEGO state-tracking experiments have separate launchers:

```bash
cd learning-algorithms
bash scripts/train_lego/train_cyclic_ours.sh
bash scripts/train_lego/train_symmetric_ours.sh
```

## Train transformers on linear and quadratic functions

The synthetic experiments train a one-head linear transformer on linear functions and a three-head nonlinear transformer on quadratic functions.

```bash
cd training-linear-self-attention-model

# Linear functions with T=20
bash train.sh

# Quadratic functions with T=200 
bash train_quadratic.sh 200 0
```

- Checkpoints and evaluation results are written to `training-linear-self-attention-model/checkpoints/`. Edit `--device` in `train.sh` if needed. 
- Weights & Biases logging is enabled in both scripts; remove the `--use_wandb` options to disable it.

## Project structure

```text
.
├── learning-algorithms/                  
│   ├── train_clrs_text.py                # Graph and LEGO training entry point
│   ├── evaluate_cot_jacobian.py          # Jacobian and bound evaluation
│   ├── evaluate_cot_jacobian_post_processing.py # Bound summaries
│   ├── clrs_text_tasks/                  # CLRS task definitions and encoders
│   ├── data/                             # LEGO and generated task data
│   ├── src/                              # Models, data modules, and quantization
│   └── scripts/
│       ├── train_clrs_text/              # Graph training and bound evaluation
│       ├── train_lego/                   # Symbolic state-tracking experiments
│
├── training-linear-self-attention-model/ 
│   ├── train.py                          # Model and training implementation
│   ├── train.sh                          # Linear-function experiment
│   ├── train_quadratic.sh                # Quadratic-function experiment
```

## Reference

If you find this repository useful or use it in your research, please cite our work:

```bibtex
@inproceedings{li2026quantifying,
  title={Quantifying the Stability of Multi-Step Reasoning via Error Amplification},
  author={Li, Dongyue and Zhang, Ziniu and Duan, Minxuan and Zhang, Hongyang R.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```