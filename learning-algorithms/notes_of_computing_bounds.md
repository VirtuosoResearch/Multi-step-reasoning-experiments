# Computing PAC-Bayes Bounds for CLRS Text Models

This guide explains how to compute generalization bounds for trained CLRS text models using the PAC-Bayes framework adapted from SubLoRA.

## Overview

The bound computation follows these steps:
1. **Load trained model** - Load a checkpoint from training
2. **Quantize parameters** - Compress trainable parameters to compute message length
3. **Compute empirical BPD** - Evaluate bits-per-dimension on training set
4. **Calculate bounds** - Apply PAC-Bayes bound formula

## Quick Start

### Example: Computing bounds for a trained DFS model

```bash
cd learning-algorithms

# Edit the checkpoint path in the script
vim scripts/compute_bounds_example.sh

# Run bound computation
bash scripts/compute_bounds_example.sh
```

## Usage

### Basic Command

```bash
python compute_clrs_bounds.py \
    --model_key gpt2 \
    --checkpoint_path path/to/checkpoint.pt \
    --task_names dfs \
    --train_lengths 10 \
    --train_lora \
    --lora_rank 4 \
    --output_dir ./bounds_results
```

### With Intrinsic Dimension

```bash
python compute_clrs_bounds.py \
    --model_key gpt2 \
    --checkpoint_path path/to/checkpoint.pt \
    --task_names dfs \
    --train_lengths 10 \
    --intrinsic_dim 5000 \
    --intrinsic_mode rdkronqr \
    --output_dir ./bounds_results
```

## Arguments

### Model Configuration
- `--model_key`: Model name (e.g., `gpt2`, `meta-llama/Llama-2-7b-hf`)
- `--checkpoint_path`: Path to trained model checkpoint (`.pt` file)
- `--device`: GPU device ID (default: 0)

### Parameter Efficiency
- `--train_lora`: Use LoRA (must match training configuration)
- `--lora_rank`: LoRA rank (default: 4)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--intrinsic_dim`: Intrinsic dimensionality (0 = disabled)
- `--intrinsic_mode`: Projection mode (default: `rdkronqr`)

### Data Configuration
- `--task_names`: CLRS task names (e.g., `dfs`, `bfs`)
- `--train_lengths`: Training sequence lengths
- `--max_length`: Maximum input length (default: 256)
- `--only_answer_output`: Only use final answer for evaluation

### Bound Computation
- `--eval_batch_size`: Batch size for evaluation (default: 8)
- `--bound_samples`: Number of samples to evaluate (default: 1000)
- `--levels`: Quantization levels (default: 11, must be odd)
- `--use_kmeans`: Use k-means for quantization (default: random)
- `--misc_extra_bits`: Extra bits for hyperparameter search (default: 5)

### Output
- `--output_dir`: Directory to save results (required)

## Output Files

The script creates two files in the output directory:

### `bounds.yml`
Contains the final PAC-Bayes bounds:
```yaml
prefix_message_len: 12345.67        # Bits to encode model
divergence: 85321.45                # KL divergence term
sample_size: 1000                   # Number of samples evaluated
data_size: 2000                     # Estimated dataset size
best_bpd_bound: 3.65               # Tightest bound across all α
best_alpha: 0.01                    # Best smoothing parameter
train_bpd_alpha_0.01: 2.34         # Empirical BPD (α=0.01)
bound_bpd_alpha_0.01: 3.45         # Bound on test BPD (α=0.01)
# ... bounds for other α values ...
```

### `metrics.yml`
Contains intermediate metrics:
```yaml
bpd_alpha_0.0001: 2.456
bpd_alpha_0.001: 2.445
bpd_alpha_0.01: 2.340
n_train: 1000
total_tokens: 50000
```

## Understanding the Results

### Message Length (Compression)
- **Lower is better**: Indicates better compression
- Represents bits needed to encode trainable parameters
- LoRA and intrinsic dimension provide natural compression

### BPD (Bits-Per-Dimension)
- Measures prediction quality
- Lower BPD = better predictions
- `train_bpd`: Empirical performance on training data
- `bound_bpd`: Upper bound on test performance with 95% confidence

### The Bound
The PAC-Bayes bound states:
```
test_BPD ≤ bound_BPD (with 95% probability)
```

Where:
```
bound = train_error + complexity_term
complexity_term = f(message_length, sample_size, dataset_size)
```

**Better compression → Smaller message length → Tighter bounds**

### Smoothing Parameter α
- Interpolates between empirical distribution (α=0) and uniform (α=1)
- Different α values give different bounds
- Best α is automatically selected (tightest bound)

## Example Workflow

### 1. Train a model with LoRA
```bash
python train_clrs_text.py \
    --task_names dfs \
    --train_lengths 10 \
    --model_key gpt2 \
    --train_lora \
    --lora_rank 4 \
    --epochs 20 \
    --save_name dfs_lora_r4
```

This saves checkpoint to: `external_lightning_logs/gpt2_dfs_lora_r_4_run_0/epoch_*.pt`

### 2. Compute bounds
```bash
python compute_clrs_bounds.py \
    --model_key gpt2 \
    --checkpoint_path external_lightning_logs/gpt2_dfs_lora_r_4_run_0/epoch_19.pt \
    --task_names dfs \
    --train_lengths 10 \
    --train_lora \
    --lora_rank 4 \
    --bound_samples 1000 \
    --output_dir ./bounds_results/dfs_lora_r4
```

### 3. View results
```bash
cat bounds_results/dfs_lora_r4/bounds.yml
```

## Tips

### Increasing Bound Samples
More samples = more accurate empirical BPD:
```bash
--bound_samples 5000  # Better estimate, but slower
```

### Quantization Levels
More levels = better approximation, but larger message length:
```bash
--levels 21  # More levels (must be odd)
```

### K-means vs Random Quantization
K-means may give better compression:
```bash
--use_kmeans  # Use k-means clustering
```

### Matching Training Configuration
**Critical**: Ensure bound computation arguments match training:
- If trained with `--train_lora --lora_rank 4`, use same for bounds
- If trained with `--intrinsic_dim 5000`, use same for bounds
- If trained with `--only_answer_output`, use same for bounds

## Troubleshooting

### "Checkpoint not found"
Ensure the checkpoint path is correct and the file exists:
```bash
ls external_lightning_logs/gpt2_*/epoch_*.pt
```

### "CUDA out of memory"
Reduce batch size:
```bash
--eval_batch_size 4
```

### "Model architecture mismatch"
Ensure model configuration matches training. Check:
- LoRA settings (`--train_lora`, `--lora_rank`)
- Intrinsic dimension (`--intrinsic_dim`)
- Model key (`--model_key`)

## Comparison with SubLoRA

### Similarities
- Uses same quantization and PAC-Bayes bound functions
- Computes message length for compression
- Evaluates BPD with α-smoothing

### Differences
- **Data format**: CLRS uses HuggingFace datasets vs binary files
- **Bound type**: Uses sequence-level sampling (not document-level)
- **Dataset size**: Smaller CLRS datasets vs large language corpora
- **Model types**: Supports various LLMs with LoRA/adapters

## Citation

If you use this bound computation in your research, please cite:
- The original SubLoRA paper for the bound computation framework
- The CLRS dataset paper
