# Computing Bounds for SubLoRA Models

This guide explains how to compute generalization bounds for trained SubLoRA models using the document-level bound evaluation.

## Overview

The bound computation requires two preprocessing files that identify document boundaries in the training data:
1. **EOT indices file**: Array of positions where End-Of-Text tokens appear (document separators)
2. **Document lengths file**: Array of lengths for each document in the dataset

## Quick Start

### Option 1: Using the Automated Script (Recommended)

Run the all-in-one script that handles preprocessing and bound evaluation:

```bash
cd compression-bound-for-llm
bash scripts/run_bound_evaluation.sh
```

**Edit the script first** to set your paths:
- `DATA_DIR`: Path to your data directory (e.g., `./data/openwebtext`)
- `CHECKPOINT_PATH`: Path to your trained model checkpoint
- `EOT_TOKEN`: Token ID for end-of-text (default: 50256 for GPT-2)

### Option 2: Manual Step-by-Step

#### Step 1: Generate Document Indices Files (One-time preprocessing)

```bash
cd compression-bound-for-llm

python scripts/generate_document_indices.py \
    --data_dir ./data/openwebtext \
    --eot_token 50256 \
    --output_dir ./data/openwebtext
```

**Parameters:**
- `--data_dir`: Directory containing `train.bin`
- `--eot_token`: End-of-text token ID (50256 for GPT-2, 50257 for other tokenizers)
- `--output_dir`: Where to save the output files (defaults to data_dir)

**Output files:**
- `openwebtext_train_eot_indices_file.npy`: EOT token positions
- `empirical_document_length_distribution_file.npy`: Document lengths

**Note:** This step scans the entire training dataset and may take a few minutes. The files are saved for reuse.

#### Step 2: Run Bound Evaluation

```bash
python experiments/eval_bounds.py \
    --config-file=config/sublora_bounds.yaml \
    --data.dataset_dir=./data \
    --model.best_checkpoint_path=./checkpoints/compression_bound/id50000_lr0.005_r4/2025-12-05/20-18 \
    --bounds.bound_type=document_level \
    --data.openwebtext_train_eot_indices_file=./data/openwebtext/openwebtext_train_eot_indices_file.npy \
    --data.empirical_document_length_distribution_file=./data/openwebtext/empirical_document_length_distribution_file.npy
```

## Understanding the Parameters

### Required Path Parameters

```bash
--data.dataset_dir=./data
```
Base directory containing the dataset folder (e.g., `openwebtext/`)

```bash
--model.best_checkpoint_path=./checkpoints/.../...
```
Path to your trained model checkpoint directory

```bash
--data.openwebtext_train_eot_indices_file=./data/openwebtext/openwebtext_train_eot_indices_file.npy
```
Full path to the EOT indices file (generated in Step 1)

```bash
--data.empirical_document_length_distribution_file=./data/openwebtext/empirical_document_length_distribution_file.npy
```
Full path to the document lengths file (generated in Step 1)

### Bound Type

```bash
--bounds.bound_type=document_level
```
- `document_level`: Sample entire documents (can be > 1024 tokens), uses sliding window
- `sequence_level`: Sample fixed-length sequences (exactly 1024 tokens)

For theoretical guarantees matching the paper, use `document_level`.

## What Happens During Bound Computation

### Phase 1: Model Quantization
1. Loads your trained model from checkpoint
2. Quantizes trainable parameters to `levels` discrete values (default: 11)
3. Computes the **message length** (bits needed to encode the model)
4. Optional: Runs quantization-aware training if `max_quant_iters > 0`

### Phase 2: Statistics Collection
For `bound_samples` documents (default: 10,000):
1. Samples a random document from the training set
2. Runs forward pass to get predictions
3. Records:
   - **Top-k accuracy**: How often the true token is in top-k predictions
   - **BPD (Bits Per Dimension)**: Prediction quality for various α values
4. Saves intermediate results continuously

### Phase 3: Bound Calculation
1. Computes **divergence term** from message length (compression)
2. Applies **PAC-Bayesian subsampling bound** formula
3. Outputs bounds for:
   - Top-1, Top-2, ..., Top-10, Top-50, Top-100 accuracy
   - BPD for α ∈ {0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5}
4. Reports **best_bpd_bound**: tightest generalization guarantee

## Output Files

All outputs are saved in your checkpoint directory with filenames based on quantization settings:

```
your_checkpoint_dir/
├── quant_ckpt_levels11_iters0.pt          # Quantized model
├── bounds_levels11_iters0.yml             # Final bounds (THIS IS YOUR MAIN RESULT)
├── metrics_levels11_iters0.yml            # Running statistics
├── ix_levels11_iters0.txt                 # Sampled document indices
├── top_k_indices_levels11_iters0.txt      # Prediction ranks
├── selected_prob_scores_levels11_iters0.txt  # Predicted probabilities
└── percentile_vec_levels11_iters0.txt     # Percentile scores
```

### Key Results File: `bounds_levels11_iters0.yml`

```yaml
prefix_message_len: 12345.67        # Bits to encode model
acc_divergence: 85321.45            # KL divergence for accuracy bounds
bpd_divergence: 85324.89            # KL divergence for BPD bounds
bound_top_1_acc: 0.524              # Upper bound on test top-1 error
bound_top_5_acc: 0.231              # Upper bound on test top-5 error
bound_bpd_alpha_0.01: 3.78          # Bound on test BPD (α=0.01)
best_bpd_bound: 3.65                # Tightest BPD bound across all α
```

## Configuration Options

Edit `config/sublora_bounds.yaml` to adjust:

```yaml
bounds:
  levels: 11                  # Number of quantization levels (must be odd)
  max_quant_iters: 0          # Quantization-aware training iterations (0=none)
  bound_samples: 10000        # Number of documents to sample
  sliding_window_size: 100    # Sliding window for long documents
  misc_extra_bits: 7          # Bits for hyperparameter search overhead
```

## Interpreting Results

### Message Length (Compression)
- **Lower is better**: Indicates the model compresses well
- Measured in bits needed to encode all trainable parameters
- LoRA naturally provides compression via low-rank structure

### Bounds
- Values are **upper bounds** on test error with high probability (95%)
- **Smaller bounds** = better generalization guarantees
- Bound formula: `test_error ≤ train_error + complexity_term`
- Complexity term depends on message length (compression)

### Best BPD Bound
- Primary metric for language model generalization
- Measures bits-per-dimension on test data
- Compared to empirical test BPD to validate tightness

## Troubleshooting

### "File not found: train.bin"
Ensure you have preprocessed your data. For OpenWebText, run the data preparation script first.

### "CUDA out of memory"
Reduce `bounds.eval_batch_size` in the config file (default: 6).

### "EOT indices file not found"
Run Step 1 (generate_document_indices.py) first to create the required files.

### Very slow execution
- Sampling 10,000 documents takes time (several hours on GPU)
- Results are saved incrementally in `metrics_*.yml`
- You can stop and resume, though it will restart from beginning

## Example: Full Workflow

```bash
# 1. Navigate to project
cd compression-bound-for-llm

# 2. Generate document indices (one time only)
python scripts/generate_document_indices.py \
    --data_dir ./data/openwebtext \
    --eot_token 50256

# 3. Run bound evaluation
python experiments/eval_bounds.py \
    --config-file=config/sublora_bounds.yaml \
    --data.dataset_dir=./data \
    --model.best_checkpoint_path=./checkpoints/compression_bound/id50000_lr0.005_r4/2025-12-05/20-18 \
    --bounds.bound_type=document_level \
    --data.openwebtext_train_eot_indices_file=./data/openwebtext/openwebtext_train_eot_indices_file.npy \
    --data.empirical_document_length_distribution_file=./data/openwebtext/empirical_document_length_distribution_file.npy

# 4. Check results
cat ./checkpoints/compression_bound/id50000_lr0.005_r4/2025-12-05/20-18/bounds_levels11_iters0.yml
```

## Citation

If you use this bound computation in your research, please cite the SubLoRA paper.
