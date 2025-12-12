#!/bin/bash
# Example script to compute bounds for a trained SubLoRA model
# This demonstrates the complete workflow from generating required files to running bound evaluation

set -e  # Exit on error

# ============================================================
# STEP 1: Set your paths
# ============================================================
DATA_DIR="./data/openwebtext"
CHECKPOINT_PATH="./checkpoints/compression_bound/id50000_lr0.005_r4/2025-12-05/20-18"
EOT_TOKEN=50256  # GPT-2's <|endoftext|> token

echo "============================================================"
echo "SubLoRA Bound Computation Workflow"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Checkpoint path: $CHECKPOINT_PATH"
echo "  EOT token: $EOT_TOKEN"
echo ""

# ============================================================
# STEP 2: Generate document indices (first time only)
# ============================================================
EOT_INDICES_FILE="${DATA_DIR}/openwebtext_train_eot_indices_file.npy"
DOC_LENGTHS_FILE="${DATA_DIR}/empirical_document_length_distribution_file.npy"

# if [ ! -f "$EOT_INDICES_FILE" ] || [ ! -f "$DOC_LENGTHS_FILE" ]; then
#     echo "============================================================"
#     echo "STEP 1: Generating document indices files..."
#     echo "============================================================"
#     echo "This is a one-time preprocessing step that scans the entire"
#     echo "training data to find document boundaries."
#     echo ""
    
#     python scripts/generate_document_indices.py \
#         --data_dir "$DATA_DIR" \
#         --eot_token $EOT_TOKEN \
#         --output_dir "$DATA_DIR"
    
#     echo ""
#     echo "✓ Document indices files created successfully!"
#     echo ""
# else
#     echo "============================================================"
#     echo "STEP 1: Document indices files already exist (skipping)"
#     echo "============================================================"
#     echo "  Found: $EOT_INDICES_FILE"
#     echo "  Found: $DOC_LENGTHS_FILE"
#     echo ""
# fi

# ============================================================
# STEP 3: Run bound evaluation
# ============================================================
echo "============================================================"
echo "STEP 2: Running bound evaluation..."
echo "============================================================"
echo "This will:"
echo "  1. Load the trained model from checkpoint"
echo "  2. Quantize the model parameters"
echo "  3. Sample documents and compute prediction statistics"
echo "  4. Calculate PAC-Bayesian generalization bounds"
echo ""

python experiments/eval_bounds.py \
    --config-file=config/sublora_bounds.yaml \
    --data.dataset_dir="./data" \
    --model.best_checkpoint_path="$CHECKPOINT_PATH" \
    --bounds.bound_type=document_level \
    --data.openwebtext_train_eot_indices_file="$EOT_INDICES_FILE" \
    --data.empirical_document_length_distribution_file="$DOC_LENGTHS_FILE"

echo ""
echo "============================================================"
echo "✓ Bound computation completed!"
echo "============================================================"
echo ""
echo "Results are saved in: $CHECKPOINT_PATH"
echo ""
echo "Output files:"
echo "  - quant_ckpt_levels*.pt: Quantized model checkpoint"
echo "  - bounds_levels*.yml: Final bound values"
echo "  - metrics_levels*.yml: Prediction statistics"
echo "  - ix_levels*.txt: Sampled indices"
echo "  - top_k_indices_levels*.txt: Top-k prediction ranks"
echo "  - selected_prob_scores_levels*.txt: Predicted probabilities"
echo "  - percentile_vec_levels*.txt: Percentile scores"
echo ""
