# Training a Linear Self-Attention Model for Weight Prediction
    
Training a one-layer linear self-attention model for predicting weights of linear functions from in-context examples.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Experimental Setup

The task is to learn to predict the weights `w*` of a linear function from example input-output pairs:
- Given examples: `(x_1, y_1), ..., (x_n, y_n)` where `y_i = w* · x_i`
- Goal: Predict `w*`

### Model Architecture

- **Input**: Sequence of `(x, y)` pairs concatenated
- **Embedding**: Linear projection to `d_model` dimensions
- **Linear Attention**: One layer without softmax normalization
  - `Attention(Q, K, V) = Q(K^T V)`
- **Output**: Linear projection to predict weight vector

## Usage

### Basic Training

```bash
python train.py --input_dim 20 --n_examples 50 --epochs 100
```

### With Custom Parameters

```bash
python train.py \
  --input_dim 20 \
  --n_examples 50 \
  --d_model 128 \
  --d_key 64 \
  --d_value 64 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-3 \
  --use_scheduler \
  --save_dir checkpoints
```

### With Wandb Logging

```bash
python train.py \
  --use_wandb \
  --wandb_project linear-attention-weight-prediction \
  --wandb_run_name experiment-1
```

## Key Parameters

- `--input_dim`: Dimension of the weight vector (default: 20)
- `--n_examples`: Number of in-context examples per task (default: 50)
- `--n_train_tasks`: Number of training tasks (default: 10000)
- `--d_model`: Model embedding dimension (default: 128)
- `--d_key`, `--d_value`: Attention key/value dimensions (default: 64)
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--noise_std`: Standard deviation of output noise (default: 0.0)

## Results

The model is evaluated using mean squared error (MSE) between predicted and true weights.
Best model is saved based on validation loss.
