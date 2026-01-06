"""
Training script for one-layer linear attention model on weight prediction task.
Following the experimental setup from the paper for linear function learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import wandb
import os


class LinearAttentionLayer(nn.Module):
    """
    One-layer linear attention model without softmax normalization.
    Implements: Attn(Q, K, V) = Q(K^T V) where Q, K, V are linear projections.
    """
    def __init__(self, d_model, d_key, d_value):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_key, bias=False)
        self.W_k = nn.Linear(d_model, d_key, bias=False)
        self.W_v = nn.Linear(d_model, d_value, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_value, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Compute Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_key)
        K = self.W_k(x)  # (batch_size, seq_len, d_key)
        V = self.W_v(x)  # (batch_size, seq_len, d_value)
        
        # Linear attention: Q @ (K^T @ V)
        # K^T @ V: (batch_size, d_key, d_value)
        KV = torch.matmul(K.transpose(-2, -1), V)
        
        # Q @ (K^T @ V): (batch_size, seq_len, d_value)
        attention_output = torch.matmul(Q, KV)
        
        # Output projection
        output = self.W_o(attention_output)
        
        return output
    
class LinearAttentionLayerSimplified(nn.Module):
    """
    One-layer linear attention model without softmax normalization.
    Implements: Attn(Q, K, V) = Q(K^T V) where Q, K, V are linear projections.
    """
    def __init__(self, d_model, n):
        super().__init__()
        self.d_model = d_model
        self.n = n # normalization factor
        
        # Linear projections for merging Q and K
        self.W_kq = nn.Linear(d_model, d_model, bias=False)
        # Linear projection for merging V and projecting to output
        self.W_pv = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute W^{KQ}(X): (batch_size, seq_len, d_model)
        W_kq_x = self.W_kq(x)  # (batch_size, seq_len, d_model)
        
        # Compute attention score matrix: X^T @ W^{KQ}(X)
        # (batch_size, d_model, seq_len) @ (batch_size, seq_len, d_model) -> (batch_size, d_model, d_model)
        x_T = x.transpose(-2, -1)  # (batch_size, d_model, seq_len)
        
        # But wait - we need the score matrix to be (seq_len, seq_len) for masking
        # The correct formulation: W^{KQ}(X) @ X^T gives us (seq_len, seq_len) scores
        attention_scores = torch.matmul(W_kq_x, x_T) / self.n  # (batch_size, seq_len, seq_len)
        
        # Apply causal mask: only attend to past positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))  # (seq_len, seq_len)
        masked_scores = attention_scores * causal_mask  # (batch_size, seq_len, seq_len)
        
        # Apply attention to values: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, d_model)
        PV = self.W_pv(x)  # (batch_size, seq_len, d_model)
        attention_output = torch.matmul(masked_scores, PV)  # (batch_size, seq_len, d_model)
        
        # Add residual connection
        output = attention_output + x
        
        return output


class WeightPredictionModel(nn.Module):
    """
    Model for predicting linear function weights from in-context examples.
    Causal language modeling style: predict next position from all preceding positions.
    """
    def __init__(self, d_model, n_examples):
        super().__init__()
        self.d_model = d_model
        self.n_examples = n_examples
                
        # Linear attention layer
        self.attention = LinearAttentionLayerSimplified(d_model, n_examples)
        
        
    def forward(self, Z):
        """
        Args:
            Z: (batch_size, seq_len, d_model) - input sequence
        Returns:
            Z_pred: (batch_size, seq_len, d_model) - predictions for all positions
        """
        
        # Apply linear attention with causal masking
        Z_hat = self.attention(Z)  # (batch_size, seq_len, d_model)
        
        return Z_hat
    
    def compute_loss(self, Z, label_masks=None, n_examples=None):
        """
        Compute causal language modeling loss.
        At each position t, predict Z[:, :, t+1] from Z[:, :, :t+1]
        
        Args:
            Z: (batch_size, seq_len, d_model) - input sequence
            label_masks: (batch_size, seq_len-1) - optional mask for loss computation
            n_examples: int - if provided, masks out loss on first n_examples positions
        Returns:
            loss: scalar - average MSE loss over all positions
        """
        batch_size, seq_len, d_model = Z.shape
        
        # Get predictions for all positions
        Z_pred = self.forward(Z)  # (batch_size, seq_len, d_model)
        
        # Compute loss: predict position t from positions 0..t-1
        # So we compare Z_pred[:, :-1, :] with Z[:, 1:, :]
        # This gives us predictions for positions 1 to seq_len-1
        if seq_len > 1:
            predictions = Z_pred[:, :-1, :]  # (batch_size, seq_len-1, d_model)
            targets = Z[:, 1:, :]  # (batch_size, seq_len-1, d_model)
            
            # Create default label mask if n_examples is provided
            if label_masks is None and n_examples is not None:
                # Mask out first n_examples positions (input examples)
                # Only compute loss on CoT and final answer positions
                label_masks = torch.zeros(batch_size, seq_len - 1, device=Z.device)
                label_masks[:, n_examples:] = 1.0  # Enable loss after n_examples positions
            
            if label_masks is not None:
                # Apply label masks
                masked_loss = (predictions - targets) ** 2 * label_masks.unsqueeze(-1)  # (batch_size, seq_len-1, d_model)
                loss = masked_loss.sum() / (label_masks.sum() * d_model + 1e-8)  # Avoid division by zero
            else:
                # Compute MSE loss over all positions
                loss = torch.mean((predictions - targets) ** 2)
        else:
            # If sequence length is 1, no next token to predict
            loss = torch.tensor(0.0, device=Z.device)
        
        return loss
    
    def generate(self, Z_init, T):
        """
        Generate T steps of chain-of-thought predictions autoregressively.
        
        Args:
            Z_init: (batch_size, seq_len_init, d_model) - initial sequence containing:
                    - n_examples columns of [x_i, y_i, 0, 0]
                    - 1 column of [0, 0, w_0, 1] (initial weight guess)
            T: int - number of generation steps
        Returns:
            Z_generated: (batch_size, seq_len_init + T, d_model) - full sequence with T generated steps
        """
        batch_size = Z_init.size(0)
        device = Z_init.device
        
        # Start with initial sequence
        Z_current = Z_init  # (batch_size, seq_len_init, d_model)
        
        # Generate T steps autoregressively
        for step in range(T):
            # Forward pass on current sequence
            Z_pred = self.forward(Z_current)  # (batch_size, seq_len_current, d_model)
            
            # Take the prediction at the last position (predicts next token)
            next_token = Z_pred[:, -1:, :]  # (batch_size, 1, d_model)
            
            # Append to sequence
            Z_current = torch.cat([Z_current, next_token], dim=1)  # (batch_size, seq_len_current + 1, d_model)
        
        return Z_current


class LinearFunctionDataset(Dataset):
    """
    Dataset for linear function weight prediction task.
    Generates random linear functions w* and examples (x, y) where y = w* · x.
    """
    def __init__(self, n_tasks, n_examples, input_dim, noise_std=0.0, 
                lr=0.4, T=20, split='train'):
        self.n_tasks = n_tasks
        self.task_seeds = np.arange(n_tasks) if split == 'train' else np.arange(10000000, 10000000 + n_tasks)
        self.n_examples = n_examples
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.lr = lr
        self.T = T
        
    def __len__(self):
        return self.n_tasks
    
    def __getitem__(self, idx):
        # Sample random weight vector w* from standard Gaussian
        rng = torch.Generator()
        rng.manual_seed(int(self.task_seeds[idx]))
        w_star = torch.randn(self.input_dim, generator=rng)
        
        # Sample n_examples random input vectors from standard Gaussian
        x = torch.randn(self.n_examples, self.input_dim, generator=rng) # (n_examples, input_dim)
        
        # Compute y = w* · x + noise
        y = torch.matmul(x, w_star).unsqueeze(-1)  # (n_examples, 1)
        
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
            
        # Generate the gradient descent on x, y for T steps as the chain-of-thought
        w_0 = torch.zeros_like(w_star)
        w_t = w_0.clone()
        
        cot = [w_0.unsqueeze(0)]  # list of (1, input_dim)
        for t in range(self.T):
            grad = x.T @ (torch.matmul(x, w_t.unsqueeze(-1)) - y).squeeze(-1) / self.n_examples
            w_t = w_t - self.lr * grad
            cot.append(w_t.unsqueeze(0))
        cot = torch.cat(cot, dim=0)  # (T+1, input_dim)
        
        # format the input sequences
        # Z structure: [x, y, w_cot, w_star] with indicator rows
        # Row structure: [x rows (input_dim), y row (1), w rows (input_dim), indicator row (1)]
        # Total d_model = 2*input_dim + 2
        
        # Z_input: n_examples columns of [x_i, y_i, 0, 0]
        Z_input = torch.cat([x.T, y.T, torch.zeros(self.input_dim + 1, self.n_examples)], dim=0)  # (2*input_dim + 2, n_examples)
        
        # Z_cot: T+1 columns of [0, 0, w_t, 1] for chain-of-thought steps
        Z_cot = torch.cat([torch.zeros(self.input_dim + 1, self.T + 1), cot.T, torch.ones(1, self.T + 1)], dim=0)  # (2*input_dim + 2, T+1)
        
        # Z_star: 1 column of [0, 0, w_star, 1] for final answer
        Z_star = torch.cat([torch.zeros(self.input_dim + 1, 1), w_star.unsqueeze(-1), torch.ones(1, 1)], dim=0)  # (2*input_dim + 2, 1)
        
        # Concatenate all parts
        Z_final = torch.cat([Z_input, Z_cot, Z_star], dim=1).T  # (n_examples + T + 2, 2*input_dim + 2)
        
        return Z_final


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataset by generating T+1 steps and comparing final prediction to w_star.
    Returns the mean squared error between predicted and true weights.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for Z in dataloader:
            Z = Z.to(device)  # (batch_size, seq_len, d_model)
            batch_size = Z.size(0)
            
            # Extract initial sequence: input examples + w_0
            # Z contains: n_examples positions of input + T+1 positions of CoT + 1 position of w_star
            n_examples = model.n_examples
            Z_init = Z[:, :n_examples+1, :]  # (batch_size, n_examples+1, d_model)
            
            # Extract true w_star from the last position
            # w_star is in rows input_dim+1 to 2*input_dim
            input_dim = (model.d_model - 2) // 2
            w_star_true = Z[:, -1, input_dim+1:2*input_dim+1]  # (batch_size, input_dim)
            
            # Generate T+1 steps
            T = Z.size(1) - n_examples - 2  # Total positions - n_examples - w_0 - w_star
            Z_generated = model.generate(Z_init, T + 1)  # (batch_size, n_examples+1+T+1, d_model)
            
            # Extract predicted w_star from the last generated position
            w_star_pred = Z_generated[:, -1, input_dim+1:2*input_dim+1]  # (batch_size, input_dim)
            
            # Check for NaN values
            if torch.isnan(w_star_pred).any() or torch.isnan(w_star_true).any():
                print(f"Warning: NaN detected in predictions or targets")
                print(f"NaN in pred: {torch.isnan(w_star_pred).any()}, NaN in true: {torch.isnan(w_star_true).any()}")
                continue
            
            # Compute MSE between predicted and true w_star
            loss = torch.mean((w_star_pred - w_star_true) ** 2)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    model.train()
    return total_loss / total_samples if total_samples > 0 else float('nan')


def train(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if torch.cuda.is_available() and args.device:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Create datasets
    train_dataset = LinearFunctionDataset(
        n_tasks=args.n_train_tasks,
        n_examples=args.n_examples,
        input_dim=args.input_dim,
        noise_std=args.noise_std,
        lr=args.gd_lr,
        T=args.T,
        split='train'
    )
    
    test_dataset = LinearFunctionDataset(
        n_tasks=args.n_test_tasks,
        n_examples=args.n_examples,
        input_dim=args.input_dim,
        noise_std=args.noise_std,
        lr=args.gd_lr,
        T=args.T,
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    # d_model is determined by the dataset structure: 2*input_dim + 2
    d_model = 2 * args.input_dim + 2
    model = WeightPredictionModel(
        d_model=d_model,
        n_examples=args.n_examples
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min
        )
    
    # Training loop
    best_val_loss = float('inf')

    train_loss = 0.0
    train_samples = 0
    steps = 0
    for epoch in range(args.epochs):
        model.train()
        
        for Z in train_loader:
            Z = Z.to(device)  # (batch_size, d_model, seq_len)
            
            # Compute loss over all positions (CLM style), masking out input examples
            loss = model.compute_loss(Z, n_examples=model.n_examples)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Track loss
            train_loss += loss.item() * Z.size(0)
            train_samples += Z.size(0)
            steps += 1
        
            if steps % args.eval_interval == 0:
                avg_train_loss = train_loss / train_samples
                
                # Validation
                val_loss = evaluate(model, test_loader, device)
                
                print(f"Steps {steps} - Train Loss: {avg_train_loss:.6f}, Test Loss: {val_loss:.6f}")
                
                # Log to wandb
                if args.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'lr': optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.save_dir:
                        os.makedirs(args.save_dir, exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'args': vars(args)
                        }, os.path.join(args.save_dir, 'best_model.pt'))
                        print(f"Saved best model with val loss: {val_loss:.6f}")
                        
                # Reset training loss counters
                train_loss = 0.0
                train_samples = 0
                
            # Update learning rate
            if args.use_scheduler:
                scheduler.step()
                
    
    # Final test evaluation
    test_loss = evaluate(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.6f}")
    
    if args.use_wandb:
        wandb.log({'test_loss': test_loss})
        wandb.finish()
    
    return model, test_loss


def main():
    parser = argparse.ArgumentParser(description='Train linear attention model on weight prediction')
    
    # Dataset parameters
    parser.add_argument('--input_dim', type=int, default=20,
                        help='Dimension of input vectors')
    parser.add_argument('--n_examples', type=int, default=50,
                        help='Number of in-context examples per task')
    parser.add_argument('--n_train_tasks', type=int, default=100000000,
                        help='Number of training tasks')
    parser.add_argument('--n_test_tasks', type=int, default=1000,
                        help='Number of test tasks')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='Standard deviation of output noise')
    parser.add_argument('--gd_lr', type=float, default=0.4,
                        help='Learning rate for gradient descent in CoT generation')
    parser.add_argument('--T', type=int, default=20,
                        help='Number of gradient descent steps in CoT')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use cosine annealing LR scheduler')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluation interval (in steps)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='linear-attention-weight-prediction',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    
    args = parser.parse_args()
    
    # Train the model
    train(args)


if __name__ == '__main__':
    main()
