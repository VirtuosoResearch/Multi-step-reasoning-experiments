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
    def __init__(self, d_model, n, input_dim, no_cot=False, use_noise_injection=False, noise_sigma=1e-3):
        super().__init__()
        self.d_model = d_model
        self.n = n # normalization factor
        self.input_dim = input_dim
        self.use_noise_injection = use_noise_injection
        self.noise_sigma = noise_sigma
        
        # Linear projections for merging Q and K
        self.W_kq = nn.Linear(d_model, d_model, bias=False)
        # Linear projection for merging V and projecting to output
        self.W_pv = nn.Linear(d_model, d_model, bias=False)
        
        # Block initialization according to the paper
        # d_model = 2*input_dim + 2
        # Block structure: [x (input_dim), y (1), w (input_dim), indicator (1)]
        # Blocks: (0: x), (1: y), (2: w), (3: indicator)
        
        # Initialize all weights with small random values to break symmetry
        # Use smaller std for non-important blocks
        nn.init.zeros_(self.W_kq.weight)
        nn.init.zeros_(self.W_pv.weight)
        
        # W^{KQ}: emphasize (1,3) block - from indicator to y
        # This maps: indicator (col) -> y (row)
        # Row indices for y: input_dim to input_dim+1
        # Col indices for indicator: 2*input_dim+1 to 2*input_dim+2
        block_13_kq = self.W_kq.weight[:input_dim, input_dim+1:2*input_dim+1]
        # nn.init.normal_(block_13_kq, mean=0.0, std=1.0 / d_model)
        # Initialize as diagonal matrix with random values from N(0, 1/d_model)
        with torch.no_grad():
            diagonal_values = torch.abs(torch.randn(input_dim) * (1.0 / d_model)) + 0.1
            block_13_kq.copy_(torch.diag(diagonal_values))
        block_24_kq = self.W_kq.weight[input_dim:input_dim+1, 2*input_dim+1:2*input_dim+2]
        nn.init.constant_(block_24_kq, -1.0)
        
        # W^{PV}: emphasize (3,1) block - from y to indicator
        # This maps: y (col) -> indicator (row)
        # Row indices for indicator: 2*input_dim+1 to 2*input_dim+2
        # Col indices for y: input_dim to input_dim+1
        block_31_pv = self.W_pv.weight[input_dim+1:2*input_dim+1, 0:input_dim]
        # nn.init.normal_(block_31_pv, mean=0.0, std=1.0 / d_model)
        # Initialize as diagonal matrix with random values from N(0, 1/d_model)
        with torch.no_grad():
            diagonal_values = -torch.abs(torch.randn(input_dim) * (1.0 / d_model))
            block_31_pv.copy_(torch.diag(diagonal_values))
        
        self.no_cot = no_cot
        if not self.no_cot:
            # Register hook to zero out gradients for block_24_kq
            def zero_block_24_grad(grad):
                grad_copy = grad.clone()
                grad_copy[input_dim:input_dim+1, 2*input_dim+1:2*input_dim+2] = 0
                return grad_copy
            
            self.W_kq.weight.register_hook(zero_block_24_grad)
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Inject noise into weights if enabled (only during training)
        if self.use_noise_injection and self.training:
            # Generate noise without gradients
            with torch.no_grad():
                noise_kq = torch.randn_like(self.W_kq.weight) * self.noise_sigma * (self.W_kq.weight != 0)
                noise_pv = torch.randn_like(self.W_pv.weight) * self.noise_sigma * (self.W_pv.weight != 0)
                # set 24 block noise to zero
                noise_kq[self.input_dim:self.input_dim+1, 2*self.input_dim+1:2*self.input_dim+2] = 0.0
            
            # Compute with noisy weights (noise doesn't require grad)
            W_kq_noisy = self.W_kq.weight + noise_kq
            W_pv_noisy = self.W_pv.weight + noise_pv
            
            # Use functional API to compute with noisy weights
            W_kq_x = torch.nn.functional.linear(x, W_kq_noisy)  # (batch_size, seq_len, d_model)
        else:
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
        if self.use_noise_injection and self.training:
            PV = torch.nn.functional.linear(x, W_pv_noisy)  # (batch_size, seq_len, d_model)
        else:
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
    def __init__(self, d_model, n_examples, no_cot=False, use_noise_injection=False, noise_sigma=1e-3):
        super().__init__()
        self.d_model = d_model
        self.n_examples = n_examples
        self.no_cot = no_cot
        self.use_noise_injection = use_noise_injection
        self.noise_sigma = noise_sigma
        
        # Calculate input_dim from d_model
        # d_model = 2*input_dim + 2
        input_dim = (d_model - 2) // 2
                
        # Linear attention layer
        self.attention = LinearAttentionLayerSimplified(d_model, n_examples, input_dim, no_cot, 
                                                        use_noise_injection, noise_sigma)
        
        
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
            
            # if self.no_cot:
            #     # compute loss on the last position only
            #     final_pred = predictions[:, -1, :]  # (batch_size, d_model)
            #     final_target = targets[:, -1, :]    # (batch_size, d_model)
            #     loss = torch.mean(((final_pred - final_target) ** 2).sum(dim=1))
            #     return loss
            
            # Create default label mask if n_examples is provided
            if label_masks is None and n_examples is not None:
                # Mask out first n_examples positions (input examples)
                # Only compute loss on CoT and final answer positions
                label_masks = torch.zeros(batch_size, seq_len - 1, device=Z.device)
                label_masks[:, n_examples:] = 1.0  # Enable loss after n_examples positions
            
            if label_masks is not None:
                # Apply label masks and compute mean per batch example
                squared_error = (predictions - targets) ** 2  # (batch_size, seq_len-1, d_model)
                masked_squared_error = squared_error * label_masks.unsqueeze(-1)  # (batch_size, seq_len-1, d_model)
                
                # Sum over seq_len and d_model dimensions, then divide by number of active elements per example
                loss_per_example = masked_squared_error.sum(dim=[1, 2])  # (batch_size,)
                # num_active_elements = label_masks.sum(dim=1)  # (batch_size,)
                # loss_per_example = loss_per_example / (num_active_elements + 1e-8)  # Avoid division by zero
                
                # Mean over batch
                loss = loss_per_example.mean()
            else:
                # Compute MSE loss over all positions
                loss = torch.mean(((predictions - targets) ** 2).sum(dim=[2]))  
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
                lr=0.4, T=20, split='train', no_cot=False):
        self.n_tasks = n_tasks
        self.task_seeds = np.arange(n_tasks) if split == 'train' else (np.arange(10000000, 10000000 + n_tasks) if split == 'test' else np.arange(20000000, 20000000 + n_tasks))
        self.n_examples = n_examples
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.lr = lr
        self.T = T
        self.no_cot = no_cot
        
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
        
        if self.no_cot:
            cot = torch.zeros((self.T + 1, self.input_dim))  # (T+1, input_dim)
        else:
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

def evaluate_noise_stability(model, dataloader, device, sigma, runs=10):
    # copy the current model weights
    state_dict_original = model.state_dict()
    state_dict_original = {k: v.clone() for k, v in state_dict_original.items()}
    
    perturbed_loss = []
    for i in range(runs):
        # add Gaussian noise to model weights
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * sigma * (param.data != 0)
            param.data.add_(noise)
        
        # evaluate on the dataset
        loss = evaluate(model, dataloader, device)
        perturbed_loss.append(loss)
        
        # restore original weights
        model.load_state_dict(state_dict_original)
    return np.mean(perturbed_loss), np.std(perturbed_loss)

def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataset by generating T+1 steps and comparing final prediction to w_star.
    Returns the mean squared error between predicted and true weights.
    """
    model.eval(); model.training = False; model.attention.training = False
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
            
            # # Check for NaN values
            # if torch.isnan(w_star_pred).any() or torch.isnan(w_star_true).any():
            #     print(f"Warning: NaN detected in predictions or targets")
            #     print(f"NaN in pred: {torch.isnan(w_star_pred).any()}, NaN in true: {torch.isnan(w_star_true).any()}")
            #     continue
            
            loss = torch.mean(((w_star_pred - w_star_true) ** 2).sum(dim=1))  # mean over batch
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    model.train(); model.training = True; model.attention.training = True
    return total_loss / total_samples if total_samples > 0 else float('nan')


def generate_and_filter_cot(model, no_cot_dataset, device, error_threshold, batch_size=64,
                            inject_noise=False, noise_sigma=0.001):
    """
    Generate CoT sequences for samples without CoT and filter based on prediction error.
    
    Args:
        model: The trained model
        no_cot_dataset: Dataset without CoT sequences
        device: Device to run on
        error_threshold: Maximum MSE error to accept a generated CoT
        batch_size: Batch size for generation
        
    Returns:
        filtered_samples: List of (Z_with_cot, error) tuples that passed the threshold
    """
    model.eval()
    filtered_samples = []
    
    if inject_noise:
        state_dict_original = model.state_dict()
        state_dict_original = {k: v.clone() for k, v in state_dict_original.items()}
        # add Gaussian noise to model weights
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * noise_sigma * (param.data != 0)
            param.data.add_(noise)
    
    dataloader = DataLoader(no_cot_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, Z in enumerate(dataloader):
            Z = Z.to(device)  # (batch_size, seq_len, d_model)
            batch_size_actual = Z.size(0)
            
            # Extract initial sequence: input examples + w_0
            n_examples = model.n_examples
            Z_init = Z[:, :n_examples+1, :]  # (batch_size, n_examples+1, d_model)
            
            # Extract true w_star from the last position
            input_dim = (model.d_model - 2) // 2
            w_star_true = Z[:, -1, input_dim+1:2*input_dim+1]  # (batch_size, input_dim)
            
            # Generate T+1 steps
            T = Z.size(1) - n_examples - 2
            Z_generated = model.generate(Z_init, T + 1)  # (batch_size, n_examples+1+T+1, d_model)
            
            # Extract predicted w_star
            w_star_pred = Z_generated[:, -1, input_dim+1:2*input_dim+1]  # (batch_size, input_dim)
            
            # Compute error for each sample
            errors = ((w_star_pred - w_star_true) ** 2).sum(dim=1)  # (batch_size,)
            
            # Filter samples based on error threshold using tensor operations
            mask = errors < error_threshold  # (batch_size,)
            if mask.any():
                # Get indices where error is below threshold
                valid_indices = mask.nonzero(as_tuple=True)[0]  # (n_valid,)
                
                # Extract valid samples and errors in batch
                valid_Z = Z_generated[valid_indices].cpu()  # (n_valid, seq_len, d_model)
                valid_errors = errors[valid_indices].cpu()  # (n_valid,)
                
                # replace the last column as the true w_star
                valid_Z[:, -1, input_dim+1:2*input_dim+1] = w_star_true[valid_indices].cpu()
                # make positions as zeros in a similar format as the dataset
                valid_Z[:, :n_examples, input_dim+1:2*input_dim+2] = 0.0  # zero out w_cot and indicator in input examples
                valid_Z[:, n_examples:n_examples+T+1, :input_dim+1] = 0.0  # zero out x and y in CoT positions
                valid_Z[:, n_examples:n_examples+T+1, 2*input_dim+1:] = 1.0  # set indicator to 1 in CoT positions
                
                # Append to filtered samples
                for i in range(len(valid_indices)):
                    filtered_samples.append((valid_Z[i:i+1], valid_errors[i].item()))
    
    if inject_noise:
        # restore original weights
        model.load_state_dict(state_dict_original)
    model.train()
    print(f"Generated CoT for {len(no_cot_dataset)} samples, {len(filtered_samples)} passed threshold {error_threshold}")
    return filtered_samples


class MixedCoTDataset(Dataset):
    """
    Dataset that mixes samples with ground-truth CoT and generated CoT.
    """
    def __init__(self, cot_dataset, generated_samples):
        """
        Args:
            cot_dataset: Dataset with ground-truth CoT
            generated_samples: List of (Z_with_cot, error) tuples from generate_and_filter_cot
        """
        self.cot_dataset = cot_dataset
        self.generated_samples = [sample[0] for sample in generated_samples]  # Extract Z tensors
        self.total_len = len(cot_dataset) + len(self.generated_samples)
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.cot_dataset):
            return self.cot_dataset[idx]
        else:
            gen_idx = idx - len(self.cot_dataset)
            return self.generated_samples[gen_idx].squeeze(0)  # Remove batch dimension


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
    # Split training data into CoT and no-CoT portions
    if args.cot_ratio < 1.0:
        n_cot_tasks = int(args.n_train_tasks * args.cot_ratio)
        n_no_cot_tasks = args.n_train_tasks - n_cot_tasks
        
        print(f"Creating mixed dataset: {n_cot_tasks} with CoT, {n_no_cot_tasks} without CoT")
        
        # Dataset with CoT
        cot_dataset = LinearFunctionDataset(
            n_tasks=n_cot_tasks,
            n_examples=args.n_examples,
            input_dim=args.input_dim,
            noise_std=args.noise_std,
            lr=args.gd_lr,
            T=args.T,
            split='train',
            no_cot=False
        )
        
        # Dataset without CoT
        no_cot_dataset = LinearFunctionDataset(
            n_tasks=n_no_cot_tasks,
            n_examples=args.n_examples,
            input_dim=args.input_dim,
            noise_std=args.noise_std,
            lr=args.gd_lr,
            T=args.T,
            split='train_no_cot',
            no_cot=True
        )
        # Offset the seeds for no_cot_dataset to avoid overlap
        no_cot_dataset.task_seeds = np.arange(n_cot_tasks, n_cot_tasks + n_no_cot_tasks)
        
        # Initially train only on CoT dataset
        train_dataset = cot_dataset
    else:
        # All training data has CoT
        train_dataset = LinearFunctionDataset(
            n_tasks=args.n_train_tasks,
            n_examples=args.n_examples,
            input_dim=args.input_dim,
            noise_std=args.noise_std,
            lr=args.gd_lr,
            T=args.T,
            split='train',
            no_cot=args.no_cot
        )
        no_cot_dataset = None
    
    test_dataset = LinearFunctionDataset(
        n_tasks=args.n_test_tasks,
        n_examples=args.n_examples,
        input_dim=args.input_dim,
        noise_std=args.noise_std,
        lr=args.gd_lr,
        T=args.T,
        split='test',
        no_cot=args.no_cot
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
        n_examples=args.n_examples,
        no_cot=args.no_cot,
        use_noise_injection=args.use_noise_injection,
        noise_sigma=args.train_noise_sigma
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
    
    max_steps = args.epochs * len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        
        # Check if we should regenerate CoT for no_cot_dataset
        if (no_cot_dataset is not None and 
            args.regen_interval > 0 and 
            steps > 0 and steps % args.regen_interval == 0):
            
            print(f"\n[Step {steps}] Regenerating CoT sequences for no-CoT dataset...")
            
            # Generate and filter CoT sequences
            filtered_samples = generate_and_filter_cot(
                model, 
                no_cot_dataset, 
                device, 
                args.cot_error_threshold,
                batch_size=args.batch_size,
                inject_noise=args.cot_generate_inject_noise,
                noise_sigma=args.cot_generate_noise_sigma
            )
            # shuffle
            filtered_samples = sorted(filtered_samples, key=lambda x: x[1])
            
            if len(filtered_samples) > 0:
                # Create new mixed dataset
                train_dataset = MixedCoTDataset(cot_dataset, filtered_samples)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers
                )
                print(f"Updated training set: {len(cot_dataset)} original + {len(filtered_samples)} generated = {len(train_dataset)} total")
            else:
                print("No samples passed the error threshold, keeping original dataset")
        
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
                
                # Evaluate noise stability
                perturbed_mean_loss, perturbed_std_loss = evaluate_noise_stability(model, test_loader, device, sigma=args.sigma, runs=5)
                perturbed_mean_loss = perturbed_mean_loss - val_loss
                
                print(f"Steps {steps} - Train Loss: {avg_train_loss:.6f}, Test Loss: {val_loss:.6f}")
                print(f"Perturbed Loss: {perturbed_mean_loss:.6f} ± {perturbed_std_loss:.6f}")
                # print(model.attention.W_kq.weight)
                # print(model.attention.W_pv.weight)
                # Log to wandb
                if args.use_wandb:
                    wandb.log({
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'perturbed_mean_loss': perturbed_mean_loss,
                        'perturbed_std_loss': perturbed_std_loss,
                        'lr': optimizer.param_groups[0]['lr']
                    }, step=steps)
                
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
                
            if args.regen_interval > 0 and steps % args.regen_interval == 0:
                break
        
        # # At end of epoch, if no regeneration during epoch, continue as normal
        # train_loader = DataLoader(
        #     cot_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=args.num_workers
        # )
        
        if steps >= max_steps:
            break
                    
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
    parser.add_argument('--no_cot', action='store_true', 
                        help='Disable chain-of-thought generation')
    
    # Noise injection parameters
    parser.add_argument('--use_noise_injection', action='store_true',
                        help='Inject Gaussian noise into model weights during forward pass for training robustness')
    parser.add_argument('--train_noise_sigma', type=float, default=1e-3,
                        help='Standard deviation of weight noise injection during training')
    
    # CoT generation and filtering parameters
    parser.add_argument('--cot_ratio', type=float, default=1.0,
                        help='Ratio of training data with ground-truth CoT (0.0-1.0)')
    parser.add_argument('--regen_interval', type=int, default=0,
                        help='Steps between CoT regeneration (0 to disable)')
    parser.add_argument('--cot_error_threshold', type=float, default=1.0,
                        help='Maximum MSE error threshold to accept generated CoT')
    parser.add_argument('--cot_generate_inject_noise', action='store_true',
                        help='Inject noise into model weights during CoT generation for robustness')
    parser.add_argument('--cot_generate_noise_sigma', type=float, default=1e-3,
                        help='Standard deviation of weight noise during CoT generation')
    
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
    parser.add_argument('--sigma', type=float, default=1e-3,
                        help='Standard deviation of weight perturbation for noise stability evaluation')
    
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
