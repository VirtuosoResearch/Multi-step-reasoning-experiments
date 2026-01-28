"""
Noise Injection Multitask Model

This module implements a MultitaskModel with noise injection during training.
The noise injection follows the method described:
- Forward: W_noisy = W + €, where € is constant noise sampled in no_grad()
- Backward: Gradients flow to W, but not to € (since € is treated as constant)

This is a standalone class that can be used instead of MultitaskModel for training with noise injection.
"""

import torch
import torch.nn.functional as F

from src.custom.multitask_model import MultitaskModel


class NoiseInjectionMultitaskModel(MultitaskModel):
    """
    A MultitaskModel with noise injection during training.
    
    This class extends MultitaskModel to add noise injection to trainable parameters
    during the forward pass. The noise is sampled in no_grad() context, so gradients
    flow back to the original weights but not to the noise itself.
    
    Args (in addition to MultitaskModel args):
        use_noise_injection: Whether to enable noise injection (default: False)
        noise_std: Standard deviation of Gaussian noise (default: 1e-3)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        model_type: str,
        use_cpu_offload=False,
        lr=3e-4,
        truncate_early=True,
        max_length=1024,
        max_output_length=64,
        weight_decay=1e-4,
        use_wandb=False,
        optimizer="adamw",
        generate_output=True,
        task_names=[],
        use_sample_weights=False,
        fit_least_square=False,
        compute_gradients=False,
        compute_gradients_seed=0,
        project_gradients_dim=200,
        gradients_dir="test",
        compute_gradients_steps=1e7,
        start_step=0,
        only_compute_outputs=False,
        evaluate_cot=False,
        train_invariant_mix=False,
        eval_math=False,
        eval_clrs=False,
        eval_step_num=0,
        # Noise injection parameters
        use_noise_injection=False,
        noise_std=1e-3,
    ):
        # Initialize parent class
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            use_cpu_offload=use_cpu_offload,
            lr=lr,
            truncate_early=truncate_early,
            max_length=max_length,
            max_output_length=max_output_length,
            weight_decay=weight_decay,
            use_wandb=use_wandb,
            optimizer=optimizer,
            generate_output=generate_output,
            task_names=task_names,
            use_sample_weights=use_sample_weights,
            fit_least_square=fit_least_square,
            compute_gradients=compute_gradients,
            compute_gradients_seed=compute_gradients_seed,
            project_gradients_dim=project_gradients_dim,
            gradients_dir=gradients_dir,
            compute_gradients_steps=compute_gradients_steps,
            start_step=start_step,
            only_compute_outputs=only_compute_outputs,
            evaluate_cot=evaluate_cot,
            train_invariant_mix=train_invariant_mix,
            eval_math=eval_math,
            eval_clrs=eval_clrs,
            eval_step_num=eval_step_num,
        )
        
        # Store noise injection parameters
        self.use_noise_injection = use_noise_injection
        self.noise_std = noise_std
    
    def on_fit_start(self) -> None:
        """Called when training starts."""
        super().on_fit_start()
        if self.use_noise_injection:
            num_trainable = sum(1 for p in self.model.named_parameters() if p[1].requires_grad)
            print(f"[NoiseInjection] Enabled with Gaussian noise (std={self.noise_std})")
            print(f"[NoiseInjection] Will apply noise to {num_trainable} trainable parameters")
    
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - NO noise injection.
        Ensures clean weights are used for validation.
        """
        # Explicitly ensure no noise is applied during validation
        # Remove any residual noise before validation
        if hasattr(self, 'param_noise') and self.param_noise:
            self._remove_noise_from_params()
        
        # Call parent validation step (no noise will be applied)
        return super().validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        """
        Test step - NO noise injection.
        Ensures clean weights are used for testing.
        """
        # Explicitly ensure no noise is applied during test
        # Remove any residual noise before test
        if hasattr(self, 'param_noise') and self.param_noise:
            self._remove_noise_from_params()
        
        # Call parent test step (no noise will be applied)
        return super().test_step(batch, batch_idx)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with noise injection.
        
        According to the noise injection method:
        - Forward: W_noisy = W + €, where € is constant noise sampled in no_grad()
        - Backward: Gradients flow to W, but not to € (since € is treated as constant)
        
        Process:
        1. At the start of each batch: Add random noise to all trainable parameters
        2. Forward pass: Use noisy weights
        3. Backward pass: Gradients flow to original weights (not noise)
        4. After backward: Remove the noise that was added at the start
        """
        # Step 1: Apply random noise to all trainable parameters at the start of this batch
        if self.use_noise_injection and self.training:
            self._apply_noise_to_params()
        
        # Step 2: Call parent training step (forward + backward happens here)
        # The model will use the noisy weights during forward pass
        loss = super().training_step(batch, batch_idx)
        
        # Step 3: Noise will be removed in on_after_backward() after backward pass completes
        
        return loss
    
    def on_after_backward(self) -> None:
        """
        Called after backward pass completes.
        Remove the noise that was added at the start of this batch.
        """
        # Remove noise after backward pass completes
        if self.use_noise_injection and self.training:
            self._remove_noise_from_params()
        
        # Call parent's on_after_backward if it exists
        # (PyTorch Lightning's LightningModule has this method, but it may be empty)
        try:
            super().on_after_backward()
        except AttributeError:
            # Parent doesn't have this method, which is fine
            pass
    
    def _apply_noise_to_params(self):
        """
        Apply random noise to all trainable parameters at the start of each batch.
        
        This implements: W_noisy = W + €
        where € is random noise sampled in no_grad() context (making it a constant).
        Each batch gets a fresh random noise sample.
        
        IMPORTANT: Noise gradient guarantee:
        - Noise is sampled in torch.no_grad() context, so noise.requires_grad = False
        - When we do param.data.add_(noise), the noise tensor has no gradient tracking
        - During backward: ∂L/∂noise = 0 (because noise is not in computation graph)
        - Gradients only flow to original W: ∂L/∂W = ∂L/∂W_noisy (since ∂W_noisy/∂W = 1)
        
        The noise is stored in self.param_noise so it can be removed after backward.
        """
        # Only apply noise during training
        if not (self.training and self.use_noise_injection):
            return
        
        # First, remove any previously applied noise to avoid accumulation
        # (This should not happen if we properly remove noise after backward, but safety check)
        if hasattr(self, 'param_noise') and self.param_noise:
            for name, param in self.model.named_parameters():
                if name in self.param_noise:
                    param.data.sub_(self.param_noise[name])
        
        # Initialize or clear noise storage for this batch
        self.param_noise = {}
        
        # Apply fresh random noise to all trainable parameters for this batch
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Sample random Gaussian noise in no_grad() context - this makes € a constant
                # CRITICAL: noise.requires_grad = False, so ∂L/∂noise = 0
                # Each batch gets a new random noise sample
                with torch.no_grad():
                    noise = torch.randn_like(param) * self.noise_std
                
                # Verify noise has no gradient tracking (safety check)
                assert not noise.requires_grad, "Noise must not require gradients!"
                
                # Store noise so we can remove it after backward pass
                self.param_noise[name] = noise
                
                # Add noise to parameter: W_noisy = W + €
                # Since € is a constant (sampled in no_grad), gradients flow to W but not to €
                # ∂L/∂W = ∂L/∂W_noisy * ∂W_noisy/∂W = ∂L/∂W_noisy * 1 = ∂L/∂W_noisy
                # ∂L/∂noise = 0 (noise is not in computation graph)
                param.data.add_(noise)
    
    def _remove_noise_from_params(self):
        """
        Remove the noise that was added at the start of this batch.
        
        This is called after backward pass completes to restore original weights.
        Can also be called during validation/test to ensure clean weights.
        """
        if not hasattr(self, 'param_noise') or not self.param_noise:
            return
        
        # Remove the noise that was added at the start of this batch
        for name, param in self.model.named_parameters():
            if name in self.param_noise:
                # Remove the noise: W = W_noisy - €
                param.data.sub_(self.param_noise[name])
        
        # Clear noise storage (will be re-populated in next training batch)
        self.param_noise = {}
