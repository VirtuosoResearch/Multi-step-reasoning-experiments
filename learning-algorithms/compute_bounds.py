"""
Compute PAC-Bayes generalization bounds for CLRS text models.
Adapted from SubLoRA's bound computation framework.
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm
from contextlib import nullcontext

from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from src.custom.clrs_text_task_graph_data_module import TextGraphCLRSDataModule
from src.custom.multitask_model import MultitaskModel
from src.model.projectors import create_intrinsic_model

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from adapters import SeqBnInvConfig, PrefixTuningConfig, BnConfig, DoubleSeqBnConfig, SeqBnConfig
from adapters import AutoAdapterModel,list_adapters, BnConfig

# Import bound computation utilities from SubLoRA
from sublora.bounds.compute_bounds import llm_subsampling_bound
import sublora.bounds.quantize_fns as quantize

logging.basicConfig(level=logging.INFO, force=True)

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) 
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key, torch_dtype=torch.bfloat16)
        model_type = "decoder"
        append_eos = True
    elif "Qwen" in model_key:
        hf_key = args.model_key.replace("_", "-")
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) 
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=args.max_length+ args.max_output_length)
        model_type = "decoder"
        append_eos = True
    else:
        raise NotImplementedError(args.model_key)

    if args.train_adapter:
        
        if args.use_qadapter:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4' 
            )

            model = AutoAdapterModel.from_pretrained(
                hf_key, 
                quantization_config=quantization_config, 
                torch_dtype=torch.bfloat16, 
                device_map={"": args.devices[0]}
            )
        
        else: model = AutoAdapterModel.from_pretrained(hf_key)

        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,    
            output_adapter=True,    
            reduction_factor=args.reduction_factor,     
            non_linearity="relu"     
        )

        model.add_adapter(adapter_name="seq_bn",config=bottleneck_config)

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count/all_params_count}")
        print("-"*20,"Bottleneck_Adapter","-"*20)        

    elif args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "k", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        model = get_peft_model(model, config)

        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if args.intrinsic_dim > 0:
        model = create_intrinsic_model(base_net=model,
                                        ckpt_path=None,
                                        intrinsic_mode=args.intrinsic_mode,
                                        intrinsic_dim=args.intrinsic_dim,
                                        seed=137,
                                        data_type="bfloat16" if args.precision == "bf16-true" else "float32")  
        
        print(f"trainable params (M) after applying projection = {model.get_num_params(only_trainable=True)}")

    return model, tokenizer, hf_key, model_type, append_eos

@torch.no_grad()
def compute_bpd_on_batch(model, batch, tokenizer, device, ctx, vocab_size, alpha_array):
    """
    Compute bits-per-dimension (BPD) for a batch.
    Similar to SubLoRA's compute_bound_scores but adapted for CLRS format.
    """
    batch = batch['data']
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Filter out padding tokens (label = -100)
    valid_mask = shift_labels != -100
    shift_logits = shift_logits[valid_mask]
    shift_labels = shift_labels[valid_mask]
    
    if shift_logits.size(0) == 0:
        return None
    
    # Compute softmax probabilities
    softmax_probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    
    # Get probabilities of true tokens
    selected_probs = softmax_probs[torch.arange(softmax_probs.size(0)), shift_labels]
    
    # Compute BPD for different alpha values
    bpd_dict = {}
    num_tokens = selected_probs.size(0)
    
    for alpha in alpha_array:
        # Interpolate between empirical distribution and uniform
        smoothed_probs = (1 - alpha) * selected_probs + alpha / vocab_size
        log_probs = torch.log2(smoothed_probs)
        bpd_alpha = -log_probs.sum().item() / num_tokens
        bpd_dict[f'bpd_alpha_{alpha}'] = bpd_alpha
    
    return bpd_dict, num_tokens


@torch.no_grad()
def evaluate_model_loss(model, dataloader, device, ctx, max_batches=None):
    """
    Evaluate model loss on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader to evaluate on
        device: Device to run on
        ctx: Autocast context for mixed precision
        max_batches: Maximum number of batches to evaluate (None for all)
    
    Returns:
        Average loss across batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating loss")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        batch = batch['data']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def quantize_model(model, train_dataloader, intrinsic_dim, device, ddp, 
                   max_quant_iters, use_kmeans, levels, quant_lr):
    """
    Quantize model parameters with optional quantization-aware training on CLRS dataset.
    
    Args:
        model: The model to quantize
        train_dataloader: CLRS DataLoader for quantization-aware training
        intrinsic_dim: Intrinsic dimension (>0 if using subspace projection)
        device: Device to run on
        ddp: Whether using DistributedDataParallel
        max_quant_iters: Number of quantization-aware training iterations (0 to skip)
        use_kmeans: Whether to use k-means for codebook initialization
        levels: Number of quantization levels
        quant_lr: Learning rate for quantization-aware training
    """
    from torch.optim import SGD
    
    if max_quant_iters > 0 and intrinsic_dim > 0:
        # Quantization-aware training for intrinsic dimension models
        vector = model.subspace_params.cpu().data.numpy()
        cluster_fn = quantize.get_random_symbols_and_codebook
        if use_kmeans:
            cluster_fn = quantize.get_kmeans_symbols_and_codebook
        _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
        centroids = torch.tensor(centroids, dtype=torch.float32)
        centroids = centroids.to(device)
        quantizer_fn = quantize.Quantize().apply
        qw = quantize.QuantizingWrapper(model, quantizer=quantizer_fn, centroids=centroids)
        optim = SGD(
            [qw.subspace_params, qw.centroids],
            lr=quant_lr, momentum=0.9)

        # Iterate over CLRS batches for quantization-aware training
        train_iter = iter(train_dataloader)
        for e in tqdm(range(max_quant_iters), desc="Quantization-aware training"):
            qw.train()
            optim.zero_grad()
            
            # Get batch from CLRS dataloader
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart dataloader if we run out of batches
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            batch = batch['data']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass through quantized model
            outputs = qw(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optim.step()
            
            if e % 10 == 0:
                metrics = {"iter": e, "mini_loss": loss.detach().item()}
                print(metrics)
        
        # Extract quantized parameters and compute message length
        quantized_vec = qw.quantizer(qw.subspace_params, qw.centroids)
        quantized_vec = quantized_vec.cpu().detach().numpy()
        vec = (qw.centroids.unsqueeze(-2) - qw.subspace_params.unsqueeze(-1))**2.0
        symbols = torch.min(vec, -1)[-1]
        symbols = symbols.cpu().detach().numpy()
        centroids = qw.centroids.cpu().detach().numpy()
        probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
        _, coded_symbols_size = quantize.do_arithmetic_encoding(symbols, probabilities,
                                                    qw.centroids.shape[0])
        message_len = quantize.get_message_len(
            coded_symbols_size=coded_symbols_size,
            codebook=centroids,
            max_count=len(symbols),
        )
    # Simple quantization without training
    elif intrinsic_dim > 0:
        module = model.module if isinstance(model,
                                        torch.nn.parallel.DistributedDataParallel) else model
        vector = module.subspace_params.cpu().data.numpy()
        quantized_vec, message_len = quantize.quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)
    else:
        aux = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names, vector = zip(*aux)
        vector_list = [p.cpu().data.numpy().flatten() for p in vector]
        fvector = np.concatenate(vector_list)
        quantized_vec, message_len = quantize.quantize_vector(fvector, levels=levels, use_kmeans=use_kmeans)
        ## free memory 
        fvector = None 

    # Apply quantized weights back to model
    if intrinsic_dim > 0:
        module = model.module if ddp else model
        module.subspace_params.data = torch.tensor(quantized_vec).float().to(device)
    else:
        # Unflatten quantized vector back to original parameter shapes
        unfquantized_vec = []
        offset = 0
        for v in vector:
            numel = v.numel() if hasattr(v, 'numel') else np.prod(v.shape)
            shape = v.shape
            unfquantized_vec.append(quantized_vec[offset:offset+numel].reshape(shape))
            offset += numel
        
        ## free memory  
        quantized_vec, vector = None, None
        
        for n, p in model.named_parameters():
            for name, quantp in zip(names, unfquantized_vec):
                if n == name:
                    p.data = torch.tensor(quantp).float().to(device)
            
    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    
    return model, prefix_message_len

def compute_bounds(args):
    """Main function to compute PAC-Bayes bounds."""
    
    # Initialize model
    print("Initializing model...")
    model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
    
    # Load trained checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded model from {args.checkpoint_path}")
    else:
        raise ValueError(f"Checkpoint not found at {args.checkpoint_path}")
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Setup context for mixed precision
    dtype = torch.bfloat16
    ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=dtype)
    
    # Initialize data module
    print("Loading data...")
    data_module = TextCLRSDataModule(
        task_names=args.task_names,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        inference_batch_size=args.eval_batch_size,
        max_input_length=args.max_length,
        max_output_length=args.max_output_length,
        eval_all=True,
        eval_split=args.eval_split,
        downsample_ratio=args.downsample_ratio,
        minimum_samples=args.minimum_samples,
        minimum_samples_validation=args.minimum_samples_validation,
        train_lengths=args.train_lengths,
        test_lengths=args.test_lengths,
        use_few_shot=(args.few_shot_k > 0), 
        few_shot_k=args.few_shot_k,
        only_answer_output=args.only_answer_output)
    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()
    
    # Get test dataloader for evaluation
    test_dataloader = data_module.test_dataloader()
    
    # Evaluate loss before quantization
    print("\n" + "="*80)
    print("EVALUATING MODEL BEFORE QUANTIZATION")
    print("="*80)
    
    train_loss_before = evaluate_model_loss(model, train_dataloader, device, ctx, max_batches=100)
    test_loss_before = evaluate_model_loss(model, test_dataloader, device, ctx, max_batches=100)
    
    print(f"Train loss before quantization: {train_loss_before:.4f}")
    print(f"Test loss before quantization: {test_loss_before:.4f}")
    
    # Quantize model using the new quantize_model function
    print("\n" + "="*80)
    print("QUANTIZING MODEL")
    print("="*80)
    
    model, prefix_message_len = quantize_model(
        model=model,
        train_dataloader=train_dataloader,
        intrinsic_dim=args.intrinsic_dim,
        device=device,
        ddp=False,
        max_quant_iters=args.max_quant_iters,
        use_kmeans=args.use_kmeans,
        levels=args.levels,
        quant_lr=args.quant_lr
    )
    
    print(f"Prefix message length: {prefix_message_len:.2f} bits")
    print("Quantization complete. Model now uses quantized weights.")
    
    # Set model back to eval mode after potential quantization training
    model.eval()
    
    # Evaluate loss after quantization
    print("\n" + "="*80)
    print("EVALUATING MODEL AFTER QUANTIZATION")
    print("="*80)
    
    train_loss_after = evaluate_model_loss(model, train_dataloader, device, ctx, max_batches=100)
    test_loss_after = evaluate_model_loss(model, test_dataloader, device, ctx, max_batches=100)
    
    print(f"Train loss after quantization: {train_loss_after:.4f}")
    print(f"Test loss after quantization: {test_loss_after:.4f}")
    print(f"Train loss change: {train_loss_after - train_loss_before:+.4f}")
    print(f"Test loss change: {test_loss_after - test_loss_before:+.4f}")
    
    # Compute empirical BPD on training set (with no_grad since only evaluation)
    print("\n" + "="*80)
    print("COMPUTING EMPIRICAL BPD ON TRAINING SET")
    print("="*80)
    with torch.no_grad():
        alpha_array = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
        metrics_dict = {f'bpd_alpha_{alpha}': 0.0 for alpha in alpha_array}
        metrics_dict['n_train'] = 0
        metrics_dict['total_tokens'] = 0
        
        vocab_size = tokenizer.vocab_size
        
        num_samples = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Computing BPD")):
            if num_samples >= args.bound_samples:
                break
            
            bpd_result = compute_bpd_on_batch(
                model, batch, tokenizer, device, ctx, vocab_size, alpha_array
            )
            
            if bpd_result is None:
                continue
            
            bpd_dict, num_tokens = bpd_result
            
            # Update running averages
            for alpha in alpha_array:
                key = f'bpd_alpha_{alpha}'
                current_total = metrics_dict['total_tokens']
                new_total = current_total + num_tokens
                
                if new_total > 0:
                    metrics_dict[key] = (
                        metrics_dict[key] * current_total + bpd_dict[key] * num_tokens
                    ) / new_total
            
            metrics_dict['total_tokens'] = new_total
            metrics_dict['n_train'] += batch['data']['input_ids'].size(0)
            num_samples += batch['data']['input_ids'].size(0)
            
            if batch_idx % 10 == 0:
                print(f"Samples: {num_samples}, Tokens: {metrics_dict['total_tokens']}")
                for alpha in alpha_array[:3]:  # Print first 3 alphas
                    print(f"  BPD (α={alpha}): {metrics_dict[f'bpd_alpha_{alpha}']:.4f}")
        
        sample_size = metrics_dict['n_train']
        print(f"\nTotal samples processed: {sample_size}")
        print(f"Total tokens processed: {metrics_dict['total_tokens']}")
    
    # Compute PAC-Bayes bounds
    print("\n" + "="*80)
    print("COMPUTING PAC-BAYES BOUNDS")
    print("="*80)
    
    bounds_dict = {}
    bounds_dict["prefix_message_len"] = float(prefix_message_len)
    bounds_dict["sample_size"] = sample_size
    bounds_dict["total_tokens"] = metrics_dict['total_tokens']
    
    # Add loss metrics to bounds_dict
    bounds_dict["train_loss_before_quant"] = float(train_loss_before)
    bounds_dict["test_loss_before_quant"] = float(test_loss_before)
    bounds_dict["train_loss_after_quant"] = float(train_loss_after)
    bounds_dict["test_loss_after_quant"] = float(test_loss_after)
    
    # Estimate total dataset size (number of samples)
    # For CLRS, this depends on the task and length
    data_size = len(data_module.train_dataset) if hasattr(data_module, 'train_dataset') else sample_size * 2
    bounds_dict["data_size"] = data_size
    
    misc_extra_bits = args.misc_extra_bits
    misc_extra_bits += np.ceil(np.log2(len(alpha_array)))  # Cost of searching over alphas
    
    divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
    bounds_dict["divergence"] = float(divergence)
    
    best_bpd_bound = np.inf
    
    for alpha in alpha_array:
        key = f'bpd_alpha_{alpha}'
        train_bpd = metrics_dict[key]
        
        # Scaling factor for BPD bound
        delta = np.log2(1 + (1 - alpha) * vocab_size / alpha)
        
        # Compute bound
        bound = llm_subsampling_bound(
            train_error=train_bpd,
            div=divergence,
            data_size=data_size,
            sample_size=sample_size,
            delta=delta,
            epsilon=0.05
        )
        
        bounds_dict[f"train_{key}"] = float(train_bpd)
        bounds_dict[f"bound_{key}"] = float(bound)
        
        if bound < best_bpd_bound:
            best_bpd_bound = bound
            best_alpha = alpha
    
    bounds_dict["best_bpd_bound"] = float(best_bpd_bound)
    bounds_dict["best_alpha"] = float(best_alpha)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Prefix message length: {prefix_message_len:.2f} bits")
    print(f"Divergence: {divergence:.2f}")
    print(f"Sample size: {sample_size}")
    print(f"Data size: {data_size}")
    print(f"\nLoss Evaluation:")
    print(f"  Train loss before quantization: {train_loss_before:.4f}")
    print(f"  Train loss after quantization:  {train_loss_after:.4f} ({train_loss_after - train_loss_before:+.4f})")
    print(f"  Test loss before quantization:  {test_loss_before:.4f}")
    print(f"  Test loss after quantization:   {test_loss_after:.4f} ({test_loss_after - test_loss_before:+.4f})")
    print(f"\nBest BPD bound: {best_bpd_bound:.4f} (α={best_alpha})")
    print("\nAll bounds:")
    for alpha in alpha_array:
        train_key = f"train_bpd_alpha_{alpha}"
        bound_key = f"bound_bpd_alpha_{alpha}"
        print(f"  α={alpha}: train={bounds_dict[train_key]:.4f}, bound={bounds_dict[bound_key]:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    bounds_file = os.path.join(args.output_dir, 'bounds.yml')
    with open(bounds_file, 'w') as f:
        yaml.safe_dump(bounds_dict, f, indent=2)
    print(f"\nResults saved to {bounds_file}")
    
    metrics_file = os.path.join(args.output_dir, 'metrics.yml')
    with open(metrics_file, 'w') as f:
        yaml.safe_dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    return bounds_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PAC-Bayes bounds for CLRS text models")
    
    # Model arguments
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", type=str, default="float32")
    
    # LoRA arguments
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action="store_true")
    
    # Intrinsic dimension arguments
    parser.add_argument("--intrinsic_dim", type=int, default=0)
    parser.add_argument("--intrinsic_mode", type=str, default="rdkronqr")
    
    # Data arguments
    parser.add_argument("--task_names", type=str, nargs="+", default=['dfs'])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--train_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--test_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1e6)
    parser.add_argument("--minimum_samples_validation", type=int, default=1e6)
    parser.add_argument("--few_shot_k", type=int, default=0,
                       help="Number of few-shot examples to prepend (0 to disable)")
    parser.add_argument("--only_answer_output", action="store_true")
    
    # Bound computation arguments
    parser.add_argument("--eval_batch_size", type=int, default=8,
                       help="Batch size for bound evaluation")
    parser.add_argument("--bound_samples", type=int, default=1000,
                       help="Number of samples to use for bound computation")
    parser.add_argument("--levels", type=int, default=11,
                       help="Number of quantization levels")
    parser.add_argument("--use_kmeans", action="store_true",
                       help="Use k-means for quantization")
    parser.add_argument("--max_quant_iters", type=int, default=0,
                       help="Number of quantization-aware training iterations (0 to skip)")
    parser.add_argument("--quant_lr", type=float, default=0.01,
                       help="Learning rate for quantization-aware training")
    parser.add_argument("--misc_extra_bits", type=int, default=5,
                       help="Extra bits for hyperparameter search")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save bound results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CLRS TEXT BOUND COMPUTATION")
    print("="*80)
    print(f"Model: {args.model_key}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Task: {args.task_names}")
    print(f"Intrinsic dim: {args.intrinsic_dim}")
    print(f"LoRA rank: {args.lora_rank if args.train_lora else 'N/A'}")
    print(f"Quantization levels: {args.levels}")
    print(f"Quantization-aware training iters: {args.max_quant_iters}")
    print(f"Bound samples: {args.bound_samples}")
    print("="*80)
    
    compute_bounds(args)
