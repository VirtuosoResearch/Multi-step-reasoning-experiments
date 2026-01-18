import copy
import json
import logging
from typing import List, Dict
import wandb
from collections import defaultdict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
from src.utils.compute_metrics import compute_accuracy
from pynvml import *
from src.utils.math_utils import eval_results_math, eval_results_gsm8k

def print_gpu_utilization():
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if device_count == 0:
            print("No GPU devices found.")
            return
        
        # Use GPU 0 by default, or the last available GPU if only one exists
        gpu_index = min(1, device_count - 1) if device_count > 1 else 0
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        info = nvmlDeviceGetMemoryInfo(handle)
        # print(info.used, info.total)
        print(f"GPU {gpu_index} memory occupied: {info.used//1024**2} MB.")
    except Exception as e:
        print(f"Failed to get GPU utilization: {e}")


'''
TODO:
- Modify compute gradients
'''

def _find_sep_positions(answer_ids_1d: torch.Tensor, sep_token_id: int):
    # answer_ids_1d: [T_ans], no pad
    # return positions where token == sep_token_id
    return (answer_ids_1d == sep_token_id).nonzero(as_tuple=False).view(-1)

@torch.no_grad()
def compute_step_losses_from_concatenated(
    model,
    tokenizer,
    input_ids: torch.Tensor,          # [B, T] full sequence (input + labels already concatenated in input_ids)
    labels: torch.Tensor,              # [B, T] with -100 for input tokens, and answer tokens as ids
    attention_mask: torch.Tensor,      # [B, T] attention mask
    sep_strs: List[str] = [",", "|"],  # List of separator strings
    pad_token_id: int = None,
):
    """
    Compute step losses by using input_ids (which contains input + gold_answers concatenated),
    forward through model, mask input part, and split by separators.
    
    Args:
        model: The model to use for forward pass
        tokenizer: Tokenizer for encoding separator strings
        input_ids: [B, T] full sequence (input + gold_answers concatenated)
        labels: [B, T] with -100 for input tokens, and answer tokens as ids
        attention_mask: [B, T] attention mask
        sep_strs: List of separator strings to split steps (e.g., [",", "|"])
        pad_token_id: Pad token id (if None, use tokenizer.pad_token_id)
    
    Returns:
        step_loss_list: List[List[float]] length B, each inner list per step avg loss
        final_loss_list: List[float] length B, final step loss for each sample
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    step_loss_list = []
    final_loss_list = []
    
    # 1) Extract input and gold_answers for each batch
    for b in range(batch_size):
        # Get input part: positions where labels == -100
        input_mask = labels[b] == -100
        input_tokens = input_ids[b, input_mask]  # [T_in]
        
        # Get gold answer part: positions where labels != -100 and != pad
        gold_mask = (labels[b] != -100) & (labels[b] != pad_token_id)
        gold_tokens = labels[b, gold_mask]  # [T_gold]
        
        if gold_tokens.numel() == 0:
            step_loss_list.append([])
            final_loss_list.append(None)
            continue
        
        # 2) Decode to string and split by separators at string level
        input_str = tokenizer.decode(input_tokens, skip_special_tokens=True)
        gold_str = tokenizer.decode(gold_tokens, skip_special_tokens=True)
        
        # Split gold_str by separators while preserving separator information
        import re
        # Create a regex pattern that matches any separator
        sep_pattern = "|".join([re.escape(sep) for sep in sep_strs])
        # Split but keep separators in the result
        parts = re.split(f"({sep_pattern})", gold_str)
        # print(f"???????????????????????????????????????\nparts: {parts}\n???????????????????????????????????????\n")
        
        # Reconstruct steps and separators
        gold_steps_str = []
        separators_after_steps = []  # Separator after each step (None for last step)
        
        current_step = ""
        for i, part in enumerate(parts):
            if part in sep_strs:
                # This is a separator
                if current_step.strip():
                    gold_steps_str.append(current_step.strip())
                    separators_after_steps.append(part)
                    current_step = ""
            else:
                current_step += part
        
        # Add the last step if not empty
        if current_step.strip():
            gold_steps_str.append(current_step.strip())
            separators_after_steps.append(None)  # No separator after last step
        
        # Filter out empty steps
        if len(gold_steps_str) == 0:
            step_loss_list.append([])
            final_loss_list.append(None)
            continue
        
        # 3) For each step, compute loss
        per_step_losses = []
        accumulated_output = ""  # Accumulated output from previous steps (including separators)
        
        for step_idx, current_step_str in enumerate(gold_steps_str):
            # Construct input: original_input + accumulated_output
            if accumulated_output:
                step_input_str = input_str + accumulated_output
            else:
                step_input_str = input_str
            
            # Tokenize step input and output
            step_input_ids = tokenizer(step_input_str, return_tensors="pt", padding=False, truncation=True)["input_ids"].to(device)[0]  # [T_step_in]
            step_output_ids = tokenizer(current_step_str, return_tensors="pt", padding=False, truncation=True)["input_ids"].to(device)[0]  # [T_step_out]
            
            # Concatenate input and output
            step_concat_ids = torch.cat([step_input_ids, step_output_ids], dim=0)  # [T_step_in + T_step_out]
            step_concat_attention = torch.ones_like(step_concat_ids, dtype=torch.long)
            
            # Create labels: mask input part, keep output part
            step_labels = step_concat_ids.clone()
            step_labels[:step_input_ids.size(0)] = -100  # Mask input part
            
            # Forward through model
            step_concat_ids = step_concat_ids.unsqueeze(0)  # [1, T]
            step_concat_attention = step_concat_attention.unsqueeze(0)  # [1, T]
            step_labels = step_labels.unsqueeze(0)  # [1, T]
            
            out = model(input_ids=step_concat_ids, attention_mask=step_concat_attention)
            step_logits = out.logits  # [1, T, V]
            
            # Compute loss for this step
            shift_logits = step_logits[:, :-1, :].contiguous()  # [1, T-1, V]
            shift_labels = step_labels[:, 1:].contiguous()  # [1, T-1]
            
            # Only compute loss on output part (where shift_labels != -100)
            valid_mask = shift_labels != -100
            if valid_mask.sum().item() == 0:
                per_step_losses.append(0.0)
                accumulated_output += current_step_str
                # Add separator after this step if it exists
                if step_idx < len(separators_after_steps) and separators_after_steps[step_idx] is not None:
                    accumulated_output += separators_after_steps[step_idx]
                continue
            
            # Compute cross entropy loss
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            step_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)
            step_losses = step_losses[valid_mask.view(-1)]
            
            if step_losses.numel() > 0:
                per_step_losses.append(step_losses.mean().item())
            else:
                per_step_losses.append(0.0)
            
            # Update accumulated output for next step
            accumulated_output += current_step_str
            # Add separator after this step if it exists
            if step_idx < len(separators_after_steps) and separators_after_steps[step_idx] is not None:
                accumulated_output += separators_after_steps[step_idx]
        
        # Get final step loss (last step)
        if len(per_step_losses) > 0:
            final_loss = per_step_losses[-1]
        else:
            final_loss = None
        
        step_loss_list.append(per_step_losses)
        final_loss_list.append(final_loss)
    
    return step_loss_list, final_loss_list

class MultitaskModel(pl.LightningModule):
    validation_predictions: Dict

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, model_type: str, use_cpu_offload=False,
                lr=3e-4, truncate_early=True, max_length=1024, max_output_length = 64, weight_decay=1e-4, use_wandb=False,
                optimizer="adamw", generate_output=True, task_names=[],
                # more advanced parameters (set to default if only fine-tuning)
                use_sample_weights=False, fit_least_square = False, compute_gradients = False,
                compute_gradients_seed = 0, project_gradients_dim = 200, gradients_dir = "test", 
                compute_gradients_steps = 1e7, start_step = 0, only_compute_outputs = False,
                evaluate_cot=False, train_invariant_mix=False, eval_math=False, eval_clrs=False, eval_step_num=0):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.lr = lr
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.truncate_early = truncate_early
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        self.validation_step_outputs = []
        self.optimizer = optimizer
        self.generate_output = generate_output
        self.task_names = task_names
        self.use_sample_weights = use_sample_weights # AdaBoost
        self.fit_least_square = fit_least_square # Gradient Boosting

        self.compute_gradients = compute_gradients
        self.gradients_dim = 0
        self.removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]
        for name, param in model.named_parameters():
            if any([key in name for key in self.removing_keys]):
                continue
            if param.requires_grad:
                self.gradients_dim += param.numel()
        self.project_gradients_dim = project_gradients_dim
        self.compute_gradients_seed = compute_gradients_seed
        self.only_compute_outputs = only_compute_outputs
        if compute_gradients:
            self.gradients_dir = f"./gradients/{gradients_dir}"
            if not os.path.exists(self.gradients_dir):
                os.makedirs(self.gradients_dir)
            for task_name in self.task_names:
                task_dir = os.path.join(self.gradients_dir, task_name)
                if not os.path.exists(task_dir):
                    os.makedirs(task_dir)
            self.initialize_project_matrix(self.compute_gradients_seed)
        
        self.param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        self.compute_gradients_steps = compute_gradients_steps
        self.start_step = start_step

        self.evaluate_cot = evaluate_cot # For GraphWiz
        self.eval_math = eval_math # For MATH/GSM8K dataset
        self.eval_clrs = eval_clrs # For CLRS dataset
        self.eval_step_num = eval_step_num # Number of intermediate steps to evaluate (0 means only final, N means first N steps + final)
        self.train_invariant_mix = train_invariant_mix

    def initialize_project_matrix(self, seed):
        """
        Initialize the project matrix with random values.
        """
        print("Creating project matrix with dimensions: ", self.gradients_dim, self.project_gradients_dim, "seed: ", seed)
        np.random.seed(seed)
        if self.project_gradients_dim < 0:
            self.project_matrix = None
        else:
            self.project_matrix = (2 * np.random.randint(2, size=(self.gradients_dim, self.project_gradients_dim)) - 1).astype(float)
            self.project_matrix *= 1 / np.sqrt(self.project_gradients_dim)
        self.current_project_seed = seed

    def get_trainable_parameters(self):
        return [param for name, param in self.model.named_parameters()\
                if (name in self.param_names) and (not any([key in name for key in self.removing_keys]))]

    def on_validation_end(self) -> None:
        return super().on_validation_end()

    def training_step(self, batch, batch_idx):
        task_name = batch["task_name"]; batch = batch["data"]
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if "graph_data" in batch:
            kwargs["graph_data"] = batch["graph_data"]
            kwargs["original_input_ids"] = batch["original_input_ids"]
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        if self.train_invariant_mix:
            kwargs["invariant_mask"] = batch["invariant_mask"]
        
        if self.use_sample_weights:
            logits = self.model(**kwargs)["logits"]
            labels = batch["labels"]
            weights = batch["weights"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = F.cross_entropy
            weights = (torch.ones_like(shift_labels)*weights.view(-1, 1)).view(-1)
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = (loss_fct(shift_logits, shift_labels, reduction="none")*weights).sum()/torch.sum(shift_labels!=-100) # normalize by batch size
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        elif self.fit_least_square:
            logits = self.model(**kwargs)["logits"]
            labels = batch["labels"] 
            residuals = batch['residuals'] 

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            is_label_valid = shift_labels != -100
            shift_logits = shift_logits[is_label_valid].view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels[is_label_valid].view(-1)
            residuals = residuals[is_label_valid.sum(dim=1)>0].float()

            # flatten the tokens
            loss_fct = F.mse_loss
            correct_class_logits = shift_logits[range(shift_labels.size(0)), shift_labels]
            loss = loss_fct(correct_class_logits, residuals) # reduction="sum"
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            loss = self.model(**kwargs)["loss"]
            if self.use_wandb:
                wandb.log({"train_loss": loss})
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """        
        logging.info(f"==========================================This is the validation step for task {batch['task_name']}==========================================")
        task_name = batch["task_name"]; batch = batch["data"]
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if "graph_data" in batch:
            kwargs["graph_data"] = batch["graph_data"]
            kwargs["original_input_ids"] = batch["original_input_ids"]
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        forward_output = self.model(**kwargs)

        if batch_idx == 1:
            print_gpu_utilization()

        if self.model_type == "encoder_decoder":
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type)) # TODO
        elif self.model_type == "decoder":            
            ''' Generate the labels '''

            gold_answers = batch["labels"].clone()
            gold_answers[gold_answers == -100] = self.tokenizer.pad_token_id
            output_len = (gold_answers != self.tokenizer.pad_token_id).sum(dim=1).max().item()

            ''' Compute the logits of the labels '''
            is_label_mask = batch["labels"][:, 1:].contiguous() != -100
            logits = forward_output["logits"][:, :-1].contiguous()
            preds = logits[is_label_mask]
            preds = torch.argmax(preds, dim=-1).cpu().numpy()

            labels = batch["labels"][:, 1:][is_label_mask]
            labels = labels.cpu().numpy()

            # Compute step losses
            step_losses, final_losses = compute_step_losses_from_concatenated(
                self.model,
                self.tokenizer,
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
                sep_strs=[",", "|"],
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generate_output = self.generate_output
            if hasattr(self.model, "only_train_graph") and self.model.only_train_graph:
                generate_output = False
            if generate_output:
                # Remove labels in inputs_ids
                is_label_mask = batch["labels"] != -100
                batch["input_ids"][is_label_mask] = self.tokenizer.pad_token_id
                is_input_mask = batch["input_ids"] != self.tokenizer.pad_token_id
                batch["attention_mask"][is_label_mask] = 0

                if "graph_data" in batch:
                    # convert to left padding
                    new_input_ids = []
                    for tmp_input_ids in batch["input_ids"]:
                        tmp_input_ids = tmp_input_ids[tmp_input_ids != self.tokenizer.pad_token_id]
                        new_input_ids.append(tmp_input_ids)
                    new_input_ids = torch.stack(new_input_ids)
                    inputs = self.tokenizer.batch_decode(new_input_ids, skip_special_tokens=False)
                    self.tokenizer.padding_side = 'left'
                    inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                    self.tokenizer.padding_side = 'right'
                    inputs = self.transfer_batch_to_device(inputs, self.device, batch_idx)

                    output = self.model.generate(**inputs, graph_data=batch["graph_data"], max_new_tokens=self.max_output_length,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id).detach()
                else:
                    # convert to left padding
                    inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    self.tokenizer.padding_side = 'left'
                    inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                    self.tokenizer.padding_side = 'right'
                    inputs = self.transfer_batch_to_device(inputs, self.device, batch_idx)
                    output = self.model.generate(**inputs, max_new_tokens=self.max_output_length,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                do_sample=True, temperature=1.0
                                                ).detach()
                input_len = inputs["input_ids"].shape[1]
                output[:, :input_len] = self.tokenizer.pad_token_id
                if not self.evaluate_cot:
                    output[:, input_len+output_len:] = self.tokenizer.pad_token_id
            else:
                output = None
        else:
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

        output_dict = {
            "task_name": task_name,
            "loss": forward_output['loss'],
            "answers": gold_answers,
            "generates": output,
            "pred_ids": preds,
            "label_ids": labels,
            "step_losses": step_losses,
            "final_losses": final_losses,
        }
        # logging.info(f"step_losses: {step_losses}")
        if hasattr(self.model, "only_train_graph") and self.model.only_train_graph:
            output_dict.update({"graph_accuracy": forward_output.graph_accuracy})
        if "only_answer" in batch:
            output_dict.update({"only_answer": batch["only_answer"]})
        self.validation_step_outputs.append(output_dict)
        return output_dict
    
    def predict_step(self, batch, batch_idx):
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """
        if batch_idx < self.start_step:
            return {}
        if batch_idx >= self.compute_gradients_steps:
            return {}
        if self.compute_gradients:
            torch.set_grad_enabled(True)

        # forward pass
        task_name = batch["task_name"]
        batch = batch["data"]        
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if "graph_data" in batch:
            kwargs["graph_data"] = batch["graph_data"]
            kwargs["original_input_ids"] = batch["original_input_ids"]
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        forward_output = self.model(**kwargs)

        assert self.model_type == "decoder", "Only decoder model type is supported for prediction"
        ''' Compute the logits of the labels '''
        is_label_mask = batch["labels"][:, 1:].contiguous() != -100
        logits = forward_output["logits"][:, :-1].contiguous()
        labels = batch["labels"][:, 1:].contiguous()

        # obtain the outputs and gradients of the outputs
        if self.compute_gradients:
            gradients = []; returned_outputs = np.zeros(labels.shape[0])
            label_counts = is_label_mask.sum(dim = 1)
            tmp_logits = logits[is_label_mask]
            tmp_probs = torch.softmax(tmp_logits, dim=-1)
            tmp_labels = labels[is_label_mask]

            outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
            outputs[outputs>0.9] -= 1e-2 # in case (1-tmp_outputs) is less than zero
            outputs[outputs<0.001] += 1e-2            
            outputs = torch.log(outputs/(1-outputs))
            
            if not self.only_compute_outputs:
                for i in range(len(labels)):
                    start = label_counts[:i].sum() if i > 0 else 0  
                    end = label_counts[:i+1].sum()
                    if end <= start :
                        print("No gradients to compute for this sample")
                        continue
                    else:
                        tmp_outputs = outputs[start:end] # taking the average over the output positions
                        tmp_gradient = torch.autograd.grad(tmp_outputs.mean(), self.get_trainable_parameters(), retain_graph=True, create_graph=False)
                        tmp_gradient = torch.cat([gradient.reshape(-1) for gradient in tmp_gradient]).cpu().type(torch.float32).numpy() # flatten gradients
                        if self.project_matrix is None:
                            tmp_seed_gradient = tmp_gradient
                        else:
                            tmp_seed_gradient = (tmp_gradient.reshape(1, -1) @ self.project_matrix).flatten()
                        gradients.append(tmp_seed_gradient)
                        returned_outputs[i] = tmp_outputs.clone().detach().mean().cpu().type(torch.float32).item()
                gradients = [np.array(gradient) for gradient in gradients]
                np.save(f"{self.gradients_dir}/{task_name}/train_batch_{batch_idx}_gradients.npy", gradients)
            else:
                for i in range(len(labels)):
                    start = label_counts[:i].sum() if i > 0 else 0  
                    end = label_counts[:i+1].sum()
                    if end <= start :
                        print("No gradients to compute for this sample")
                        continue
                    else:
                        tmp_outputs = outputs[start:end]
                        returned_outputs[i] = tmp_outputs.clone().detach().mean().cpu().type(torch.float32).item()
            np.save(f"{self.gradients_dir}/{task_name}/train_batch_{batch_idx}_outputs.npy", returned_outputs)
            forward_output['loss'].detach(); logits.detach()
            forward_output['logits'].detach()
            return {} 

        
        ''' Generate the labels '''
        gold_answers = batch["labels"].clone()
        gold_answers[gold_answers == -100] = self.tokenizer.pad_token_id
        output_len = (gold_answers != self.tokenizer.pad_token_id).sum(dim=1).max().item()
        
        generate_output = self.generate_output
        if hasattr(self.model, "only_train_graph") and self.model.only_train_graph:
                generate_output = False
        if generate_output:
            # Remove labels in inputs_ids
            is_label_mask = batch["labels"] != -100
            batch["input_ids"][is_label_mask] = self.tokenizer.pad_token_id
            is_input_mask = batch["input_ids"] != self.tokenizer.pad_token_id
            batch["attention_mask"][is_label_mask] = 0

            if "graph_data" in batch:
                # convert to left padding
                new_input_ids = []
                for tmp_input_ids in batch["input_ids"]:
                    tmp_input_ids = tmp_input_ids[tmp_input_ids != self.tokenizer.pad_token_id]
                    new_input_ids.append(tmp_input_ids)
                new_input_ids = torch.stack(new_input_ids)
                inputs = self.tokenizer.batch_decode(new_input_ids, skip_special_tokens=False)
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                self.tokenizer.padding_side = 'right'
                inputs = self.transfer_batch_to_device(inputs, self.device, batch_idx)

                output = self.model.generate(**inputs, graph_data=batch["graph_data"], max_new_tokens=self.max_output_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id).detach()
            else:
                # convert to left padding
                inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                self.tokenizer.padding_side = 'right'
                inputs = self.transfer_batch_to_device(inputs, self.device, batch_idx)
                output = self.model.generate(**inputs, max_new_tokens=self.max_output_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            do_sample=True, temperature=1.0).detach()
            input_len = inputs["input_ids"].shape[1]
            output[:, :input_len] = self.tokenizer.pad_token_id
            if not self.evaluate_cot:
                output[:, input_len+output_len:] = self.tokenizer.pad_token_id


        output_dict = {
            "task_name": task_name,
            "loss": forward_output['loss'],
            "answers": gold_answers,
            "generates": output,
        }
        if hasattr(self.model, "only_train_graph") and self.model.only_train_graph:
            output_dict.update({"graph_accuracy": forward_output.graph_accuracy})
        if "weights" in batch:
            output_dict.update({"weights": batch["weights"]})
        return output_dict
    
    def on_validation_epoch_end(self) -> None:
        """
        Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
        log validation metrics.

        Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
        """
        summary = {f"{task_name}_loss": 0 for task_name in self.task_names}
        summary.update({f"{task_name}_accuracy_score": 0 for task_name in self.task_names})
        summary.update({f"{task_name}_accuracy": 0 for task_name in self.task_names})
        summary.update({f"{task_name}_edit_distance": 0 for task_name in self.task_names})
        if self.eval_math:
            summary.update({f"{task_name}_reasoning_accuracy": 0 for task_name in self.task_names})

        # average loss
        outputs = self.validation_step_outputs
        losses = [output["loss"] for output in outputs]
        losses = torch.stack(losses)
        losses = losses[torch.isnan(losses) == False]
        summary.update({"loss": losses.mean().item()})

        generate_output = self.generate_output
        if hasattr(self.model, "only_train_graph") and self.model.only_train_graph:
            generate_output = False

            task_counts = {task_name: 0 for task_name in self.task_names}
            for step, batch in enumerate(outputs):
                task_name = batch["task_name"]
                summary[f"{task_name}_loss"] += batch["loss"].item()*len(batch["answers"]) if torch.isnan(batch["loss"]) == False else 0
                summary[f"{task_name}_accuracy"] += batch["graph_accuracy"]*len(batch["answers"])
                task_counts[task_name] += len(batch["answers"])
            
            for task_name in self.task_names:
                if task_counts[task_name] > 0:
                    summary[f"{task_name}_loss"] /= task_counts[task_name]
                    summary[f"{task_name}_accuracy"] /= task_counts[task_name]

        task_counts = {task_name: 0 for task_name in self.task_names}
        label_counts = {task_name: 0 for task_name in self.task_names}
        # Collect step losses: {task_name: {step_idx: (sum, count)}}
        step_loss_dict = {task_name: defaultdict(lambda: [0.0, 0]) for task_name in self.task_names}
        # Collect final losses: {task_name: (sum, count)}
        final_loss_dict = {task_name: [0.0, 0] for task_name in self.task_names}
        for step, batch in enumerate(outputs):
            task_name = batch["task_name"]
            if len(batch["answers"]) == 0:
                continue
            summary[f"{task_name}_loss"] += batch["loss"].item()*len(batch["answers"]) if torch.isnan(batch["loss"]) == False else 0
            
            # Collect step losses
            if "step_losses" in batch and batch["step_losses"] is not None:
                for sample_step_losses in batch["step_losses"]:
                    # sample_step_losses is List[float], one loss per step
                    for step_idx, step_loss in enumerate(sample_step_losses):
                        step_loss_dict[task_name][step_idx][0] += step_loss
                        step_loss_dict[task_name][step_idx][1] += 1
            
            # Collect final losses
            if "final_losses" in batch and batch["final_losses"] is not None:
                for final_loss in batch["final_losses"]:
                    if final_loss is not None:
                        final_loss_dict[task_name][0] += final_loss
                        final_loss_dict[task_name][1] += 1
            # accuracy score is the token-level accuracy conditioned on the correct input (even for the intermediate steps)
            summary[f"{task_name}_accuracy_score"] += accuracy_score(batch["label_ids"], batch["pred_ids"])*len(batch["label_ids"])*100
            if generate_output:
                pred_answers = self.tokenizer.batch_decode(batch['generates'], skip_special_tokens=True)
                gold_answers = self.tokenizer.batch_decode(batch['answers'], skip_special_tokens=True)
                logging.info(f"DEBUG: gold_answers: {gold_answers}, pred_answers: {pred_answers}")
                if step == 0:
                    print("gold_answers", gold_answers[:4])
                    print("pred_answers", pred_answers[:4])
                if self.evaluate_cot:
                    # remove the steps and only keeps the answers
                    for i, answer in enumerate(pred_answers):
                        if "###" in answer:
                            answer = answer[answer.index("###")+3:]
                            pred_answers[i] = answer
                        else:
                            pred_answers[i] = answer
                    for i, answer in enumerate(gold_answers):
                        if "###" in answer:
                            answer = answer[answer.index("###")+3:]
                            gold_answers[i] = answer
                        else:
                            gold_answers[i] = answer
                if self.eval_clrs:
                    # Parse steps and compute step-by-step accuracy
                    pred_steps_list = []
                    gold_steps_list = []
                    pred_final_list = []
                    gold_final_list = []
                    
                    # Debug: check if | exists in the answers
                    if step == 0:
                        has_pipe_pred = sum(1 for a in pred_answers[:4] if "|" in a)
                        has_pipe_gold = sum(1 for a in gold_answers[:4] if "|" in a)
                        print(f"DEBUG: First 4 pred_answers with |: {has_pipe_pred}/4")
                        print(f"DEBUG: First 4 gold_answers with |: {has_pipe_gold}/4")
                        if has_pipe_gold > 0:
                            print(f"DEBUG: Example gold with |: {gold_answers[0][:200] if len(gold_answers[0]) > 200 else gold_answers[0]}")
                    
                    for i, (pred, gold) in enumerate(zip(pred_answers, gold_answers)):
                        # Parse prediction
                        logging.info(f"DEBUG: pred: {pred}, gold: {gold}")
                        if "|" in pred:
                            pred_steps_str = pred[:pred.index("|")].strip()
                            pred_final = pred[pred.index("|")+1:].strip()
                        else:
                            # If model didn't generate |, treat entire string as steps, last one is final
                            # This is a fallback for when model output doesn't match expected format
                            pred_steps_str = pred.strip()
                            pred_final = ""
                            if step == 0 and i < 2:
                                print(f"WARNING: pred_answers[{i}] has no | separator (first 100 chars): {pred[:100]}")
                        
                        # Parse gold - should always have | according to user
                        if "|" in gold:
                            gold_steps_str = gold[:gold.index("|")].strip()
                            gold_final = gold[gold.index("|")+1:].strip()
                        else:
                            # This shouldn't happen if data format is correct
                            gold_steps_str = gold.strip()
                            gold_final = ""
                            if step == 0 and i < 2:
                                print(f"ERROR: gold_answers[{i}] has no | separator (first 100 chars): {gold[:100]}")
                        
                        # Split steps by comma
                        if pred_steps_str:
                            pred_steps = [s.strip() for s in pred_steps_str.split(",") if s.strip()]
                        else:
                            pred_steps = []
                        
                        if gold_steps_str:
                            gold_steps = [s.strip() for s in gold_steps_str.split(",") if s.strip()]
                        else:
                            gold_steps = []
                        
                        # If no final answer separated by |, use last step as final
                        # We remove it from intermediate steps to avoid double counting
                        if not pred_final and pred_steps:
                            pred_final = pred_steps[-1]
                            pred_steps = pred_steps[:-1]  # Remove last step from intermediate steps
                        
                        if not gold_final and gold_steps:
                            gold_final = gold_steps[-1]
                            gold_steps = gold_steps[:-1]  # Remove last step from intermediate steps
                        
                        pred_steps_list.append(pred_steps)
                        gold_steps_list.append(gold_steps)
                        pred_final_list.append(pred_final)
                        gold_final_list.append(gold_final)

                        # logging.info(f"DEBUG: pred_steps: {pred_steps}, gold_steps: {gold_steps}, pred_final: {pred_final}, gold_final: {gold_final}")

                    # Compute step-by-step accuracy
                    max_steps = max(len(ps) for ps in pred_steps_list + gold_steps_list) if pred_steps_list or gold_steps_list else 0

                    # Compute accuracy for intermediate steps and final step
                    # If eval_step_num is 0, only compute final step
                    # If eval_step_num is N, compute first N steps + final step
                    if self.eval_step_num > 0:
                        steps_to_compute = list(range(min(self.eval_step_num, max_steps))) + ['final']
                    else:
                        steps_to_compute = ['final']
                    
                    for step_idx in steps_to_compute:
                        if step_idx == 'final':
                            # Compute final step accuracy
                            step_preds = pred_final_list
                            step_golds = [[g] for g in gold_final_list]
                            step_count = len(step_preds)
                        else:
                            # Compute accuracy for step step_idx
                            step_preds = []
                            step_golds = []
                            for ps, gs in zip(pred_steps_list, gold_steps_list):
                                if step_idx < len(ps) and step_idx < len(gs):
                                    step_preds.append(ps[step_idx])
                                    step_golds.append([gs[step_idx]])
                                elif step_idx < len(gs):
                                    # Prediction missing this step
                                    step_preds.append("")
                                    step_golds.append([gs[step_idx]])
                                elif step_idx < len(ps):
                                    # Gold missing this step
                                    step_preds.append(ps[step_idx])
                                    step_golds.append([""])
                                else:
                                    # Both missing, skip
                                    continue
                            
                            step_count = len(step_preds)
                            if step_count == 0:
                                continue
                        
                        # Compute accuracy for this step
                        step_metrics = compute_accuracy(step_preds, step_golds, indices=None)
                        step_key = f"{task_name}_step_{step_idx}_accuracy"
                        step_count_key = f"{task_name}_step_{step_idx}_count"
                        
                        if step_key not in summary:
                            summary[step_key] = 0
                        if step_count_key not in summary:
                            summary[step_count_key] = 0
                        
                        summary[step_key] += step_metrics["accuracy"] * step_count
                        summary[step_count_key] += step_count
                    
                    # For backward compatibility, also compute final accuracy as before
                    # Use pred_final_list and gold_final_list to ensure consistency with step_final_accuracy
                    # This ensures dfs_accuracy and dfs_step_final_accuracy use the same data
                    pred_answers_for_accuracy = pred_final_list.copy()
                    gold_answers_for_accuracy = [[g] for g in gold_final_list]
                    
                    # Also update pred_answers and gold_answers for edit_distance calculation
                    for i, answer in enumerate(pred_answers):
                        if "|" in answer:
                            answer = answer[answer.index("|")+1:]
                            pred_answers[i] = answer
                        else:
                            # Use the final answer we extracted
                            if i < len(pred_final_list):
                                pred_answers[i] = pred_final_list[i]
                            else:
                                pred_answers[i] = answer
                    for i, answer in enumerate(gold_answers):
                        if "|" in answer:
                            answer = answer[answer.index("|")+1:]
                            gold_answers[i] = answer
                        else:
                            # Use the final answer we extracted
                            if i < len(gold_final_list):
                                gold_answers[i] = gold_final_list[i]
                            else:
                                gold_answers[i] = answer
                    
                    # Compute accuracy using the same final answers as step_final_accuracy
                    metrics = compute_accuracy(pred_answers_for_accuracy, gold_answers_for_accuracy, indices=batch.get("indexes", None))
                    summary[f"{task_name}_accuracy"] += metrics["accuracy"]*len(batch["answers"])
                    
                    # Also compute edit_distance using the same final answers
                    summary[f"{task_name}_edit_distance"] += metrics["edit_distance"]*len(batch["answers"])
                else:
                    # Original behavior for non-CLRS datasets
                    gold_answers = [[answer] for answer in gold_answers]
                    metrics = compute_accuracy(pred_answers, gold_answers, indices=batch.get("indexes", None))
                    summary[f"{task_name}_accuracy"] += metrics["accuracy"]*len(batch["answers"])
                    summary[f"{task_name}_edit_distance"] += metrics["edit_distance"]*len(batch["answers"])
            
                if self.eval_math:
                    only_answer = self.tokenizer.batch_decode(batch['only_answer'], skip_special_tokens=True)
                    if step == 0:
                        print("only_answer", only_answer[:4])
                    correct = 0; count = 0; eval_func = eval_results_math if task_name == "math" else eval_results_gsm8k
                    for pred, gold in zip(pred_answers, only_answer):
                        if eval_func(pred, gold[0]):
                            correct += 1
                        count += 1
                    summary[f"{task_name}_reasoning_accuracy"] += correct * 100

            task_counts[task_name] += len(batch["answers"])
            label_counts[task_name] += batch["label_ids"].shape[0]
        
        for task_name in self.task_names:
            if task_counts[task_name] > 0:
                summary[f"{task_name}_loss"] /= task_counts[task_name]
                summary[f"{task_name}_accuracy_score"] = (summary[f"{task_name}_accuracy_score"]/label_counts[task_name]) if label_counts[task_name] > 0 else 0
                
                # Average step losses
                if task_name in step_loss_dict:
                    for step_idx in sorted(step_loss_dict[task_name].keys()):
                        step_loss_sum, step_loss_count = step_loss_dict[task_name][step_idx]
                        if step_loss_count > 0:
                            step_loss_avg = step_loss_sum / step_loss_count
                            summary[f"{task_name}_step_{step_idx}_loss"] = step_loss_avg
                
                # Average final loss
                if task_name in final_loss_dict:
                    final_loss_sum, final_loss_count = final_loss_dict[task_name]
                    if final_loss_count > 0:
                        final_loss_avg = final_loss_sum / final_loss_count
                        summary[f"{task_name}_final_loss"] = final_loss_avg

                if generate_output:
                    summary[f"{task_name}_accuracy"] /= task_counts[task_name]
                    summary[f"{task_name}_edit_distance"] /= task_counts[task_name]
                    if self.eval_math:
                        summary[f"{task_name}_reasoning_accuracy"] /= task_counts[task_name]
                    
                    # Average step accuracies for CLRS tasks
                    if self.eval_clrs:
                        step_keys = [k for k in summary.keys() if k.startswith(f"{task_name}_step_") and k.endswith("_accuracy")]
                        for step_key in step_keys:
                            # Get the corresponding count key
                            step_count_key = step_key.replace("_accuracy", "_count")
                            if step_count_key in summary and summary[step_count_key] > 0:
                                summary[step_key] /= summary[step_count_key]
                            else:
                                # Fallback to task_counts if count not found
                                summary[step_key] /= task_counts[task_name]

        # average accuracy and f1 score
        summary.update({"accuracy_score": np.mean([summary[f"{task_name}_accuracy_score"] for task_name in self.task_names])})
        if generate_output:
            summary.update({"accuracy": np.mean([summary[f"{task_name}_accuracy"] for task_name in self.task_names])})
            summary.update({"edit_distance": np.mean([summary[f"{task_name}_edit_distance"] for task_name in self.task_names])})
            if self.eval_math:
                summary.update({"reasoning_accuracy": np.mean([summary[f"{task_name}_reasoning_accuracy"] for task_name in self.task_names])})

        # Log metrics
        print(summary)
        if summary:
            for key, value in summary.items():
                if "accuracy" in key:
                    self.log(key, value, prog_bar=True, logger=True)
                else:
                    self.log(key, value, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()
        return summary

    def forward(self, batch, batch_idx):
        task_name = batch["task_name"]; batch = batch["data"]
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if "graph_data" in batch:
            kwargs["graph_data"] = batch["graph_data"]
            kwargs["original_input_ids"] = batch["original_input_ids"]
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        outputs = self.model(**kwargs)
        return outputs

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            import bitsandbytes as bnb
            optimizer_dict = {
                "adamw_8bit": bnb.optim.AdamW8bit,
                "paged_adamw_8bit": bnb.optim.PagedAdamW8bit,
                "paged_adamw_32bit": bnb.optim.PagedAdamW32bit,
            }

            optimizer = optimizer_dict[self.optimizer](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # force embedding layers to use 32 bit for numerical stability
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
        return optimizer