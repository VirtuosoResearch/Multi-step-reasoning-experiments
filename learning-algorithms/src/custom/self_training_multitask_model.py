"""
Self-Training Multitask Model (validate → add correct val to train → train on extra).

Each epoch:
1. Train on original training data.
2. Run validation.
3. Add validation samples the model got *correct* (pred final == gold final) to an
   extra training list, with deduplication (never add the same (input, gold) twice).
4. Run extra training steps on this accumulated extra list.
"""

import logging
import random
from typing import List, Tuple, Optional, Set

import torch
import torch.nn.functional as F

from src.custom.noise_injection_multitask_model import NoiseInjectionMultitaskModel
from src.utils.compute_metrics import normalize_answer

logger = logging.getLogger(__name__)


def _parse_final_answer(full_output: str) -> str:
    """Extract final answer from 'CoT | z' or plain 'z'."""
    s = full_output.strip()
    if "|" in s:
        return s.split("|")[-1].strip()
    return s


def _verify_with_true_label(pred_final: str, gold_final: str) -> bool:
    """Accept iff pred final answer matches gold (normalized)."""
    return normalize_answer(pred_final) == normalize_answer(gold_final)


class SelfTrainingMultitaskModel(NoiseInjectionMultitaskModel):
    """
    MultitaskModel with self-training: each epoch end, run validation, add correct
    validation (input, gold) to training data (dedup), then train on that extra set.

    Args (in addition to MultitaskModel):
        use_self_training: Enable self-training (default: False).
        self_train_n_samples: Unused; kept for CLI compat.
        self_train_temperature: Unused; kept for CLI compat.
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
        task_names=(),
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
        use_self_training=False,
        self_train_n_samples=4,
        self_train_temperature=1.0,
        # gradient reweighting
        use_reweight=False,
        # noise injection (forwarded to NoiseInjectionMultitaskModel)
        use_noise_injection=False,
        noise_std=1e-3,
    ):
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
            use_noise_injection=use_noise_injection,
            noise_std=noise_std,
        )
        self.use_self_training = use_self_training
        self.use_reweight = use_reweight
        self.self_train_n_samples = self_train_n_samples
        self.self_train_temperature = self_train_temperature
        self._extra_train_pairs: List[Tuple[str, str]] = []
        self._extra_train_keys: Set[Tuple[str, str]] = set()

        # Reweight hyper-parameters (can be overridden from outside)
        if not hasattr(self, "reweight_eta"):
            self.reweight_eta = 0.1
        if not hasattr(self, "weight_clip_exp"):
            self.weight_clip_exp = 50.0

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.use_self_training:
            logger.info("[SelfTraining] Enabled: add correct validation samples to train (dedup).")
            self._extra_train_pairs = []
            self._extra_train_keys = set()

        # Initialize sample indices / weights for gradient-based reweighting
        if self.use_self_training and self.use_reweight:
            self._initialize_sample_indices_and_weights()

    def on_train_epoch_end(self) -> None:
        # Keep any parent hooks (e.g., logging) intact
        super().on_train_epoch_end()
        # Update per-sample weights using gradient norms (only when enabled)
        if self.use_self_training and self.use_reweight:
            self._update_sample_weights()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if not self.use_self_training or not self._extra_train_pairs:
            return
        dm = getattr(self.trainer, "datamodule", None)
        bs = int(getattr(dm, "batch_size", 8))
        pairs = list(self._extra_train_pairs)
        random.shuffle(pairs)
        opt = self.trainer.optimizers[0]
        opt.zero_grad(set_to_none=True)
        self.model.train()
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            # Optional noise injection also applies to these extra self-training steps.
            if getattr(self, "use_noise_injection", False):
                # Reuse NoiseInjectionMultitaskModel's mechanism
                self._apply_noise_to_params()
            loss = self._compute_loss_on_accepted(chunk)
            if loss is None:
                if getattr(self, "use_noise_injection", False):
                    self._remove_noise_from_params()
                continue
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if getattr(self, "use_noise_injection", False):
                self._remove_noise_from_params()
        self.model.eval()

    # ------------------------------------------------------------------
    # Gradient-norm based reweighting (ported from GradientNormReweightMultitaskModel)
    # ------------------------------------------------------------------
    def _initialize_sample_indices_and_weights(self) -> None:
        """
        Initialize sample weights. The sample_idx should already be added in the
        datamodule.setup() method, so we just need to verify and initialize weights.
        """
        if not (self.use_self_training and self.use_reweight):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "task_to_train_datasets"):
            return

        task_to_weights = {}
        for task_name, train_dataset in datamodule.task_to_train_datasets.items():
            # Verify that sample_idx exists (should have been added in datamodule.setup())
            if "sample_idx" not in train_dataset.column_names:
                raise ValueError(
                    f"sample_idx not found in dataset for task '{task_name}'. "
                    "It should be added in the datamodule.setup() method after downsample."
                )

            # Initialize weights for each original data point (one weight per data point)
            num_samples = len(train_dataset)
            task_to_weights[task_name] = [1.0] * num_samples

        self._set_class_weights(task_to_weights)

    def _update_sample_weights(self) -> None:
        """
        Update sample weights at the end of each epoch.
        This iterates through all original training data points (not dataloader
        batches), computes gradient norms for each, and updates the global weights.
        """
        if not (self.use_self_training and self.use_reweight):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "task_to_train_datasets"):
            return

        was_training = self.model.training
        self.model.eval()

        trainable_params = self.get_trainable_parameters()
        prev_weights_by_task = getattr(self, "sample_weight_by_task", {})

        task_to_norms_by_idx = {}

        with torch.set_grad_enabled(True):
            for task_name, train_dataset in datamodule.task_to_train_datasets.items():
                collator = datamodule.task_to_collators[task_name]
                batch_size = int(getattr(datamodule, "batch_size", 1))

                # Initialize norms for all original data points in this task
                num_samples = len(train_dataset)
                norms_by_idx = torch.zeros(num_samples, dtype=torch.float32)

                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    samples = [train_dataset[i] for i in range(start_idx, end_idx)]

                    batch = collator(samples)

                    idxs = batch["sample_idx"]
                    if not isinstance(idxs, torch.Tensor):
                        idxs = torch.tensor(idxs, dtype=torch.long)
                    idxs_cpu = idxs.detach().cpu().long()

                    batch = self.transfer_batch_to_device(batch, self.device, 0)

                    per_sample_loss = self._compute_per_sample_loss(batch)
                    grad_norms = self._compute_per_sample_grad_norms(
                        per_sample_loss, trainable_params
                    ).detach().cpu().float()

                    norms_by_idx[idxs_cpu] = grad_norms
                    self.model.zero_grad(set_to_none=True)

                task_to_norms_by_idx[task_name] = norms_by_idx

        # Update weights based on gradient norms
        task_to_raw = {}
        for task_name, norms_by_idx in task_to_norms_by_idx.items():
            prev = prev_weights_by_task.get(task_name, None)
            if prev is None or len(prev) != norms_by_idx.numel():
                prev_tensor = torch.ones_like(norms_by_idx)
            else:
                prev_tensor = torch.tensor(prev, dtype=torch.float32)

            scaled = torch.clamp(
                norms_by_idx * float(self.reweight_eta),
                max=float(self.weight_clip_exp),
            )
            # raw = prev_tensor * torch.exp(-scaled)
            raw = torch.exp(-scaled)
            task_to_raw[task_name] = raw

        all_raw = torch.cat(list(task_to_raw.values())) if task_to_raw else torch.zeros(0)
        norm_sum = all_raw.sum()
        total_len = all_raw.numel()
        task_to_weights = {}
        for task_name, raw in task_to_raw.items():
            if total_len == 0 or float(norm_sum) == 0.0:
                w = torch.ones_like(raw)
            else:
                w = raw / (norm_sum + 1e-12) * total_len
            task_to_weights[task_name] = w.tolist()

        print("================================================")
        print("task_to_weights: ", task_to_weights)
        print("================================================")

        # Update the global weights list
        self._set_class_weights(task_to_weights)

        # Print summary of updated weights
        total_samples = sum(len(weights) for weights in task_to_weights.values())
        print(
            f"[GradientReweight] Updated weights for {total_samples} total data points across all tasks"
        )

        if was_training:
            self.model.train()

    def _set_class_weights(self, task_to_weights):
        """
        Store the global weights list. Each task has a list of weights, one per
        original data point. The sample_idx in each batch corresponds to the index
        in this weights list (0 to N-1 per task).
        """
        self.sample_weight_by_task = {}
        self.sample_index_by_task = {}
        self.sample_weight_list = []
        for task_name, weights in task_to_weights.items():
            # Store weights for this task (one weight per original data point)
            self.sample_weight_by_task[task_name] = weights
            # Store the sample_idx range for this task (0 to N-1, not global offset)
            # This matches the sample_idx in the dataset which starts from 0 for each task
            self.sample_index_by_task[task_name] = list(range(len(weights)))
            # Maintain a global weights list across all tasks (for potential future use)
            self.sample_weight_list.extend(weights)

    def _decode_input_and_gold(self, batch: dict) -> List[Tuple[str, str, str]]:
        """Decode batch to (input_str, gold_full_output, gold_final) per sample."""
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        B = batch["input_ids"].size(0)
        out = []
        for b in range(B):
            labels = batch["labels"][b]
            input_mask = labels == -100
            output_mask = (labels != -100) & (labels != pad_id)
            input_tok = batch["input_ids"][b][input_mask]
            output_tok = batch["labels"][b][output_mask]
            input_str = self.tokenizer.decode(input_tok, skip_special_tokens=True).strip()
            gold_full = self.tokenizer.decode(output_tok, skip_special_tokens=True).strip()
            gold_final = _parse_final_answer(gold_full)
            out.append((input_str, gold_full, gold_final))
        return out

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        if not self.use_self_training:
            return out
        data = batch["data"]
        if "graph_data" in data:
            return out
        if out.get("generates") is None:
            return out
        dec = self._decode_input_and_gold(data)
        pred_strs = self.tokenizer.batch_decode(out["generates"], skip_special_tokens=True)
        val_correct_info: List[Tuple[str, str]] = []
        for (inp, gold_full, gold_final), pred_str in zip(dec, pred_strs):
            pred_final = _parse_final_answer(pred_str.strip())
            if _verify_with_true_label(pred_final, gold_final):
                val_correct_info.append((inp, gold_full))
        out["val_correct_info"] = val_correct_info
        return out

    def on_validation_epoch_end(self) -> None:
        outputs = list(getattr(self, "validation_step_outputs", []))
        if self.use_self_training and outputs:
            for batch in outputs:
                for t in batch.get("val_correct_info", []):
                    if t not in self._extra_train_keys:
                        self._extra_train_keys.add(t)
                        self._extra_train_pairs.append(t)
            logger.info(
                "[SelfTraining] Extra train pairs: %d (dedup).",
                len(self._extra_train_pairs),
            )
        super().on_validation_epoch_end()

    def _compute_loss_on_accepted(
        self, accepted: List[Tuple[str, str]]
    ) -> Optional[torch.Tensor]:
        """M-step: Tokenize (input + ' ' + output), mask input in labels, forward, return loss."""
        if not accepted:
            return None
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_total = self.max_length + self.max_output_length
        all_input_ids = []
        all_labels = []
        all_attention = []
        for inp, out in accepted:
            full = inp + " " + out
            tok = self.tokenizer(
                full,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_total,
            )
            ids = tok["input_ids"][0]
            prefix = inp + " "
            inp_tok = self.tokenizer(
                prefix,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length + 1,
            )
            src_len = inp_tok["input_ids"].size(1)
            lbl = ids.clone()
            lbl[:src_len] = -100
            all_input_ids.append(ids)
            all_labels.append(lbl)
            all_attention.append(torch.ones_like(ids, dtype=torch.long))

        max_len = max(x.size(0) for x in all_input_ids)
        device = next(self.model.parameters()).device
        input_ids = torch.full(
            (len(accepted), max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(len(accepted), max_len, dtype=torch.long, device=device)
        labels = torch.full(
            (len(accepted), max_len),
            -100,
            dtype=torch.long,
            device=device,
        )
        for i, (ii, am, ll) in enumerate(zip(all_input_ids, all_attention, all_labels)):
            L = ii.size(0)
            input_ids[i, :L] = ii.to(device)
            attention_mask[i, :L] = am.to(device)
            labels[i, :L] = ll.to(device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        out = self.model(**kwargs)
        return out["loss"]

    def training_step(self, batch, batch_idx):
        """
        Training step with both:
        - Noise injection (from NoiseInjectionMultitaskModel)
        - Gradient-based sample reweighting (like GradientNormReweightMultitaskModel)

        Self-training still only uses the original training data here; extra
        training on accepted validation samples is handled in
        on_train_epoch_start().
        """
        if not (self.use_self_training and self.use_reweight):
            return super().training_step(batch, batch_idx)

        if self.use_sample_weights or self.fit_least_square:
            return super().training_step(batch, batch_idx)

        if getattr(self, "use_noise_injection", False) and self.training:
            self._apply_noise_to_params()

        task_name = batch["task_name"]
        batch = batch["data"]

        if "sample_idx" not in batch:
            raise KeyError(
                "Training batch is missing `sample_idx`. "
                "Your collator must preserve `sample_idx` from samples."
            )

        idxs = batch["sample_idx"]
        if not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.long)
        idxs = idxs.to(self.device)

        # Get weights from the global weights list for this task
        weights_table = self.sample_weight_by_task.get(task_name, None)
        if weights_table is None:
            weights = torch.ones((idxs.shape[0],), device=self.device, dtype=torch.float32)
        else:
            weights_full = torch.tensor(
                weights_table, device=self.device, dtype=torch.float32
            )
            weights = weights_full[idxs]

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

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        labels = batch["labels"]
        base_loss = outputs["loss"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, self.model.config.vocab_size)
        flat_labels = shift_labels.view(-1)

        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        )
        token_losses = token_losses.view(shift_labels.size(0), -1)

        valid_mask = shift_labels != -100
        valid_counts = valid_mask.sum(dim=1).clamp_min(1)

        per_sample_loss = (token_losses * valid_mask).sum(dim=1) / valid_counts

        loss = (per_sample_loss * weights.detach()).mean()

        if getattr(self, "use_wandb", False):
            import wandb

            wandb.log({"train_loss": base_loss})

        self.log("train_loss", base_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def _compute_per_sample_grad_norms(per_sample_loss, trainable_params):
        grad_norms = []
        for i in range(per_sample_loss.size(0)):
            grads = torch.autograd.grad(
                per_sample_loss[i],
                trainable_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            norm_sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
            grad_norms.append(torch.sqrt(norm_sq))
        return torch.stack(grad_norms)

