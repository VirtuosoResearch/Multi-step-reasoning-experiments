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

from src.custom.multitask_model import MultitaskModel
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


class SelfTrainingMultitaskModel(MultitaskModel):
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
        )
        self.use_self_training = use_self_training
        self.self_train_n_samples = self_train_n_samples
        self.self_train_temperature = self_train_temperature
        self._extra_train_pairs: List[Tuple[str, str]] = []
        self._extra_train_keys: Set[Tuple[str, str]] = set()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.use_self_training:
            logger.info("[SelfTraining] Enabled: add correct validation samples to train (dedup).")
            self._extra_train_pairs = []
            self._extra_train_keys = set()

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
            loss = self._compute_loss_on_accepted(chunk)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.model.eval()

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
        # Self-training: main loop trains on original data only. Extra training on
        # correct-validation samples runs in on_validation_epoch_end.
        return super().training_step(batch, batch_idx)
