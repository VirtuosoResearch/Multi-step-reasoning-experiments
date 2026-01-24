import torch
import torch.nn.functional as F

from src.custom.multitask_model import MultitaskModel


class GradientNormReweightMultitaskModel(MultitaskModel):
    def training_step(self, batch, batch_idx):
        if self.use_sample_weights or self.fit_least_square:
            return super().training_step(batch, batch_idx)

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

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        labels = batch["labels"]

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

        trainable_params = self.get_trainable_parameters()
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
        grad_norms = torch.stack(grad_norms)

        norm_sum = grad_norms.sum()
        batch_size = grad_norms.numel()
        if norm_sum.item() == 0:
            weights = torch.ones_like(grad_norms)
        else:
            weights = grad_norms / (norm_sum + 1e-12) * batch_size

        loss = (per_sample_loss * weights.detach()).sum()
        if self.use_wandb:
            import wandb
            wandb.log({
                "train_loss": loss,
                "reweight/weight_sum": weights.sum(),
                "reweight/weight_mean": weights.mean(),
                "reweight/weight_max": weights.max(),
                "reweight/weight_min": weights.min(),
                "reweight/grad_norm_mean": grad_norms.mean(),
                "reweight/grad_norm_max": grad_norms.max(),
                "reweight/grad_norm_min": grad_norms.min(),
                "reweight/per_sample_loss_mean": per_sample_loss.mean(),
                "reweight/per_sample_loss_max": per_sample_loss.max(),
                "reweight/per_sample_loss_min": per_sample_loss.min(),
            })
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
