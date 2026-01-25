import torch
import torch.nn.functional as F

from src.custom.multitask_model import MultitaskModel


class GradientNormReweightMultitaskModel(MultitaskModel):
    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._initialize_sample_weights()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._update_sample_weights()

    def _initialize_sample_weights(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "task_to_train_datasets"):
            return

        task_to_weights = {}
        for task_name, train_dataset in datamodule.task_to_train_datasets.items():
            num_samples = len(train_dataset)
            weights = [1.0] * num_samples
            if "weights" in train_dataset.column_names:
                train_dataset = train_dataset.remove_columns("weights")
            datamodule.task_to_train_datasets[task_name] = train_dataset.add_column("weights", weights)
            task_to_weights[task_name] = weights

        datamodule.weights = [1.0] * sum(len(dataset) for dataset in datamodule.task_to_train_datasets.values())
        self._set_class_weights(task_to_weights)

    def _update_sample_weights(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "task_to_train_datasets"):
            return

        was_training = self.model.training
        self.model.eval()
        trainable_params = self.get_trainable_parameters()
        task_to_norms = {}
        with torch.set_grad_enabled(True):
            for task_name, train_dataset in datamodule.task_to_train_datasets.items():
                collator = datamodule.task_to_collators[task_name]
                batch_size = getattr(datamodule, "batch_size", 1)
                task_weights = []
                for start_idx in range(0, len(train_dataset), batch_size):
                    samples = [train_dataset[i] for i in range(start_idx, min(start_idx + batch_size, len(train_dataset)))]
                    batch = collator(samples)
                    batch = self.transfer_batch_to_device(batch, self.device, 0)
                    per_sample_loss = self._compute_per_sample_loss(batch)
                    grad_norms = self._compute_per_sample_grad_norms(per_sample_loss, trainable_params)
                    task_weights.append(grad_norms.detach().cpu())
                    self.model.zero_grad(set_to_none=True)
                if task_weights:
                    task_weights = torch.cat(task_weights)
                else:
                    task_weights = torch.zeros(0)

                task_to_norms[task_name] = task_weights

        all_norms = torch.cat(list(task_to_norms.values())) if task_to_norms else torch.zeros(0)
        norm_sum = all_norms.sum()
        if norm_sum.item() == 0:
            norm_sum = torch.tensor(0.0, device=all_norms.device)
        total_len = all_norms.numel()
        task_to_weights = {}
        for task_name, train_dataset in datamodule.task_to_train_datasets.items():
            task_norms = task_to_norms.get(task_name, torch.zeros(0, device=all_norms.device))
            if total_len == 0 or norm_sum.item() == 0:
                task_weights = torch.ones_like(task_norms)
            else:
                task_weights = task_norms / (norm_sum + 1e-12) * total_len
            if "weights" in train_dataset.column_names:
                train_dataset = train_dataset.remove_columns("weights")
            datamodule.task_to_train_datasets[task_name] = train_dataset.add_column(
                "weights",
                task_weights.tolist(),
            )
            task_to_weights[task_name] = task_weights.tolist()

        datamodule.weights = [w for weights in task_to_weights.values() for w in weights]
        self._set_class_weights(task_to_weights)
        if was_training:
            self.model.train()

    def _set_class_weights(self, task_to_weights):
        self.sample_weight_by_task = {}
        self.sample_index_by_task = {}
        self.sample_weight_list = []
        offset = 0
        for task_name, weights in task_to_weights.items():
            self.sample_weight_by_task[task_name] = weights
            self.sample_index_by_task[task_name] = list(range(offset, offset + len(weights)))
            self.sample_weight_list.extend(weights)
            offset += len(weights)

    def _compute_per_sample_loss(self, batch):
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
        return per_sample_loss

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

        if "weights" in batch:
            weights = batch["weights"].to(per_sample_loss.device)
        else:
            weights = torch.ones_like(per_sample_loss)
        loss = (per_sample_loss * weights.detach()).mean()
        if self.use_wandb:
            import wandb
            wandb.log({
                "train_loss": base_loss,
            })
        self.log("train_loss", base_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
