import torch
import torch.nn.functional as F

from src.custom.multitask_model import MultitaskModel


class GradientNormReweightMultitaskModel(MultitaskModel):

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if not hasattr(self, "reweight_eta"):
            self.reweight_eta = 0.1
        if not hasattr(self, "weight_clip_exp"):
            self.weight_clip_exp = 50.0
        self._initialize_sample_indices_and_weights()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._update_sample_weights()

    # -------------------------
    # Initialization utilities
    # -------------------------
    def _initialize_sample_indices_and_weights(self) -> None:
        """
        Initialize sample weights. The sample_idx should already be added in the datamodule.setup()
        method, so we just need to verify and initialize weights.
        """
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

    # -------------------------
    # Weight update procedure
    # -------------------------
    def _update_sample_weights(self) -> None:
        """
        Update sample weights at the end of each epoch.
        This function iterates through all original training data points (not dataloader batches),
        computes gradient norms for each, and updates the global weights list.
        """
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
                prev_tensor = torch.tensor(prev, dtype=norms_by_idx.dtype)

            scaled = torch.clamp(norms_by_idx * float(self.reweight_eta), max=float(self.weight_clip_exp))
            # raw = prev_tensor * torch.exp(-scaled)
            raw = torch.exp(scaled)
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
        print(f"[GradientReweight] Updated weights for {total_samples} total data points across all tasks")

        if was_training:
            self.model.train()

    # -------------------------
    # Storage helpers
    # -------------------------
    def _set_class_weights(self, task_to_weights):
        """
        Store the global weights list. Each task has a list of weights, one per original data point.
        The sample_idx in each batch corresponds to the index in this weights list (0 to N-1 per task).
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

    # -------------------------
    # Loss and grad utilities
    # -------------------------
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

    # -------------------------
    # Training step
    # -------------------------
    def training_step(self, batch, batch_idx):
        if self.use_sample_weights or self.fit_least_square:
            return super().training_step(batch, batch_idx)

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
        # The sample_idx in the batch corresponds to the index in the weights list
        weights_table = self.sample_weight_by_task.get(task_name, None)
        if weights_table is None:
            weights = torch.ones((idxs.shape[0],), device=self.device, dtype=torch.float32)
        else:
            # Use the sample_idx to index into the weights list for this task
            weights_full = torch.tensor(weights_table, device=self.device, dtype=torch.float32)
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
