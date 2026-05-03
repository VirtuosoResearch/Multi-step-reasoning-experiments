import math
from typing import Dict, List, Tuple

import torch

from src.custom.multitask_model import (
    MultitaskModel,
    print_gpu_utilization,
)


def parse_coconut_answer(answer: str) -> Tuple[List[str], str]:
    if "|" not in answer:
        return [], answer.strip()

    steps_text, final_answer = answer.split("|", 1)
    steps = [step.strip() for step in steps_text.split(",") if step.strip()]
    return steps, final_answer.strip()


def coconut_schedule_fraction(global_step: int, steps_to_full_latent: int) -> float:
    steps_to_full_latent = max(1, int(steps_to_full_latent))
    if steps_to_full_latent <= 1:
        return 1.0
    return min(global_step / float(steps_to_full_latent - 1), 1.0)


class CoconutMultitaskModel(MultitaskModel):
    def __init__(
        self,
        *args,
        coconut_steps_to_full_latent=1,
        coconut_latents_per_step=1,
        coconut_eval_latent_thoughts=-1,
        coconut_bot_token="<bot>",
        coconut_eot_token="<eot>",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.coconut_steps_to_full_latent = max(1, int(coconut_steps_to_full_latent))
        self.coconut_latents_per_step = max(0, int(coconut_latents_per_step))
        self.coconut_eval_latent_thoughts = int(coconut_eval_latent_thoughts)
        self.coconut_bot_token = coconut_bot_token
        self.coconut_eot_token = coconut_eot_token
        self.coconut_bot_token_id = self.tokenizer.convert_tokens_to_ids(coconut_bot_token)
        self.coconut_eot_token_id = self.tokenizer.convert_tokens_to_ids(coconut_eot_token)
        if self.coconut_bot_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Coconut bot token {coconut_bot_token!r} is not registered in tokenizer")
        if self.coconut_eot_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Coconut eot token {coconut_eot_token!r} is not registered in tokenizer")

    @staticmethod
    def build_coconut_target(answer: str, reduction_fraction: float, latents_per_step: int) -> Tuple[str, int, int]:
        steps, final_answer = parse_coconut_answer(answer)
        if len(steps) == 0:
            return final_answer, 0, 0

        removed_steps = int(math.floor(reduction_fraction * len(steps)))
        removed_steps = min(max(removed_steps, 0), len(steps))
        remaining_steps = steps[removed_steps:]
        latent_count = removed_steps * max(0, int(latents_per_step))

        if len(remaining_steps) == 0:
            return final_answer, removed_steps, latent_count

        return ", ".join(remaining_steps) + " | " + final_answer, removed_steps, latent_count

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        return " ".join(text.split())

    def _tokenize_1d(self, text: str, max_length: int = None, device=None) -> torch.Tensor:
        tokenized = self.tokenizer(text)["input_ids"]
        if max_length is not None:
            tokenized = tokenized[:max_length]
        ids = torch.tensor(tokenized, dtype=torch.long)
        if device is not None:
            ids = ids.to(device)
        return ids

    def _embed_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(token_ids.unsqueeze(0))

    def _forward_hidden(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return output.hidden_states[-1][:, -1:, :]

    def _build_latent_prefix_embeds(self, source_text: str, latent_count: int, device) -> torch.Tensor:
        source_ids = self._tokenize_1d(self._normalize_text(source_text), self.max_length, device=device)
        bot_id = torch.tensor([self.coconut_bot_token_id], dtype=torch.long, device=device)
        prefix_ids = torch.cat([source_ids, bot_id], dim=0)
        inputs_embeds = self._embed_ids(prefix_ids)

        for _ in range(latent_count):
            next_latent = self._forward_hidden(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, next_latent], dim=1)
        return inputs_embeds

    def _plain_forward_for_target(self, source_text: str, target_text: str, device):
        source_text = self._normalize_text(source_text)
        target_text = self._normalize_text(target_text)
        source_ids = self._tokenize_1d(source_text, self.max_length, device=device)
        source_text = self.tokenizer.decode(source_ids, skip_special_tokens=True)
        source_ids = self._tokenize_1d(source_text, self.max_length, device=device)
        full_ids = self._tokenize_1d(
            source_text + " " + target_text,
            self.max_length + self.max_output_length,
            device=device,
        )
        labels = full_ids.unsqueeze(0).clone()
        labels[:, :source_ids.numel()] = -100
        attention_mask = torch.ones_like(full_ids, dtype=torch.long, device=device).unsqueeze(0)

        output = self.model(
            input_ids=full_ids.unsqueeze(0),
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return output, labels, full_ids[source_ids.numel():]

    def _coconut_forward_for_target(self, source_text: str, target_text: str, latent_count: int, device, use_latent_markers: bool = True):
        if not use_latent_markers:
            return self._plain_forward_for_target(source_text, target_text, device)

        prefix_embeds = self._build_latent_prefix_embeds(source_text, latent_count, device)
        eot_id = torch.tensor([self.coconut_eot_token_id], dtype=torch.long, device=device)
        target_ids = self._tokenize_1d(self._normalize_text(target_text), self.max_output_length, device=device)

        eot_embeds = self._embed_ids(eot_id)
        target_embeds = self._embed_ids(target_ids)
        inputs_embeds = torch.cat([prefix_embeds, eot_embeds, target_embeds], dim=1)

        labels = torch.full(
            (1, inputs_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=device,
        )
        labels[0, -target_ids.numel():] = target_ids
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return output, labels, target_ids

    def _coconut_sample_loss(self, source_text: str, answer_text: str, reduction_fraction: float, device) -> Tuple[torch.Tensor, int, int]:
        target_text, removed_steps, latent_count = self.build_coconut_target(
            answer_text,
            reduction_fraction,
            self.coconut_latents_per_step,
        )

        output, _, _ = self._coconut_forward_for_target(
            source_text,
            target_text,
            latent_count,
            device,
            use_latent_markers=(latent_count > 0),
        )
        return output["loss"], removed_steps, latent_count

    def training_step(self, batch, batch_idx):
        task_name = batch["task_name"]
        batch = batch["data"]
        if "raw_inputs" not in batch or "raw_answers" not in batch:
            raise KeyError("Coconut training requires `raw_inputs` and `raw_answers` from the text collator.")

        reduction_fraction = coconut_schedule_fraction(
            int(self.trainer.global_step),
            self.coconut_steps_to_full_latent,
        )

        losses = []
        removed_steps = []
        latent_counts = []
        device = batch["input_ids"].device
        for source_text, answer_text in zip(batch["raw_inputs"], batch["raw_answers"]):
            loss, sample_removed_steps, sample_latent_count = self._coconut_sample_loss(
                source_text,
                answer_text,
                reduction_fraction,
                device,
            )
            losses.append(loss)
            removed_steps.append(sample_removed_steps)
            latent_counts.append(sample_latent_count)

        loss = torch.stack(losses).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("coconut_reduction_fraction", reduction_fraction, on_step=True, on_epoch=False, prog_bar=False)
        self.log("coconut_removed_steps", float(sum(removed_steps)) / max(1, len(removed_steps)), on_step=True, on_epoch=False, prog_bar=False)
        self.log("coconut_latent_thoughts", float(sum(latent_counts)) / max(1, len(latent_counts)), on_step=True, on_epoch=False, prog_bar=False)
        return loss

    @torch.no_grad()
    def _coconut_generate_one(self, source_text: str, latent_count: int, device) -> torch.Tensor:
        prefix_embeds = self._build_latent_prefix_embeds(source_text, latent_count, device)
        eot_id = torch.tensor([self.coconut_eot_token_id], dtype=torch.long, device=device)
        eot_embeds = self._embed_ids(eot_id)
        inputs_embeds = torch.cat([prefix_embeds, eot_embeds], dim=1)

        generated_ids = []
        for _ in range(self.max_output_length):
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            next_id = torch.argmax(output.logits[:, -1, :], dim=-1)
            token_id = int(next_id.item())
            generated_ids.append(token_id)
            if token_id == self.tokenizer.eos_token_id:
                break
            next_embed = self._embed_ids(next_id)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

        if len(generated_ids) == 0:
            generated_ids = [self.tokenizer.pad_token_id]
        return torch.tensor(generated_ids, dtype=torch.long, device=device)

    @torch.no_grad()
    def coconut_generate_batch(self, raw_inputs: List[str], device) -> torch.Tensor:
        if self.coconut_eval_latent_thoughts < 0:
            raise ValueError("--coconut_eval_latent_thoughts must be non-negative for Coconut generation")

        generated = [
            self._coconut_generate_one(source_text, self.coconut_eval_latent_thoughts, device)
            for source_text in raw_inputs
        ]
        max_len = max(gen.numel() for gen in generated)
        output = torch.full(
            (len(generated), max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, gen in enumerate(generated):
            output[i, :gen.numel()] = gen
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Dict[str, torch.Tensor]:
        task_name = batch["task_name"]
        batch = batch["data"]
        if "raw_inputs" not in batch or "raw_answers" not in batch:
            raise KeyError("Coconut validation requires `raw_inputs` and `raw_answers` from the text collator.")
        if self.coconut_eval_latent_thoughts < 0:
            raise ValueError("--coconut_eval_latent_thoughts must be non-negative for Coconut validation")

        device = batch["input_ids"].device
        losses = []
        pred_chunks = []
        label_chunks = []
        final_losses = []
        for source_text, answer_text in zip(batch["raw_inputs"], batch["raw_answers"]):
            _, final_answer = parse_coconut_answer(answer_text)
            forward_output, labels_for_loss, _ = self._coconut_forward_for_target(
                source_text,
                final_answer,
                self.coconut_eval_latent_thoughts,
                device,
            )
            losses.append(forward_output["loss"])
            final_losses.append(float(forward_output["loss"].detach().cpu()))

            is_label_mask = labels_for_loss[:, 1:].contiguous() != -100
            logits = forward_output["logits"][:, :-1].contiguous()
            pred_chunks.append(torch.argmax(logits[is_label_mask], dim=-1).detach().cpu())
            label_chunks.append(labels_for_loss[:, 1:][is_label_mask].detach().cpu())

        validation_loss = torch.stack(losses).mean()

        if batch_idx == 1:
            print_gpu_utilization()

        gold_answers = batch["labels"].clone()
        gold_answers[gold_answers == -100] = self.tokenizer.pad_token_id

        preds = torch.cat(pred_chunks).numpy() if len(pred_chunks) > 0 else torch.empty(0, dtype=torch.long).numpy()
        labels = torch.cat(label_chunks).numpy() if len(label_chunks) > 0 else torch.empty(0, dtype=torch.long).numpy()
        step_losses = None

        generate_output = self.generate_output
        if generate_output:
            output = self.coconut_generate_batch(batch["raw_inputs"], batch["input_ids"].device).detach()
        else:
            output = None

        output_dict = {
            "task_name": task_name,
            "split": "test" if dataloader_idx == 1 else "val",
            "loss": validation_loss,
            "answers": gold_answers,
            "generates": output,
            "pred_ids": preds,
            "label_ids": labels,
            "step_losses": step_losses,
            "final_losses": final_losses,
        }
        if "only_answer" in batch:
            output_dict.update({"only_answer": batch["only_answer"]})
        if "length" in batch:
            output_dict.update({"lengths": batch["length"].detach().cpu().tolist()})
        self.validation_step_outputs.append(output_dict)
        return output_dict
