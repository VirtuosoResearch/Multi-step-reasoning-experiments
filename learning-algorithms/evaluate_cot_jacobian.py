"""
Offline evaluator for CLRS chain-of-thought error amplification.

The evaluator estimates step-to-step Jacobian spectral norms in configurable
continuous state spaces and compares their amplification proxies with the
teacher-forced versus autoregressive final-step loss gap.
"""

import argparse
import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_clrs_text import (
    DEFAULT_QUANT_MODULES,
    expand_coconut_special_token_rows,
    initialize_model,
    validate_quant_args,
)
from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from src.custom.coconut_multitask_model import parse_coconut_answer
from src.utils.compute_metrics import compute_accuracy


EPS = 1e-12


def metric_float(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    return float(value)


@dataclass
class StepSequence:
    source: str
    steps: List[str]
    raw_answer: str


@dataclass
class StepEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_start: int
    target_ids: torch.Tensor
    spans: Dict[int, Tuple[int, int]]


@dataclass
class CoconutProxySequence:
    """One-vector-per-step textual proxies for a Coconut latent rollout."""

    source: str
    reasoning_steps: List[str]
    final_answer: str
    proxy_latents: List[torch.Tensor]
    source_ids: torch.Tensor


def parse_clrs_answer(answer: str) -> List[str]:
    """Parse repo CLRS text answers: step_1, ..., step_n | final."""
    answer = " ".join(answer.replace("\n", " ").split())
    if "|" in answer:
        steps_str, final = answer.split("|", 1)
        intermediate = [s.strip() for s in steps_str.split(",") if s.strip()]
        final = final.strip()
        return intermediate + ([final] if final else [])

    pieces = [s.strip() for s in answer.split(",") if s.strip()]
    return pieces


def split_generated_answer(answer: str) -> Tuple[List[str], str, str]:
    """Return intermediate steps, final answer, and context before final."""
    answer = " ".join(answer.replace("\n", " ").split())
    if "|" in answer:
        steps_str, final = answer.split("|", 1)
        steps = [s.strip() for s in steps_str.split(",") if s.strip()]
        context = steps_str.strip()
        if context:
            context += " | "
        return steps, final.strip(), context

    pieces = [s.strip() for s in answer.split(",") if s.strip()]
    if not pieces:
        return [], "", ""
    prefix = ", ".join(pieces[:-1])
    if prefix:
        prefix += ", "
    return pieces[:-1], pieces[-1], prefix


def decode_sample(tokenizer, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> StepSequence:
    nonpad = attention_mask.bool()
    source_mask = (labels == -100) & nonpad
    answer_mask = (labels != -100) & nonpad
    if tokenizer.pad_token_id is not None:
        answer_mask = answer_mask & (labels != tokenizer.pad_token_id)

    source = tokenizer.decode(input_ids[source_mask], skip_special_tokens=True)
    answer = tokenizer.decode(labels[answer_mask], skip_special_tokens=True)
    return StepSequence(
        source=" ".join(source.replace("\n", " ").split()),
        steps=parse_clrs_answer(answer),
        raw_answer=" ".join(answer.replace("\n", " ").split()),
    )


def check_source_not_truncated(raw_input: Optional[str], source: str, max_length: int) -> None:
    """Fail loudly when --max_length cut the prompt short.

    The data module truncates the source to max_length tokens, and for these graph
    tasks that lands mid-adjacency-matrix: the model is scored on a prompt it can no
    longer solve, so every loss, Jacobian and bound downstream is meaningless. The
    correct length is tokenizer-specific and must match training. This has already
    inverted the sign of one set of results, so it aborts rather than warns.
    """
    if raw_input is None:
        return
    normalised = " ".join(raw_input.replace("\n", " ").split())
    if len(source) >= len(normalised):
        return
    raise SystemExit(
        f"--max_length {max_length} truncated the prompt: kept {len(source)} of "
        f"{len(normalised)} characters. Re-run with the max_length this checkpoint was "
        "trained with (it is tokenizer-specific; see scripts/train_clrs_text/motivation_exps/)."
    )


def assemble_step_text(
    source: str,
    prefix_steps: Sequence[str],
    target_text: str,
    target_step: int,
    total_gold_steps: int,
) -> Tuple[str, Dict[int, Tuple[int, int]], int]:
    """Lay out the flat prompt text and record each step's character span.

    The layout mirrors training, which tokenizes ``source + " " + answer`` as one
    string (cf. build_causal_lm_instruction_batch). Steps are joined with ", " and
    the final answer is marked with " | ". The separator position is decided by
    ``total_gold_steps``, the length of the *gold* sequence, so restricting the
    analysed horizon never moves the " | " onto an intermediate step.
    """
    text = source + " "
    char_spans: Dict[int, Tuple[int, int]] = {0: (0, len(source))}

    for prev_step in range(1, target_step):
        if prev_step > len(prefix_steps):
            break
        start = len(text)
        text += prefix_steps[prev_step - 1]
        char_spans[prev_step] = (start, len(text))
        text += " | " if prev_step == total_gold_steps - 1 else ", "

    target_char_start = len(text)
    text += target_text
    return text, char_spans, target_char_start


def build_encoding(
    tokenizer,
    source: str,
    prefix_steps: Sequence[str],
    target_text: str,
    target_step: int,
    total_gold_steps: int,
    device: torch.device,
) -> Optional[StepEncoding]:
    """Tokenize the step-``target_step`` prompt in a single pass.

    Teacher forcing and free generation differ only in which strings fill
    ``prefix_steps``; both take this path, so identical prefix text yields
    identical token ids. Tokenizing once (rather than piecewise per step) is what
    keeps the encoding on the same token sequence the model saw during training,
    including any leading BOS.
    """
    if target_step < 1 or not target_text:
        return None

    text, char_spans, target_char_start = assemble_step_text(
        source, prefix_steps, target_text, target_step, total_gold_steps
    )

    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    ids: List[int] = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(ids) == 0:
        return None

    def tok_at(char_pos: int) -> int:
        """Tokens fully consumed by char_pos; a straddling token joins the later span."""
        count = 0
        for _start, end in offsets:
            if end <= char_pos:
                count += 1
            else:
                break
        return count

    spans: Dict[int, Tuple[int, int]] = {}
    for step_idx, (char_start, char_end) in char_spans.items():
        token_start, token_end = tok_at(char_start), tok_at(char_end)
        if token_end > token_start:
            spans[step_idx] = (token_start, token_end)

    target_start = tok_at(target_char_start)
    target_ids = ids[target_start:]
    if target_start < 1 or len(target_ids) == 0:
        return None

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return StepEncoding(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, device=device),
        target_start=target_start,
        target_ids=torch.tensor(target_ids, dtype=torch.long, device=device),
        spans=spans,
    )


def selected_target_log_probs(model, enc: StepEncoding, max_target_tokens: int = 0) -> torch.Tensor:
    target_ids = enc.target_ids
    if max_target_tokens > 0:
        target_ids = target_ids[:max_target_tokens]
    if target_ids.numel() == 0:
        return torch.empty(0, device=enc.input_ids.device)

    outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    logits = outputs.logits
    positions = torch.arange(
        enc.target_start - 1,
        enc.target_start - 1 + target_ids.numel(),
        device=enc.input_ids.device,
    )
    log_probs = F.log_softmax(logits[0, positions, :].float(), dim=-1)
    return log_probs[torch.arange(target_ids.numel(), device=enc.input_ids.device), target_ids]


def capped_target_ids(enc: StepEncoding, max_target_tokens: int) -> torch.Tensor:
    if max_target_tokens > 0:
        return enc.target_ids[:max_target_tokens]
    return enc.target_ids


def cap_step_span(span: Tuple[int, int], max_target_tokens: int) -> Tuple[int, int]:
    start, end = span
    if max_target_tokens <= 0:
        return start, end
    return start, min(end, start + max_target_tokens)


def compute_step_dims(
    encodings: Dict[int, Optional[StepEncoding]],
    analysis_steps: int,
    max_target_tokens: int,
) -> Dict[int, int]:
    """Token width to use for each step on both sides of the Jacobian composition.

    rho_{t,i} composes J_{t,k} with rho_{k,i}, so the output width of step k's map
    (its own target tokens) has to equal the input width of step k where it sits as
    context inside a later encoding. Subword merges at a step boundary can make
    those differ by a token, so we pin each step to the narrowest width that is
    consistent everywhere it appears.
    """
    dims: Dict[int, int] = {}
    for step in range(1, analysis_steps + 1):
        widths: List[int] = []
        enc = encodings.get(step)
        if enc is not None:
            widths.append(int(enc.target_ids.numel()))
        for later_step in range(step + 1, analysis_steps + 1):
            later_enc = encodings.get(later_step)
            if later_enc is not None and step in later_enc.spans:
                start, end = later_enc.spans[step]
                widths.append(end - start)
        if not widths:
            continue
        width = min(widths)
        if max_target_tokens > 0:
            width = min(width, max_target_tokens)
        if width > 0:
            dims[step] = width
    return dims


def resolve_input_embeddings(model) -> torch.nn.Module:
    """The embedding module, unwrapped from any PEFT ModulesToSaveWrapper.

    Coconut adds <bot>/<eot> and resizes the embeddings, which makes PEFT treat them as
    trainable and wrap them in a ModulesToSaveWrapper. That wrapper forwards fine but
    exposes no `.weight`, and the matrix this evaluator needs for `p @ W` is the trained
    copy in modules_to_save[active_adapter], not the frozen original_module.
    """
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise RuntimeError("Model does not expose input embeddings.")

    modules_to_save = getattr(embeddings, "modules_to_save", None)
    if modules_to_save is not None and len(modules_to_save) > 0:
        adapter = getattr(embeddings, "active_adapter", None)
        if adapter is not None and adapter in modules_to_save:
            return modules_to_save[adapter]
        if len(modules_to_save) == 1:
            return next(iter(modules_to_save.values()))
    return embeddings


def resolve_output_embeddings(model) -> torch.nn.Module:
    """The LM head / unembedding module, unwrapped when PEFT saves it specially."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Model does not expose output embeddings.")

    modules_to_save = getattr(output_embeddings, "modules_to_save", None)
    if modules_to_save is not None and len(modules_to_save) > 0:
        adapter = getattr(output_embeddings, "active_adapter", None)
        if adapter is not None and adapter in modules_to_save:
            return modules_to_save[adapter]
        if len(modules_to_save) == 1:
            return next(iter(modules_to_save.values()))
    return output_embeddings


def target_positions(enc: StepEncoding, num_targets: int, target_token_idx: Optional[int] = None) -> torch.Tensor:
    target_offset = target_token_idx or 0
    return torch.arange(
        enc.target_start - 1 + target_offset,
        enc.target_start - 1 + target_offset + num_targets,
        device=enc.input_ids.device,
    )


def jacobian_output_vector(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    embeds: torch.Tensor,
    max_target_tokens: int,
    jacobian_output: str,
    target_token_idx: Optional[int] = None,
) -> torch.Tensor:
    """Continuous step map used as the Jacobian output.

    token_distribution is the formulation the bound is stated in: the state is the
    next-step distribution over tokens, the ground truth is a one-hot, and the loss
    is a genuine function of that state. expected_embedding instead projects the
    distribution through the embedding matrix, which loses information -- distinct
    distributions share an expected embedding, so cross-entropy is not a function
    of it and no Lipschitz constant exists. selected_logprob is the original
    diagnostic map and does not compose across steps.
    """
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return torch.empty(0, device=embeds.device)
    if target_token_idx is not None:
        if target_token_idx < 0 or target_token_idx >= target_ids.numel():
            return torch.empty(0, device=embeds.device)
        target_ids = target_ids[target_token_idx : target_token_idx + 1]

    outputs = model(
        inputs_embeds=embeds,
        attention_mask=enc.attention_mask.detach(),
        output_hidden_states=(jacobian_output == "hidden_state"),
    )
    positions = target_positions(enc, target_ids.numel(), target_token_idx)
    if jacobian_output == "hidden_state":
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        return outputs.hidden_states[-1][0, positions, :].float().reshape(-1)

    logits = outputs.logits[0, positions, :].float()

    if jacobian_output == "selected_logprob":
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[torch.arange(target_ids.numel(), device=embeds.device), target_ids]

    if jacobian_output == "expected_embedding":
        probs = F.softmax(logits, dim=-1)
        return (probs @ input_embeddings.weight.float()).reshape(-1)

    if jacobian_output == "token_distribution":
        return F.softmax(logits, dim=-1).reshape(-1)

    raise ValueError(f"Unknown jacobian_output: {jacobian_output}")


def input_parameterisation(
    jacobian_output: str,
    base_embeds: torch.Tensor,
    span: Tuple[int, int],
    weight: torch.Tensor,
):
    """Return (x0, to_embeds) for the space the Jacobian's input lives in.

    For token_distribution the free variable is a perturbation of the token
    distribution feeding step i. The model consumes embeddings, and that map is the
    fixed linear z = pW, so we push delta_p through W. Parameterising by delta_p
    around zero (rather than reconstructing onehot @ W) makes the base point
    reproduce the teacher-forced forward exactly while having the same derivative.
    """
    start, end = span
    span_slice = base_embeds[:, start:end, :]

    if jacobian_output == "token_distribution":
        x0 = torch.zeros(
            end - start, weight.shape[0], dtype=torch.float32, device=base_embeds.device
        )

        def to_embeds(delta_p: torch.Tensor) -> torch.Tensor:
            shift = (delta_p.view(end - start, -1) @ weight).unsqueeze(0)
            return span_slice + shift.to(dtype=span_slice.dtype)

        return x0, to_embeds

    return span_slice.detach().clone(), lambda x: x.view(span_slice.shape)


@torch.no_grad()
def state_vector(
    model,
    input_embeddings: torch.nn.Module,
    enc: Optional[StepEncoding],
    max_target_tokens: int,
    jacobian_output: str,
) -> Optional[torch.Tensor]:
    """The step's predicted state, in whichever space the Jacobians are taken."""
    if enc is None:
        return None
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return None
    embeds = input_embeddings(enc.input_ids).detach()
    return jacobian_output_vector(
        model,
        input_embeddings,
        enc,
        embeds,
        max_target_tokens=max_target_tokens,
        jacobian_output=jacobian_output,
    ).float().detach()


@torch.no_grad()
def state_delta_norm(
    model,
    input_embeddings: torch.nn.Module,
    enc_a: Optional[StepEncoding],
    enc_b: Optional[StepEncoding],
    max_target_tokens: int,
    jacobian_output: str,
    jacobian_granularity: str = "block",
) -> float:
    """||Delta z_T|| of Equation 9: the prediction gap between two contexts."""
    vec_a = state_vector(model, input_embeddings, enc_a, max_target_tokens, jacobian_output)
    vec_b = state_vector(model, input_embeddings, enc_b, max_target_tokens, jacobian_output)
    if vec_a is None or vec_b is None or vec_a.numel() != vec_b.numel():
        return float("nan")
    if jacobian_granularity == "token_mean" and enc_a is not None:
        target_ids = capped_target_ids(enc_a, max_target_tokens)
        if target_ids.numel() > 0:
            delta = vec_a.float().view(target_ids.numel(), -1) - vec_b.float().view(target_ids.numel(), -1)
            return float(delta.norm(dim=-1).mean().detach().cpu())
    return float((vec_a - vec_b).float().norm().detach().cpu())


def empirical_lipschitz_constant(loss_a: float, loss_b: float, delta_norm: float) -> float:
    if not (np.isfinite(loss_a) and np.isfinite(loss_b) and np.isfinite(delta_norm)):
        return float("nan")
    if delta_norm < EPS:
        return 0.0 if abs(loss_b - loss_a) < EPS else float("nan")
    return abs(loss_b - loss_a) / delta_norm


@torch.no_grad()
def state_step_error(
    model,
    input_embeddings: torch.nn.Module,
    enc: Optional[StepEncoding],
    max_target_tokens: int,
    jacobian_output: str,
    jacobian_granularity: str = "block",
) -> Optional[float]:
    """||e_i^train||: distance from the step's prediction to the ground truth.

    In token_distribution space the ground truth is the one-hot, so this is zero iff
    the model puts all mass on the gold tokens. The expected_embedding version can
    report zero for a wrong model whose mass happens to average to the gold
    embedding, which is why the probability space is the faithful one.
    """
    if enc is None:
        return None
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return None

    pred_vec = state_vector(model, input_embeddings, enc, max_target_tokens, jacobian_output)
    if pred_vec is None:
        return None
    pred = pred_vec.view(target_ids.numel(), -1)

    if jacobian_output == "token_distribution":
        gold = F.one_hot(target_ids, num_classes=pred.shape[-1]).float()
    else:
        gold = input_embeddings(target_ids).float()
    if jacobian_granularity == "token_mean":
        return float((pred.float() - gold).norm(dim=-1).mean().detach().cpu())
    return float((pred.float() - gold).norm().detach().cpu())


@torch.no_grad()
def step_ce_loss(model, enc: Optional[StepEncoding]) -> Optional[float]:
    if enc is None or enc.target_ids.numel() == 0:
        return None
    outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = enc.input_ids[:, 1:].contiguous()

    valid_start = enc.target_start - 1
    valid_end = valid_start + enc.target_ids.numel()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, valid_start:valid_end] = True
    if not mask.any():
        return None

    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        labels.view(-1),
        reduction="none",
    )
    return float(losses[mask.view(-1)].mean().detach().cpu())


def step_ce_loss_from_embeds(model, enc: StepEncoding, embeds: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs_embeds=embeds, attention_mask=enc.attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = enc.input_ids[:, 1:].contiguous()

    valid_start = enc.target_start - 1
    valid_end = valid_start + enc.target_ids.numel()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, valid_start:valid_end] = True
    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        labels.view(-1),
        reduction="none",
    )
    return losses[mask.view(-1)].mean()


@torch.no_grad()
def estimate_ce_embedding_lipschitz_by_noise(
    model,
    input_embeddings: torch.nn.Module,
    enc: Optional[StepEncoding],
    span: Optional[Tuple[int, int]],
    num_checks: int,
    epsilon: float,
) -> Dict[str, float]:
    """Sensitivity of the step loss to its *input* embeddings, reported as input_sensitivity_*.

    This is not Equation 4's C. Perturbing the conditioning step measures
    |dl| / ||d(input)||, which by the chain rule already contains a ||J|| factor, so
    multiplying it by rho would count that Jacobian twice. Use
    estimate_ce_lipschitz for C; keep this as a diagnostic.
    """
    if enc is None or span is None or num_checks <= 0 or epsilon <= 0:
        return {"ce_lipschitz_noise_max": float("nan"), "ce_lipschitz_noise_mean": float("nan")}

    start, end = span
    if end <= start:
        return {"ce_lipschitz_noise_max": float("nan"), "ce_lipschitz_noise_mean": float("nan")}

    base_embeds = input_embeddings(enc.input_ids).detach()
    base_span = base_embeds[:, start:end, :]
    ratios: List[float] = []
    with attention_kernel_context("math"):
        for _ in range(num_checks):
            direction = _normalize_like(torch.randn_like(base_span))
            perturb = epsilon * direction.float()
            perturb_norm = perturb.norm()
            if not torch.isfinite(perturb_norm) or perturb_norm.item() < EPS:
                continue

            embeds_plus = base_embeds.clone()
            embeds_minus = base_embeds.clone()
            embeds_plus[:, start:end, :] = (base_span.float() + perturb).to(dtype=base_embeds.dtype)
            embeds_minus[:, start:end, :] = (base_span.float() - perturb).to(dtype=base_embeds.dtype)

            loss_plus = step_ce_loss_from_embeds(model, enc, embeds_plus)
            loss_minus = step_ce_loss_from_embeds(model, enc, embeds_minus)
            ratio = (loss_plus.float() - loss_minus.float()).abs() / (2.0 * perturb_norm)
            if torch.isfinite(ratio):
                ratios.append(float(ratio.detach().cpu()))

    if not ratios:
        return {"ce_lipschitz_noise_max": float("nan"), "ce_lipschitz_noise_mean": float("nan")}
    return {
        "ce_lipschitz_noise_max": max(ratios),
        "ce_lipschitz_noise_mean": float(np.mean(ratios)),
    }


def ce_from_probs(probs: Optional[torch.Tensor], target_ids: torch.Tensor) -> float:
    """Cross-entropy restricted to the tokens the state space actually covers.

    The Jacobians, the error term and C are all computed on the first
    max_target_tokens_for_jacobian tokens of a step, so the loss used to check the
    bound has to be restricted the same way. Dividing a full-length CE by a capped
    ||Delta p|| mixes units and breaks the segment bound.
    """
    if probs is None or target_ids.numel() == 0:
        return float("nan")
    gold = probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
    return float(-gold.log().mean().cpu())


def logits_from_hidden(output_embeddings: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    logits = hidden.float() @ output_embeddings.weight.float().t()
    bias = getattr(output_embeddings, "bias", None)
    if bias is not None:
        logits = logits + bias.float()
    return logits


def ce_from_hidden(output_embeddings: torch.nn.Module, hidden: Optional[torch.Tensor], target_ids: torch.Tensor) -> float:
    if hidden is None or target_ids.numel() == 0:
        return float("nan")
    logits = logits_from_hidden(output_embeddings, hidden)
    loss = F.cross_entropy(logits.float(), target_ids, reduction="mean")
    return float(loss.detach().cpu())


def ce_gradient_norm(probs: torch.Tensor, target_ids: torch.Tensor) -> float:
    """||grad_p CE|| at a point of the simplex, for the mean-over-tokens reduction.

    CE(p) = -(1/n) sum_j log p_{j,y_j}, so the gradient block for token j is
    -(1/n) e_{y_j} / p_{j,y_j} and the norm is (1/n) sqrt(sum_j p_{j,y_j}^-2).
    """
    n = target_ids.numel()
    gold = probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
    return float((gold.double().pow(-2).sum().sqrt() / n).cpu())


def hidden_ce_gradient_norm(
    output_embeddings: torch.nn.Module,
    hidden: Optional[torch.Tensor],
    target_ids: torch.Tensor,
    jacobian_granularity: str,
) -> float:
    """||grad_h CE|| for mean-over-token CE in hidden-state coordinates."""
    if hidden is None or target_ids.numel() == 0:
        return float("nan")
    weight = output_embeddings.weight.float()
    probs = F.softmax(logits_from_hidden(output_embeddings, hidden), dim=-1)
    grads = probs @ weight - weight[target_ids]
    if grads.numel() == 0:
        return float("nan")
    if jacobian_granularity == "token_mean":
        return float(grads.float().norm(dim=-1).max().detach().cpu())
    return float((grads.float().norm() / target_ids.numel()).detach().cpu())


def hidden_ce_global_lipschitz(output_embeddings: torch.nn.Module) -> float:
    """Global bound 2 max_v ||w_v|| for CE as a function of hidden state."""
    weight = output_embeddings.weight.float()
    return float((2.0 * weight.norm(dim=-1).max()).detach().cpu())


def ce_segment_lipschitz(
    probs_a: Optional[torch.Tensor],
    probs_b: Optional[torch.Tensor],
    target_ids: torch.Tensor,
) -> float:
    """A provable Lipschitz constant for CE over the segment joining two predictions.

    Equation 6 needs C valid along the whole segment from the teacher-forced
    prediction to the generated one, not merely at a point. Each p_{j,y_j}(s) is
    linear in s, so sum_j p_{j,y_j}(s)^-2 is convex and attains its maximum at an
    endpoint -- evaluating both endpoints therefore bounds the segment exactly.

    Cross-entropy has no finite *global* Lipschitz constant (-log p_y is unbounded
    on a bounded simplex), so C is necessarily reported per sample and grows as
    1/p_y: the bound degrades exactly as the model loses confidence in the gold token.
    """
    candidates = [p for p in (probs_a, probs_b) if p is not None]
    if not candidates or target_ids.numel() == 0:
        return float("nan")
    return max(ce_gradient_norm(p, target_ids) for p in candidates)


@torch.no_grad()
def step_target_probs(
    model,
    enc: Optional[StepEncoding],
    max_target_tokens: int,
) -> Optional[torch.Tensor]:
    """Next-token distributions at the step's target positions, shape (n, V)."""
    if enc is None:
        return None
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return None
    outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    positions = target_positions(enc, target_ids.numel())
    return F.softmax(outputs.logits[0, positions, :].float(), dim=-1)


@torch.no_grad()
def step_target_hidden_states(
    model,
    enc: Optional[StepEncoding],
    max_target_tokens: int,
) -> Optional[torch.Tensor]:
    """Pre-unembedding hidden states at next-token target positions, shape (n, d)."""
    if enc is None:
        return None
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return None
    outputs = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        output_hidden_states=True,
    )
    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")
    positions = target_positions(enc, target_ids.numel())
    return outputs.hidden_states[-1][0, positions, :].float()


@torch.no_grad()
def estimate_ce_lipschitz(
    probs: Optional[torch.Tensor],
    probs_other: Optional[torch.Tensor],
    target_ids: torch.Tensor,
    num_checks: int,
    epsilon: float,
) -> Dict[str, float]:
    """C for cross-entropy in probability space: analytic bound plus random probes.

    Pure function of the two predictions -- no model forward passes beyond the ones
    already performed. Random perturbations are drawn on the simplex tangent
    {sum_v delta_v = 0} so the perturbed point stays a distribution to first order.
    That estimate is a sup approximated by sampling, hence a *lower* estimate; it
    exists to cross-check C_segment, not to replace it.
    """
    result = {
        "C_segment": float("nan"),
        "C_point_teacher": float("nan"),
        "C_random_max": float("nan"),
        "C_random_mean": float("nan"),
    }
    if probs is None or target_ids.numel() == 0:
        return result

    result["C_point_teacher"] = ce_gradient_norm(probs, target_ids)
    result["C_segment"] = ce_segment_lipschitz(probs, probs_other, target_ids)

    if num_checks <= 0 or epsilon <= 0:
        return result

    def ce_of(p: torch.Tensor) -> torch.Tensor:
        gold = p.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
        return -gold.log().mean()

    ratios: List[float] = []
    for _ in range(num_checks):
        direction = torch.randn_like(probs)
        direction = direction - direction.mean(dim=-1, keepdim=True)  # simplex tangent
        norm = direction.norm()
        if not torch.isfinite(norm) or norm.item() < EPS:
            continue
        direction = direction / norm

        step = epsilon * direction
        ratio = (ce_of(probs + step) - ce_of(probs - step)).abs() / (2.0 * epsilon)
        if torch.isfinite(ratio):
            ratios.append(float(ratio.cpu()))

    if ratios:
        result["C_random_max"] = max(ratios)
        result["C_random_mean"] = float(np.mean(ratios))
    return result


@torch.no_grad()
def estimate_hidden_ce_lipschitz(
    output_embeddings: torch.nn.Module,
    hidden: Optional[torch.Tensor],
    hidden_other: Optional[torch.Tensor],
    target_ids: torch.Tensor,
    jacobian_granularity: str,
    num_checks: int,
    epsilon: float,
    global_C: Optional[float] = None,
) -> Dict[str, float]:
    """C for cross-entropy as a function of the pre-unembedding hidden state."""
    result = {
        "C_segment": float("nan"),
        "C_point_teacher": float("nan"),
        "C_random_max": float("nan"),
        "C_random_mean": float("nan"),
        "C_global": hidden_ce_global_lipschitz(output_embeddings) if global_C is None else global_C,
    }
    if hidden is None or target_ids.numel() == 0:
        return result

    candidates = [h for h in (hidden, hidden_other) if h is not None and h.numel() == hidden.numel()]
    if candidates:
        values = [
            hidden_ce_gradient_norm(output_embeddings, h, target_ids, jacobian_granularity)
            for h in candidates
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        if finite_values:
            result["C_segment"] = max(finite_values)

    if num_checks <= 0 or epsilon <= 0:
        return result

    ratios: List[float] = []
    for _ in range(num_checks):
        direction = _normalize_like(torch.randn_like(hidden))
        step = epsilon * direction.float()
        perturb_norm = step.float().norm()
        if not torch.isfinite(perturb_norm) or perturb_norm.item() < EPS:
            continue

        loss_plus = ce_from_hidden(output_embeddings, hidden + step, target_ids)
        loss_minus = ce_from_hidden(output_embeddings, hidden - step, target_ids)
        ratio = abs(loss_plus - loss_minus) / (2.0 * perturb_norm.item())
        if np.isfinite(ratio):
            ratios.append(float(ratio))

    if ratios:
        result["C_random_max"] = max(ratios)
    return result


def teacher_forced_step_losses(
    model,
    encodings: Dict[int, Optional[StepEncoding]],
    analysis_steps: int,
) -> List[Optional[float]]:
    return [step_ce_loss(model, encodings.get(t)) for t in range(1, analysis_steps + 1)]


@torch.no_grad()
def generate_answer(model, tokenizer, source: str, device: torch.device, max_new_tokens: int) -> str:
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True).to(device)
    tokenizer.padding_side = previous_side

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def _normalize_like(x: torch.Tensor) -> torch.Tensor:
    denom = x.float().norm()
    if not torch.isfinite(denom) or denom.item() < EPS:
        return torch.randn_like(x).float()
    return (x.float() / denom).to(dtype=x.dtype)


def estimate_function_spectral_norm_jvp(
    fn,
    x0: torch.Tensor,
    power_iters: int,
) -> float:
    """Estimate ``||d fn(x0) / dx||_2`` for an arbitrary differentiable map."""
    output = fn(x0)
    if output.numel() == 0 or x0.numel() == 0:
        return 0.0

    v = _normalize_like(torch.randn_like(x0))
    sigma = torch.tensor(0.0, device=x0.device)
    for _ in range(max(1, power_iters)):
        _, jv = torch.autograd.functional.jvp(
            fn, (x0,), (v,), create_graph=False, strict=False
        )
        sigma = jv.float().norm()
        if not torch.isfinite(sigma) or sigma.item() < EPS:
            return 0.0

        u = (jv.float() / sigma).detach()
        x_req = x0.detach().clone().requires_grad_(True)
        y = fn(x_req)
        (jt_u,) = torch.autograd.grad(
            y,
            x_req,
            grad_outputs=u.to(dtype=y.dtype),
            retain_graph=False,
            allow_unused=False,
        )
        v_norm = jt_u.float().norm()
        if not torch.isfinite(v_norm) or v_norm.item() < EPS:
            return float(sigma.detach().cpu())
        v = (jt_u.float() / v_norm).to(dtype=x0.dtype).detach()

    _, jv = torch.autograd.functional.jvp(
        fn, (x0,), (v,), create_graph=False, strict=False
    )
    sigma = jv.float().norm()
    return float(sigma.detach().cpu()) if torch.isfinite(sigma) else float("nan")


def estimate_function_jacobian_lipschitz_by_jvp_difference(
    fn,
    x0: torch.Tensor,
    epsilon: float,
    num_checks: int,
) -> Dict[str, float]:
    """Randomized local estimate of the Jacobian Lipschitz constant of ``fn``."""
    if x0.numel() == 0 or num_checks <= 0:
        return {"M_random_max": float("nan"), "M_random_mean": float("nan")}
    try:
        if fn(x0).numel() == 0:
            return {"M_random_max": 0.0, "M_random_mean": 0.0}
    except RuntimeError:
        return {"M_random_max": float("nan"), "M_random_mean": float("nan")}

    estimates: List[float] = []
    for _ in range(num_checks):
        u = _normalize_like(torch.randn_like(x0))
        v = _normalize_like(torch.randn_like(x0))
        try:
            _, jv_base = torch.autograd.functional.jvp(
                fn, (x0,), (v,), create_graph=False, strict=False
            )
            _, jv_perturbed = torch.autograd.functional.jvp(
                fn,
                (x0 + epsilon * u,),
                (v,),
                create_graph=False,
                strict=False,
            )
        except RuntimeError:
            continue
        estimate = (jv_perturbed.float() - jv_base.float()).norm() / max(epsilon, EPS)
        if torch.isfinite(estimate):
            estimates.append(float(estimate.detach().cpu()))

    if not estimates:
        return {"M_random_max": float("nan"), "M_random_mean": float("nan")}
    return {
        "M_random_max": max(estimates),
        "M_random_mean": float(np.mean(estimates)),
    }


def finite_difference_function_check(
    fn,
    x0: torch.Tensor,
    epsilon: float,
) -> Dict[str, float]:
    """Compare one JVP with a centered finite difference for an arbitrary map."""
    direction = _normalize_like(torch.randn_like(x0))
    _, jvp = torch.autograd.functional.jvp(
        fn, (x0,), (direction,), create_graph=False, strict=False
    )
    with torch.no_grad():
        fd = (fn(x0 + epsilon * direction) - fn(x0 - epsilon * direction)) / (
            2.0 * epsilon
        )
    jvp_norm = float(jvp.float().norm().detach().cpu())
    fd_norm = float(fd.float().norm().detach().cpu())
    rel_error = abs(jvp_norm - fd_norm) / max(abs(fd_norm), EPS)
    cosine = float(
        F.cosine_similarity(jvp.float().flatten(), fd.float().flatten(), dim=0)
        .detach()
        .cpu()
    )
    return {
        "jvp_directional_norm": jvp_norm,
        "finite_difference_directional_norm": fd_norm,
        "relative_norm_error": rel_error,
        "directional_cosine": cosine,
    }


def attention_kernel_context(backend: str):
    """Select the SDPA kernel used by transformer attention during evaluation."""
    if backend == "auto" or not torch.cuda.is_available():
        return nullcontext()

    if backend != "math":
        raise ValueError(f"Unsupported attention backend: {backend}")

    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.MATH)
        except Exception:
            pass

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            )
        except TypeError:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )

    return nullcontext()


def estimate_jacobian_spectral_norm_jvp(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    span: Tuple[int, int],
    power_iters: int,
    max_target_tokens: int,
    jacobian_output: str,
    target_token_idx: Optional[int] = None,
) -> float:
    """Estimate the spectral norm of the selected step map with JVP/VJP power iteration."""
    start, end = span
    if end <= start:
        return 0.0

    base_embeds = input_embeddings(enc.input_ids).detach()
    weight = input_embeddings.weight.float()
    x0, to_embeds = input_parameterisation(jacobian_output, base_embeds, span, weight)

    def fn(x: torch.Tensor) -> torch.Tensor:
        embeds = base_embeds.clone()
        embeds[:, start:end, :] = to_embeds(x)
        return jacobian_output_vector(
            model,
            input_embeddings,
            enc,
            embeds,
            max_target_tokens=max_target_tokens,
            jacobian_output=jacobian_output,
            target_token_idx=target_token_idx,
        )

    return estimate_function_spectral_norm_jvp(fn, x0, power_iters)


def estimate_jacobian_spectral_norm_rowgrad(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    span: Tuple[int, int],
    power_iters: int,
    max_target_tokens: int,
    jacobian_output: str,
    target_token_idx: Optional[int] = None,
) -> float:
    """Compute the selected Jacobian rows with first-order reverse-mode gradients.

    This is exact but only practical for low-dimensional outputs. For
    expected_embedding, prefer the JVP method because the output dimension is
    max_target_tokens * hidden_size.
    """
    if jacobian_output in ("expected_embedding", "token_distribution", "hidden_state"):
        raise ValueError(
            f"--jacobian_method rowgrad is too expensive for --jacobian_output {jacobian_output}; "
            "use --jacobian_method jvp"
        )

    start, end = span
    if end <= start:
        return 0.0

    base_embeds = input_embeddings(enc.input_ids).detach()
    base_attention = enc.attention_mask.detach()
    span_shape = base_embeds[:, start:end, :].shape
    x_req = base_embeds[:, start:end, :].detach().clone().requires_grad_(True)

    embeds = base_embeds.clone()
    embeds[:, start:end, :] = x_req.view(span_shape).to(dtype=embeds.dtype)
    selected = jacobian_output_vector(
        model,
        input_embeddings,
        enc,
        embeds,
        max_target_tokens=max_target_tokens,
        jacobian_output=jacobian_output,
        target_token_idx=target_token_idx,
    )
    if selected.numel() == 0:
        return 0.0

    rows = []
    for row_idx in range(selected.numel()):
        (grad_row,) = torch.autograd.grad(
            selected[row_idx],
            x_req,
            retain_graph=row_idx < selected.numel() - 1,
            create_graph=False,
            allow_unused=False,
        )
        rows.append(grad_row.float().reshape(-1))

    jacobian = torch.stack(rows, dim=0)
    if jacobian.numel() == 0:
        return 0.0
    sigma = torch.linalg.svdvals(jacobian).max()
    if not torch.isfinite(sigma):
        return float("nan")
    return float(sigma.detach().cpu())


def estimate_jacobian_spectral_norm(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    span: Tuple[int, int],
    power_iters: int,
    max_target_tokens: int,
    method: str,
    jacobian_output: str,
    target_token_idx: Optional[int] = None,
) -> float:
    if method == "jvp":
        return estimate_jacobian_spectral_norm_jvp(
            model,
            input_embeddings,
            enc,
            span,
            power_iters=power_iters,
            max_target_tokens=max_target_tokens,
            jacobian_output=jacobian_output,
            target_token_idx=target_token_idx,
        )
    if method == "rowgrad":
        return estimate_jacobian_spectral_norm_rowgrad(
            model,
            input_embeddings,
            enc,
            span,
            power_iters=power_iters,
            max_target_tokens=max_target_tokens,
            jacobian_output=jacobian_output,
            target_token_idx=target_token_idx,
        )
    raise ValueError(f"Unknown Jacobian method: {method}")


def estimate_token_mean_jacobian_norm(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    source_span: Tuple[int, int],
    target_width: int,
    power_iters: int,
    max_target_tokens: int,
    method: str,
    jacobian_output: str,
) -> Tuple[float, List[Dict[str, float]]]:
    """Average spectral norms over target-token/source-token Jacobian blocks."""
    start, end = source_span
    source_width = max(0, end - start)
    target_ids = capped_target_ids(enc, max_target_tokens)
    target_width = min(target_width, int(target_ids.numel()))
    if source_width == 0 or target_width == 0:
        return 0.0, []

    token_rows: List[Dict[str, float]] = []
    norms: List[float] = []
    for target_token_idx in range(target_width):
        for source_token_idx in range(source_width):
            token_span = (start + source_token_idx, start + source_token_idx + 1)
            token_norm = estimate_jacobian_spectral_norm(
                model,
                input_embeddings,
                enc,
                token_span,
                power_iters=power_iters,
                max_target_tokens=max_target_tokens,
                method=method,
                jacobian_output=jacobian_output,
                target_token_idx=target_token_idx,
            )
            norms.append(token_norm)
            token_rows.append(
                {
                    "target_token_idx": target_token_idx,
                    "source_token_idx": source_token_idx,
                    "token_jac_norm": token_norm,
                    "token_log_jac_norm": (
                        math.log(max(token_norm, EPS)) if np.isfinite(token_norm) else float("nan")
                    ),
                }
            )

    finite_norms = [norm for norm in norms if np.isfinite(norm)]
    if not finite_norms:
        return float("nan"), token_rows
    return float(np.mean(finite_norms)), token_rows


def finite_difference_directional_check(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    span: Tuple[int, int],
    max_target_tokens: int,
    epsilon: float,
    jacobian_output: str,
) -> Dict[str, float]:
    start, end = span
    base_embeds = input_embeddings(enc.input_ids).detach()
    weight = input_embeddings.weight.float()
    x0, to_embeds = input_parameterisation(jacobian_output, base_embeds, span, weight)

    def fn(x: torch.Tensor) -> torch.Tensor:
        embeds = base_embeds.clone()
        embeds[:, start:end, :] = to_embeds(x)
        return jacobian_output_vector(
            model,
            input_embeddings,
            enc,
            embeds,
            max_target_tokens=max_target_tokens,
            jacobian_output=jacobian_output,
        )

    direction = _normalize_like(torch.randn_like(x0))
    try:
        _, jvp = torch.autograd.functional.jvp(
            fn,
            (x0,),
            (direction,),
            create_graph=False,
            strict=False,
        )
    except RuntimeError:
        x_req = x0.detach().clone().requires_grad_(True)
        y = fn(x_req)
        if y.numel() > 512:
            return {
                "jvp_directional_norm": float("nan"),
                "finite_difference_directional_norm": float("nan"),
                "relative_norm_error": float("nan"),
                "directional_cosine": float("nan"),
            }
        jvp_rows = []
        for row_idx in range(y.numel()):
            (grad_row,) = torch.autograd.grad(
                y[row_idx],
                x_req,
                retain_graph=row_idx < y.numel() - 1,
                create_graph=False,
                allow_unused=False,
            )
            jvp_rows.append((grad_row.float() * direction.float()).sum())
        jvp = torch.stack(jvp_rows) if jvp_rows else torch.empty(0, device=x0.device)
    with torch.no_grad():
        fd = (fn(x0 + epsilon * direction) - fn(x0 - epsilon * direction)) / (2.0 * epsilon)

    jvp_norm = float(jvp.float().norm().detach().cpu())
    fd_norm = float(fd.float().norm().detach().cpu())
    rel_error = abs(jvp_norm - fd_norm) / max(abs(fd_norm), EPS)
    cosine = float(
        F.cosine_similarity(jvp.float().flatten(), fd.float().flatten(), dim=0).detach().cpu()
    )
    return {
        "jvp_directional_norm": jvp_norm,
        "finite_difference_directional_norm": fd_norm,
        "relative_norm_error": rel_error,
        "directional_cosine": cosine,
    }


def estimate_jacobian_lipschitz_by_jvp_difference(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    span: Tuple[int, int],
    max_target_tokens: int,
    epsilon: float,
    jacobian_output: str,
    num_checks: int,
) -> Dict[str, float]:
    """Estimate local Jacobian Lipschitzness with randomized JVP differences."""
    start, end = span
    if end <= start or num_checks <= 0:
        return {"M_random_max": float("nan"), "M_random_mean": float("nan")}

    base_embeds = input_embeddings(enc.input_ids).detach()
    weight = input_embeddings.weight.float()
    x0, to_embeds = input_parameterisation(jacobian_output, base_embeds, span, weight)

    def fn(x: torch.Tensor) -> torch.Tensor:
        embeds = base_embeds.clone()
        embeds[:, start:end, :] = to_embeds(x)
        return jacobian_output_vector(
            model,
            input_embeddings,
            enc,
            embeds,
            max_target_tokens=max_target_tokens,
            jacobian_output=jacobian_output,
        )

    return estimate_function_jacobian_lipschitz_by_jvp_difference(
        fn, x0, epsilon, num_checks
    )


def compute_rho(jac_norms: Dict[Tuple[int, int], float], total_steps: int) -> Dict[Tuple[int, int], float]:
    rho: Dict[Tuple[int, int], float] = {}
    for t in range(2, total_steps + 1):
        for i in range(1, t):
            val = jac_norms.get((t, i), 0.0)
            for k in range(i + 1, t):
                val += jac_norms.get((t, k), 0.0) * rho.get((k, i), 0.0)
            rho[(t, i)] = val
    return rho


def compute_terminal_rho(
    intermediate_rho: Dict[Tuple[int, int], float],
    terminal_jac_norms: Dict[Tuple[int, int], float],
    total_steps: int,
) -> Dict[Tuple[int, int], float]:
    """Compute tilde rho ending at a terminal state different from the step state."""
    terminal_rho: Dict[Tuple[int, int], float] = {}
    for t in range(2, total_steps + 1):
        for i in range(1, t):
            val = terminal_jac_norms.get((t, i), 0.0)
            for k in range(i + 1, t):
                val += terminal_jac_norms.get((t, k), 0.0) * intermediate_rho.get((k, i), 0.0)
            terminal_rho[(t, i)] = val
    return terminal_rho


def compute_gamma(intermediate_rho: Dict[Tuple[int, int], float], total_steps: int) -> Dict[int, float]:
    gamma: Dict[int, float] = {1: 1.0}
    for k in range(2, total_steps + 1):
        candidates = []
        for j in range(1, k):
            candidates.append(
                1.0 + sum(intermediate_rho.get((i, j), 0.0) for i in range(j + 1, k))
            )
        gamma[k] = max(candidates) if candidates else 1.0
    return gamma


def compute_second_order_prefix_stats(
    effective_rho: Dict[Tuple[int, int], float],
    test_errors: Sequence[float],
    M_prefix_max: Dict[int, float],
    M_prefix_mean: Dict[int, float],
    total_steps: int,
) -> Dict[int, Dict[str, float]]:
    prefix_test_error_square_sums: Dict[int, float] = {0: 0.0}
    for k in range(1, total_steps + 1):
        err = test_errors[k - 1] if k - 1 < len(test_errors) else float("nan")
        prefix_test_error_square_sums[k] = prefix_test_error_square_sums[k - 1] + (
            (err ** 2) if np.isfinite(err) else 0.0
        )

    stats: Dict[int, Dict[str, float]] = {}
    for t in range(1, total_steps + 1):
        M_t = M_prefix_max.get(t, float("nan"))
        M_mean_t = M_prefix_mean.get(t, float("nan"))
        test_error_square_sum = prefix_test_error_square_sums.get(t - 1, 0.0)
        weighted_test_error_square_sum = 0.0
        for i in range(1, t):
            err = test_errors[i - 1] if i - 1 < len(test_errors) else float("nan")
            if not np.isfinite(err):
                continue
            remainder_weight = 1.0 + sum(effective_rho.get((t, k), 0.0) for k in range(i + 1, t))
            weighted_test_error_square_sum += remainder_weight * (err ** 2)

        rho_remainder_multiplier = (
            weighted_test_error_square_sum / test_error_square_sum
            if test_error_square_sum > EPS
            else float("nan")
        )

        if np.isfinite(M_t):
            geometry_amount = 0.5 * M_t * weighted_test_error_square_sum
            geometry_coefficient = (
                geometry_amount / test_error_square_sum
                if test_error_square_sum > EPS
                else float("nan")
            )
        else:
            geometry_coefficient = float("nan")
            geometry_amount = float("nan")

        stats[t] = {
            "M_random_max_t": M_t,
            "M_random_mean_t": M_mean_t,
            "gamma_t": float("nan"),
            "rho_remainder_multiplier_t": rho_remainder_multiplier,
            "test_error_square_sum_t": test_error_square_sum,
            "second_order_weighted_test_error_square_sum_t": weighted_test_error_square_sum,
            "second_order_geometry_coefficient_t": geometry_coefficient,
            "second_order_geometry_amount_t": geometry_amount,
        }
    return stats


def local_log_amplifications(jac_norms: Dict[Tuple[int, int], float], total_steps: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for i in range(1, total_steps):
        log_amp = 0.0
        for k in range(i + 1, total_steps + 1):
            log_amp += math.log(max(jac_norms.get((k, k - 1), 0.0), EPS))
        result[i] = log_amp
    return result


def local_log_amplifications_with_terminal(
    intermediate_jac_norms: Dict[Tuple[int, int], float],
    terminal_jac_norms: Dict[Tuple[int, int], float],
    total_steps: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for i in range(1, total_steps):
        log_amp = 0.0
        for k in range(i + 1, total_steps):
            log_amp += math.log(max(intermediate_jac_norms.get((k, k - 1), 0.0), EPS))
        log_amp += math.log(max(terminal_jac_norms.get((total_steps, total_steps - 1), 0.0), EPS))
        result[i] = log_amp
    return result


def prefix_log_amplifications_from_first(jac_norms: Dict[Tuple[int, int], float], total_steps: int) -> Dict[int, float]:
    result: Dict[int, float] = {1: 0.0}
    log_amp = 0.0
    for t in range(2, total_steps + 1):
        log_amp += math.log(max(jac_norms.get((t, t - 1), 0.0), EPS))
        result[t] = log_amp
    return result


def prefix_log_amplifications_from_first_with_terminal(
    intermediate_jac_norms: Dict[Tuple[int, int], float],
    terminal_jac_norms: Dict[Tuple[int, int], float],
    total_steps: int,
) -> Dict[int, float]:
    """Adjacent-chain proxy where the last edge lands in the terminal state."""
    result: Dict[int, float] = {1: 0.0}
    intermediate_log_amp = 0.0
    for t in range(2, total_steps + 1):
        if t > 2:
            intermediate_log_amp += math.log(max(intermediate_jac_norms.get((t - 1, t - 2), 0.0), EPS))
        terminal_log = math.log(max(terminal_jac_norms.get((t, t - 1), 0.0), EPS))
        result[t] = intermediate_log_amp + terminal_log
    return result


def write_csv(path: str, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_analysis_outputs(output_dir: str, example_rows: List[Dict], jacobian_rows: List[Dict]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}
    for row in example_rows:
        grouped.setdefault((row["task"], row["length"]), []).append(row)

    summary_rows = []
    for (task, length), rows in sorted(grouped.items()):
        first = rows[0]
        summary_rows.append(
            {
                "task": task,
                "length": length,
                "evaluation_mode": first.get("evaluation_mode", "text_cot"),
                "intermediate_state_space": first.get("intermediate_state_space", ""),
                "terminal_state_space": first.get("terminal_state_space", ""),
                "proxy_target": first.get("proxy_target", ""),
                "num_latent_steps": first.get("num_latent_steps", ""),
                "count": len(rows),
                "mean_teacher_final_loss": np.nanmean([r["teacher_final_loss"] for r in rows]),
                "mean_generated_final_loss": np.nanmean([r["generated_final_loss"] for r in rows]),
                "mean_loss_gap": np.nanmean([r["loss_gap"] for r in rows]),
                "mean_generated_final_correct": np.nanmean([r["generated_final_correct"] for r in rows]),
                "mean_edit_distance": np.nanmean([r["generated_edit_distance"] for r in rows]),
                "mean_log_bound_proxy": np.nanmean([r["log_bound_proxy"] for r in rows]),
                "mean_max_local_log_amp": np.nanmean([r["max_local_log_amp"] for r in rows]),
                "mean_adjacent_jac_norm": np.nanmean([r["mean_adjacent_jac_norm"] for r in rows]),
                "mean_input_jac_norm": np.nanmean([r["input_jac_norm"] for r in rows]),
                "mean_log_input_jac_norm": np.nanmean([r["log_input_jac_norm"] for r in rows]),
            }
        )

    write_csv(
        os.path.join(output_dir, "length_summary.csv"),
        summary_rows,
        [
            "task",
            "length",
            "evaluation_mode",
            "intermediate_state_space",
            "terminal_state_space",
            "proxy_target",
            "num_latent_steps",
            "count",
            "mean_teacher_final_loss",
            "mean_generated_final_loss",
            "mean_loss_gap",
            "mean_generated_final_correct",
            "mean_edit_distance",
            "mean_log_bound_proxy",
            "mean_max_local_log_amp",
            "mean_adjacent_jac_norm",
            "mean_input_jac_norm",
            "mean_log_input_jac_norm",
        ],
    )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    finite_rows = [r for r in example_rows if np.isfinite(r["log_bound_proxy"]) and np.isfinite(r["loss_gap"])]
    if finite_rows:
        is_text_proxy = finite_rows[0].get("evaluation_mode") == "coconut_text_proxy"
        plt.figure()
        plt.scatter([r["log_bound_proxy"] for r in finite_rows], [r["loss_gap"] for r in finite_rows], s=14)
        plt.xlabel("log Coconut text-proxy bound" if is_text_proxy else "log bound proxy")
        plt.ylabel("generated final loss - teacher final loss")
        if is_text_proxy:
            plt.title("coconut_text_proxy")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_gap_vs_log_bound_proxy.png"), dpi=180)
        plt.close()

        plt.figure()
        plt.scatter(
            [r["mean_adjacent_jac_norm"] for r in finite_rows],
            [r["generated_final_correct"] for r in finite_rows],
            s=14,
        )
        plt.xlabel("mean adjacent Jacobian norm")
        plt.ylabel("generated final correct")
        if is_text_proxy:
            plt.title("coconut_text_proxy")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_vs_adjacent_jacobian.png"), dpi=180)
        plt.close()

    if summary_rows:
        plt.figure()
        for task in sorted({r["task"] for r in summary_rows}):
            task_rows = [r for r in summary_rows if r["task"] == task]
            plt.plot(
                [r["length"] for r in task_rows],
                [r["mean_max_local_log_amp"] for r in task_rows],
                marker="o",
                label=task,
            )
        plt.xlabel("CLRS length")
        plt.ylabel("mean max local log amplification")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "log_amplification_vs_length.png"), dpi=180)
        plt.close()

    first_example = None
    for row in jacobian_rows:
        if row.get("computed_full_jacobian"):
            first_example = (row["sample_idx"], row["task"])
            break
    if first_example is not None:
        rows = [r for r in jacobian_rows if (r["sample_idx"], r["task"]) == first_example]
        max_step = max(max(r["i"], r["j"]) for r in rows)
        heat = np.full((max_step, max_step), np.nan)
        for row in rows:
            heat[row["i"] - 1, row["j"] - 1] = row["log_jac_norm"]
        plt.figure()
        plt.imshow(heat, origin="lower", aspect="auto")
        plt.xlabel("source step j")
        plt.ylabel("target step i")
        plt.colorbar(label="log ||J_{i,j}||")
        if rows[0].get("evaluation_mode") == "coconut_text_proxy":
            plt.title("coconut_text_proxy latent direct partials")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "first_full_jacobian_heatmap.png"), dpi=180)
        plt.close()


def _normalise_coconut_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def resolve_decoder_backbone(model):
    """Return the decoder stack without the LM head, including through PEFT."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for attr_name in ("model", "transformer", "gpt_neox"):
        if hasattr(base, attr_name):
            return getattr(base, attr_name)
    raise RuntimeError("Could not resolve the decoder backbone for Coconut evaluation.")


def decoder_hidden_from_embeds(model, embeds: torch.Tensor) -> torch.Tensor:
    attention_mask = torch.ones(
        embeds.shape[:2], dtype=torch.long, device=embeds.device
    )
    output = resolve_decoder_backbone(model)(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return output.last_hidden_state


def build_coconut_proxy_sequence(
    model,
    tokenizer,
    raw_source: str,
    raw_answer: str,
    latent_count: int,
    max_source_length: int,
    max_output_length: int,
    device: torch.device,
) -> CoconutProxySequence:
    """Extract contextual hidden targets from the fully textual gold CoT.

    The source truncation and single-pass full-answer tokenization mirror training.
    Each proxy is the last-layer state of the final token intersecting that gold
    reasoning step, including a boundary-merged token when the tokenizer creates one.
    """
    reasoning_steps, final_answer = parse_coconut_answer(raw_answer)
    if len(reasoning_steps) < latent_count:
        raise ValueError(
            f"Coconut text-proxy evaluation needs at least {latent_count} gold reasoning "
            f"steps, but this sample has {len(reasoning_steps)}."
        )

    source_text = _normalise_coconut_text(raw_source)
    source_token_ids = tokenizer(source_text)["input_ids"][:max_source_length]
    if not source_token_ids:
        raise ValueError("Coconut source tokenization produced no tokens.")
    source_ids = torch.tensor(source_token_ids, dtype=torch.long, device=device)

    # This is the same truncate/decode/re-tokenize convention used by
    # CoconutMultitaskModel._plain_forward_for_target during the textual stage.
    textual_source = tokenizer.decode(source_token_ids, skip_special_tokens=True)
    text = textual_source + " "
    char_spans: List[Tuple[int, int]] = []
    for step_idx, step in enumerate(reasoning_steps):
        step = _normalise_coconut_text(step)
        start = len(text)
        text += step
        char_spans.append((start, len(text)))
        text += ", " if step_idx < len(reasoning_steps) - 1 else " | "
    text += _normalise_coconut_text(final_answer)

    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    full_ids = list(encoded["input_ids"])[: max_source_length + max_output_length]
    offsets = list(encoded["offset_mapping"])[: len(full_ids)]
    if not full_ids:
        raise ValueError("Coconut proxy tokenization produced no tokens.")

    proxy_positions: List[int] = []
    for proxy_idx, (char_start, char_end) in enumerate(char_spans[:latent_count], start=1):
        intersecting = [
            token_idx
            for token_idx, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        ]
        if not intersecting:
            raise ValueError(
                f"Gold reasoning step {proxy_idx} has no token after truncation; "
                "increase --max_output_length."
            )
        proxy_positions.append(intersecting[-1])

    ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        embeds = resolve_input_embeddings(model)(ids)
        hidden = decoder_hidden_from_embeds(model, embeds)
    proxy_latents = [
        hidden[:, position : position + 1, :].detach()
        for position in proxy_positions
    ]
    return CoconutProxySequence(
        source=textual_source,
        reasoning_steps=reasoning_steps,
        final_answer=_normalise_coconut_text(final_answer),
        proxy_latents=proxy_latents,
        source_ids=source_ids,
    )


def coconut_source_bot_embeds(
    input_embeddings: torch.nn.Module,
    source_ids: torch.Tensor,
    bot_token_id: int,
) -> torch.Tensor:
    bot = torch.tensor([bot_token_id], dtype=torch.long, device=source_ids.device)
    ids = torch.cat([source_ids, bot], dim=0).unsqueeze(0)
    return input_embeddings(ids).detach()


def coconut_next_latent(
    model,
    source_bot_embeds: torch.Tensor,
    prior_latents: Sequence[torch.Tensor],
) -> torch.Tensor:
    parts = [source_bot_embeds] + [latent.reshape(1, 1, -1) for latent in prior_latents]
    hidden = decoder_hidden_from_embeds(model, torch.cat(parts, dim=1))
    return hidden[:, -1:, :]


def coconut_terminal_hidden(
    model,
    input_embeddings: torch.nn.Module,
    source_bot_embeds: torch.Tensor,
    latents: Sequence[torch.Tensor],
    eot_token_id: int,
    target_ids: torch.Tensor,
    max_target_tokens: int = 0,
) -> torch.Tensor:
    """Pre-unembedding states which predict the teacher-forced answer tokens."""
    if max_target_tokens > 0:
        target_ids = target_ids[:max_target_tokens]
    if target_ids.numel() == 0:
        return torch.empty(0, device=source_bot_embeds.device)

    eot = torch.tensor([eot_token_id], dtype=torch.long, device=target_ids.device)
    latent_parts = [latent.reshape(1, 1, -1) for latent in latents]
    prefix = torch.cat([source_bot_embeds] + latent_parts, dim=1)
    target_embeds = input_embeddings(target_ids.unsqueeze(0))
    full_embeds = torch.cat([prefix, input_embeddings(eot.unsqueeze(0)), target_embeds], dim=1)
    hidden = decoder_hidden_from_embeds(model, full_embeds)
    target_start = prefix.shape[1] + 1
    positions = torch.arange(
        target_start - 1,
        target_start - 1 + target_ids.numel(),
        device=target_ids.device,
    )
    return hidden[0, positions, :]


def coconut_hidden_ce(
    output_embeddings: torch.nn.Module,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    if hidden.numel() == 0 or target_ids.numel() == 0:
        return float("nan")
    logits = logits_from_hidden(output_embeddings, hidden).float()
    return float(F.cross_entropy(logits, target_ids, reduction="mean").detach().cpu())


@torch.no_grad()
def coconut_greedy_answer(
    model,
    tokenizer,
    input_embeddings: torch.nn.Module,
    output_embeddings: torch.nn.Module,
    source_bot_embeds: torch.Tensor,
    free_latents: Sequence[torch.Tensor],
    eot_token_id: int,
    max_new_tokens: int,
) -> Tuple[str, List[int]]:
    eot = torch.tensor([eot_token_id], dtype=torch.long, device=source_bot_embeds.device)
    parts = [source_bot_embeds] + [latent.reshape(1, 1, -1) for latent in free_latents]
    embeds = torch.cat(parts + [input_embeddings(eot.unsqueeze(0))], dim=1)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        hidden = decoder_hidden_from_embeds(model, embeds)
        logits = logits_from_hidden(output_embeddings, hidden[:, -1, :])
        next_id = torch.argmax(logits, dim=-1)
        token_id = int(next_id.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        embeds = torch.cat([embeds, input_embeddings(next_id.unsqueeze(0))], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def make_coconut_latent_partial_fn(
    model,
    source_bot_embeds: torch.Tensor,
    proxy_latents: Sequence[torch.Tensor],
    target_step: int,
    source_step: int,
):
    """Direct partial map: other proxy latents stay fixed and detached."""
    fixed = [latent.detach() for latent in proxy_latents[: target_step - 1]]

    def fn(x: torch.Tensor) -> torch.Tensor:
        latents = list(fixed)
        latents[source_step - 1] = x.reshape(1, 1, -1)
        return coconut_next_latent(model, source_bot_embeds, latents).float().reshape(-1)

    return fn, proxy_latents[source_step - 1].detach().clone()


def make_coconut_terminal_partial_fn(
    model,
    input_embeddings: torch.nn.Module,
    source_bot_embeds: torch.Tensor,
    proxy_latents: Sequence[torch.Tensor],
    source_step: int,
    eot_token_id: int,
    target_ids: torch.Tensor,
    max_target_tokens: int,
):
    fixed = [latent.detach() for latent in proxy_latents]

    def fn(x: torch.Tensor) -> torch.Tensor:
        latents = list(fixed)
        latents[source_step - 1] = x.reshape(1, 1, -1)
        return coconut_terminal_hidden(
            model,
            input_embeddings,
            source_bot_embeds,
            latents,
            eot_token_id,
            target_ids,
            max_target_tokens,
        ).float().reshape(-1)

    return fn, proxy_latents[source_step - 1].detach().clone()


def write_csv_with_metadata(
    path: str,
    rows: List[Dict],
    required_fields: Sequence[str],
) -> None:
    fields = list(required_fields)
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    write_csv(path, rows, fields)


def coconut_output_dir(path: str) -> str:
    """Keep latent text-proxy runs separate from legacy Coconut outputs."""
    normalized = os.path.normpath(path)
    if normalized.endswith(os.path.join("coconut_text_proxy", "hidden_state")):
        return path
    if os.path.basename(normalized) == "coconut_text_proxy":
        return os.path.join(path, "hidden_state")
    return os.path.join(path, "coconut_text_proxy", "hidden_state")


def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    candidates = [path]
    candidates.append(os.path.join("external_lightning_logs", path))
    for candidate in candidates:
        if os.path.isdir(candidate):
            pts = sorted(
                os.path.join(candidate, name)
                for name in os.listdir(candidate)
                if name.startswith("epoch_epoch=") and name.endswith(".pt")
            )
            if pts:
                return pts[0]
        if os.path.exists(candidate):
            return candidate
    return path


def load_checkpoint(
    model,
    checkpoint_path: Optional[str],
    device: torch.device,
    use_coconut: bool = False,
) -> Optional[str]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if checkpoint_path is None:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        checkpoint = {k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()}

    if use_coconut:
        compact_id_key = "__coconut_special_token_ids__"
        compact_row_suffix = ".__coconut_special_token_rows__"
        compact_row_keys = [
            key for key in checkpoint if key.endswith(compact_row_suffix)
        ]
        if compact_id_key in checkpoint and not compact_row_keys:
            raise RuntimeError(
                "Coconut checkpoint contains compact special-token ids but no "
                "compact embedding or LM-head rows."
            )
        if compact_row_keys and compact_id_key not in checkpoint:
            raise RuntimeError(
                "Coconut checkpoint contains compact special-token rows but no "
                f"{compact_id_key}."
            )
        model_keys = set(model.state_dict())
        unconsumable = [
            key
            for key in compact_row_keys
            if key[: -len(compact_row_suffix)] not in model_keys
        ]
        if unconsumable:
            raise RuntimeError(
                "Coconut checkpoint special-token rows do not match this model: "
                + ", ".join(unconsumable[:5])
            )
        checkpoint = expand_coconut_special_token_rows(checkpoint, model)
        compact_keys = [
            key
            for key in checkpoint
            if key == compact_id_key or key.endswith(compact_row_suffix)
        ]
        if compact_keys:
            raise RuntimeError(
                "Coconut checkpoint expansion left compact keys unconsumed: "
                + ", ".join(compact_keys[:5])
            )

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if use_coconut and unexpected:
        raise RuntimeError(
            "Coconut checkpoint has unexpected keys after expansion: "
            + ", ".join(unexpected[:10])
        )
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    model.to(device)
    return checkpoint_path


def evaluate_coconut(args) -> None:
    """Evaluate Coconut's latent recurrence with contextual-text proxy targets."""
    validate_quant_args(args)
    if args.coconut_latents_per_step != 1:
        raise ValueError(
            "Coconut text-proxy Jacobians require --coconut_latents_per_step 1; "
            "there is no one-to-one text proxy for multiple latents per step."
        )
    if args.coconut_eval_latent_thoughts < 1:
        raise ValueError(
            "Coconut text-proxy Jacobians require --coconut_eval_latent_thoughts >= 1."
        )
    if args.jacobian_granularity != "block":
        raise ValueError("Coconut latent Jacobians currently require --jacobian_granularity block.")
    if args.jacobian_method != "jvp":
        raise ValueError("Coconut latent Jacobians require --jacobian_method jvp.")
    if args.second_order_checks > 0 and args.second_order_pairs != "all":
        raise ValueError(
            "Coconut's conservative second-order bound requires "
            "--second_order_pairs all."
        )
    if args.jacobian_output not in (None, "hidden_state"):
        raise ValueError(
            "Coconut latent evaluation uses a hidden-state terminal. Pass "
            "--jacobian_output hidden_state or omit --jacobian_output."
        )
    if args.embedding_final_state != "hidden_state":
        raise ValueError(
            "Coconut latent evaluation requires --embedding_final_state hidden_state."
        )
    args.jacobian_output = "hidden_state"
    args.embedding_final_state = "hidden_state"
    args.output_dir = coconut_output_dir(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    args.devices = [args.device]
    model, tokenizer, _, model_type, _ = initialize_model(args)
    if model_type != "decoder":
        raise ValueError("Coconut latent evaluation requires a decoder-only model.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint_path = load_checkpoint(
        model, args.checkpoint_path, device, use_coconut=True
    )
    model.to(device).eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    bot_token_id = tokenizer.convert_tokens_to_ids(args.coconut_bot_token)
    eot_token_id = tokenizer.convert_tokens_to_ids(args.coconut_eot_token)
    if bot_token_id == tokenizer.unk_token_id or eot_token_id == tokenizer.unk_token_id:
        raise ValueError("Coconut <bot>/<eot> tokens are not registered in the tokenizer.")

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
        only_answer_output=True,
        reduce_steps_ratio=1.0,
        reduce_steps_equally_spaced=False,
    )
    data_module.setup(stage="fit")
    if args.split == "train":
        dataloader = data_module.train_dataloader()
    elif args.split == "test":
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()

    input_embeddings = resolve_input_embeddings(model)
    output_embeddings = resolve_output_embeddings(model)
    global_C = hidden_ce_global_lipschitz(output_embeddings)
    latent_count = args.coconut_eval_latent_thoughts
    terminal_step = latent_count + 1

    example_rows: List[Dict] = []
    jacobian_rows: List[Dict] = []
    terminal_jacobian_rows: List[Dict] = []
    generated_step_loss_rows: List[Dict] = []
    rho_prefix_rows: List[Dict] = []
    second_order_rows: List[Dict] = []
    finite_difference_rows: List[Dict] = []
    processed = 0
    finite_difference_count = 0
    progress = tqdm(total=args.max_examples, desc="Evaluating Coconut latent Jacobians")

    for batch in dataloader:
        task_name = batch["task_name"]
        data = batch["data"]
        if "raw_inputs" not in data or "raw_answers" not in data:
            raise KeyError(
                "Coconut Jacobian evaluation requires raw_inputs and raw_answers from the collator."
            )
        batch_size = data["input_ids"].shape[0]
        for batch_idx in range(batch_size):
            if processed >= args.max_examples:
                break
            sample_idx = (
                int(data["sample_idx"][batch_idx])
                if "sample_idx" in data
                else processed
            )
            length = int(data["length"][batch_idx]) if "length" in data else -1
            raw_source = data["raw_inputs"][batch_idx]
            raw_answer = data["raw_answers"][batch_idx]
            proxy = build_coconut_proxy_sequence(
                model,
                tokenizer,
                raw_source,
                raw_answer,
                latent_count,
                args.max_length,
                args.max_output_length,
                device,
            )
            source_bot = coconut_source_bot_embeds(
                input_embeddings, proxy.source_ids, bot_token_id
            )

            with torch.no_grad():
                proxy_predictions: List[torch.Tensor] = []
                free_latents: List[torch.Tensor] = []
                for latent_idx in range(latent_count):
                    proxy_predictions.append(
                        coconut_next_latent(
                            model, source_bot, proxy.proxy_latents[:latent_idx]
                        ).detach()
                    )
                    free_latents.append(
                        coconut_next_latent(model, source_bot, free_latents).detach()
                    )

                train_errors = [
                    float((pred.float() - target.float()).norm().cpu())
                    for pred, target in zip(proxy_predictions, proxy.proxy_latents)
                ]
                test_errors = [
                    float((free.float() - target.float()).norm().cpu())
                    for free, target in zip(free_latents, proxy.proxy_latents)
                ]

                target_token_ids = tokenizer(proxy.final_answer)["input_ids"][
                    : args.max_output_length
                ]
                if not target_token_ids:
                    raise ValueError("Coconut final answer tokenization produced no tokens.")
                target_ids = torch.tensor(
                    target_token_ids, dtype=torch.long, device=device
                )
                capped_ids = (
                    target_ids[: args.max_target_tokens_for_jacobian]
                    if args.max_target_tokens_for_jacobian > 0
                    else target_ids
                )
                proxy_terminal_full = coconut_terminal_hidden(
                    model,
                    input_embeddings,
                    source_bot,
                    proxy.proxy_latents,
                    eot_token_id,
                    target_ids,
                )
                free_terminal_full = coconut_terminal_hidden(
                    model,
                    input_embeddings,
                    source_bot,
                    free_latents,
                    eot_token_id,
                    target_ids,
                )
                proxy_terminal = proxy_terminal_full[: capped_ids.numel()]
                free_terminal = free_terminal_full[: capped_ids.numel()]
                teacher_final_loss = coconut_hidden_ce(
                    output_embeddings, proxy_terminal_full, target_ids
                )
                generated_final_loss = coconut_hidden_ce(
                    output_embeddings, free_terminal_full, target_ids
                )
                teacher_final_loss_capped = coconut_hidden_ce(
                    output_embeddings, proxy_terminal, capped_ids
                )
                generated_final_loss_capped = coconut_hidden_ce(
                    output_embeddings, free_terminal, capped_ids
                )
                final_state_delta_norm = float(
                    (free_terminal.float() - proxy_terminal.float()).norm().cpu()
                )
                empirical_C = empirical_lipschitz_constant(
                    teacher_final_loss_capped,
                    generated_final_loss_capped,
                    final_state_delta_norm,
                )
                generated_answer, _ = coconut_greedy_answer(
                    model,
                    tokenizer,
                    input_embeddings,
                    output_embeddings,
                    source_bot,
                    free_latents,
                    eot_token_id,
                    args.generation_max_new_tokens or args.max_output_length,
                )

            final_metrics = compute_accuracy(
                [generated_answer], [[proxy.final_answer]]
            )
            generated_final_correct = final_metrics["accuracy"] / 100.0
            generated_edit_distance = final_metrics["edit_distance"]

            common = {
                "task": task_name,
                "checkpoint": checkpoint_path or "",
                "jacobian_output": "hidden_state",
                "jacobian_granularity": "block",
                "embedding_final_state": "hidden_state",
                "evaluation_mode": "coconut_text_proxy",
                "intermediate_state_space": "latent_hidden",
                "terminal_state_space": "hidden_state",
                "proxy_target": "contextual_gold_step_last_hidden",
                "split": args.split,
                "length": length,
                "sample_idx": sample_idx,
                "num_steps": terminal_step,
                "num_gold_steps": len(proxy.reasoning_steps) + 1,
                "num_latent_steps": latent_count,
                "final_is_gold_final": True,
            }

            latent_jacobians: Dict[Tuple[int, int], float] = {}
            terminal_jacobians: Dict[Tuple[int, int], float] = {}
            second_order_by_target: Dict[int, List[Tuple[float, float]]] = {}
            with attention_kernel_context(args.attention_backend):
                for target_step in range(2, latent_count + 1):
                    for source_step in range(1, target_step):
                        fn, x0 = make_coconut_latent_partial_fn(
                            model,
                            source_bot,
                            proxy.proxy_latents,
                            target_step,
                            source_step,
                        )
                        jac_norm = estimate_function_spectral_norm_jvp(
                            fn, x0, args.power_iters
                        )
                        latent_jacobians[(target_step, source_step)] = jac_norm
                        if args.second_order_checks > 0:
                            second = estimate_function_jacobian_lipschitz_by_jvp_difference(
                                fn,
                                x0,
                                args.second_order_epsilon,
                                args.second_order_checks,
                            )
                            if np.isfinite(second["M_random_max"]):
                                second_order_by_target.setdefault(target_step, []).append(
                                    (second["M_random_max"], second["M_random_mean"])
                                )
                            second_order_rows.append(
                                {
                                    **common,
                                    "i": target_step,
                                    "j": source_step,
                                    "map_type": "latent",
                                    **second,
                                    "second_order_checks": args.second_order_checks,
                                    "second_order_epsilon": args.second_order_epsilon,
                                    "second_order_pairs": args.second_order_pairs,
                                }
                            )
                        if finite_difference_count < args.finite_difference_checks:
                            check = finite_difference_function_check(
                                fn, x0, args.finite_difference_epsilon
                            )
                            finite_difference_rows.append(
                                {**common, "i": target_step, "j": source_step, **check}
                            )
                            finite_difference_count += 1

                for source_step in range(1, latent_count + 1):
                    fn, x0 = make_coconut_terminal_partial_fn(
                        model,
                        input_embeddings,
                        source_bot,
                        proxy.proxy_latents,
                        source_step,
                        eot_token_id,
                        target_ids,
                        args.max_target_tokens_for_jacobian,
                    )
                    jac_norm = estimate_function_spectral_norm_jvp(
                        fn, x0, args.power_iters
                    )
                    terminal_jacobians[(terminal_step, source_step)] = jac_norm
                    if args.second_order_checks > 0:
                        second = estimate_function_jacobian_lipschitz_by_jvp_difference(
                            fn,
                            x0,
                            args.second_order_epsilon,
                            args.second_order_checks,
                        )
                        if np.isfinite(second["M_random_max"]):
                            second_order_by_target.setdefault(terminal_step, []).append(
                                (second["M_random_max"], second["M_random_mean"])
                            )
                        second_order_rows.append(
                            {
                                **common,
                                "i": terminal_step,
                                "j": source_step,
                                "map_type": "terminal",
                                **second,
                                "second_order_checks": args.second_order_checks,
                                "second_order_epsilon": args.second_order_epsilon,
                                "second_order_pairs": args.second_order_pairs,
                            }
                        )

            intermediate_rho = compute_rho(latent_jacobians, latent_count)
            terminal_rho_all = compute_terminal_rho(
                intermediate_rho, terminal_jacobians, terminal_step
            )
            terminal_rho = {
                key: value
                for key, value in terminal_rho_all.items()
                if key[0] == terminal_step
            }
            effective_rho = dict(intermediate_rho)
            effective_rho.update(terminal_rho)

            M_prefix_max: Dict[int, float] = {}
            M_prefix_mean: Dict[int, float] = {}
            for prefix_step in range(1, terminal_step + 1):
                values = [
                    value
                    for target, pairs in second_order_by_target.items()
                    if target <= prefix_step
                    for value, _ in pairs
                    if np.isfinite(value)
                ]
                means = [
                    mean_value
                    for target, pairs in second_order_by_target.items()
                    if target <= prefix_step
                    for _, mean_value in pairs
                    if np.isfinite(mean_value)
                ]
                M_prefix_max[prefix_step] = max(values) if values else float("nan")
                M_prefix_mean[prefix_step] = (
                    float(np.mean(means)) if means else float("nan")
                )
            second_stats = compute_second_order_prefix_stats(
                effective_rho,
                test_errors + [float("nan")],
                M_prefix_max,
                M_prefix_mean,
                terminal_step,
            )

            bound_by_step: Dict[int, float] = {}
            for target_step in range(1, terminal_step + 1):
                rho_map = terminal_rho if target_step == terminal_step else intermediate_rho
                bound_by_step[target_step] = sum(
                    rho_map.get((target_step, source_step), 0.0)
                    * train_errors[source_step - 1]
                    for source_step in range(1, target_step)
                    if source_step <= latent_count
                )

            for (target_step, source_step), jac_norm in sorted(latent_jacobians.items()):
                jacobian_rows.append(
                    {
                        **common,
                        "i": target_step,
                        "j": source_step,
                        "jac_norm": jac_norm,
                        "log_jac_norm": (
                            math.log(max(jac_norm, EPS))
                            if np.isfinite(jac_norm)
                            else float("nan")
                        ),
                        "rho_T_i": terminal_rho.get(
                            (terminal_step, source_step), float("nan")
                        ),
                        "local_log_amp_i": float("nan"),
                        "step_loss_i": float("nan"),
                        "train_error_i": train_errors[source_step - 1],
                        "train_error_type": "contextual_hidden_l2_proxy",
                        "teacher_final_loss": teacher_final_loss,
                        "generated_final_loss": generated_final_loss,
                        "generated_final_correct": generated_final_correct,
                        "computed_full_jacobian": True,
                    }
                )
            for (target_step, source_step), jac_norm in sorted(terminal_jacobians.items()):
                terminal_jacobian_rows.append(
                    {
                        **common,
                        "i": target_step,
                        "j": source_step,
                        "terminal_jac_norm": jac_norm,
                        "terminal_log_jac_norm": (
                            math.log(max(jac_norm, EPS))
                            if np.isfinite(jac_norm)
                            else float("nan")
                        ),
                        "terminal_rho_T_i": terminal_rho.get(
                            (terminal_step, source_step), float("nan")
                        ),
                        "train_error_i": train_errors[source_step - 1],
                        "train_error_type": "contextual_hidden_l2_proxy",
                        "computed_full_jacobian": True,
                    }
                )

            for target_step in range(1, terminal_step + 1):
                if target_step == 1:
                    adjacent_rho = recursive_rho = 1.0
                else:
                    recursive_rho = effective_rho.get(
                        (target_step, 1), float("nan")
                    )
                    adjacent_rho = 1.0
                    for step in range(2, target_step + 1):
                        if step == terminal_step:
                            factor = terminal_jacobians.get(
                                (terminal_step, step - 1), float("nan")
                            )
                        else:
                            factor = latent_jacobians.get(
                                (step, step - 1), float("nan")
                            )
                        adjacent_rho *= factor
                rho_prefix_rows.append(
                    {
                        **common,
                        "t": target_step,
                        "adjacent_rho_t_1": adjacent_rho,
                        "adjacent_log_rho_t_1": (
                            math.log(max(adjacent_rho, EPS))
                            if np.isfinite(adjacent_rho)
                            else float("nan")
                        ),
                        "recursive_rho_t_1": recursive_rho,
                        "recursive_log_rho_t_1": (
                            math.log(max(recursive_rho, EPS))
                            if np.isfinite(recursive_rho)
                            else float("nan")
                        ),
                        "embedding_adjacent_rho_t_1": adjacent_rho,
                        "embedding_adjacent_log_rho_t_1": (
                            math.log(max(adjacent_rho, EPS))
                            if np.isfinite(adjacent_rho)
                            else float("nan")
                        ),
                        "embedding_recursive_rho_t_1": recursive_rho,
                        "embedding_recursive_log_rho_t_1": (
                            math.log(max(recursive_rho, EPS))
                            if np.isfinite(recursive_rho)
                            else float("nan")
                        ),
                        "computed_full_jacobian": True,
                    }
                )

                stats = second_stats[target_step]
                raw_bound = bound_by_step[target_step]
                is_terminal = target_step == terminal_step
                first_bound = global_C * raw_bound if is_terminal else float("nan")
                second_amount = stats["second_order_geometry_amount_t"]
                second_bound = (
                    global_C * second_amount
                    if is_terminal and np.isfinite(second_amount)
                    else float("nan")
                )
                if is_terminal:
                    row_teacher_loss = teacher_final_loss
                    row_generated_loss = generated_final_loss
                    row_teacher_capped = teacher_final_loss_capped
                    row_generated_capped = generated_final_loss_capped
                    row_delta = final_state_delta_norm
                    row_test_error = float("nan")
                    row_empirical_C = empirical_C
                    row_global_C = global_C
                    gold_step = proxy.final_answer
                    generated_step = generated_answer
                    step_correct = generated_final_correct
                    step_edit = generated_edit_distance
                else:
                    row_teacher_loss = row_generated_loss = float("nan")
                    row_teacher_capped = row_generated_capped = float("nan")
                    row_delta = float(
                        (
                            free_latents[target_step - 1].float()
                            - proxy_predictions[target_step - 1].float()
                        )
                        .norm()
                        .cpu()
                    )
                    row_test_error = test_errors[target_step - 1]
                    row_empirical_C = row_global_C = float("nan")
                    gold_step = proxy.reasoning_steps[target_step - 1]
                    generated_step = "<latent>"
                    step_correct = step_edit = float("nan")

                generated_step_loss_rows.append(
                    {
                        **common,
                        "t": target_step,
                        "state_space": "hidden_state" if is_terminal else "latent_hidden",
                        "teacher_step_loss": row_teacher_loss,
                        "generated_prefix_step_loss": row_generated_loss,
                        "step_loss_gap": row_generated_loss - row_teacher_loss,
                        "teacher_step_loss_capped": row_teacher_capped,
                        "generated_step_loss_capped": row_generated_capped,
                        "step_loss_gap_capped": row_generated_capped - row_teacher_capped,
                        "state_delta_norm": row_delta,
                        "test_error_norm": row_test_error,
                        "test_error_square": (
                            row_test_error ** 2
                            if np.isfinite(row_test_error)
                            else float("nan")
                        ),
                        "empirical_lipschitz_C": row_empirical_C,
                        "C_segment": float("nan"),
                        "C_point_teacher": float("nan"),
                        "C_random_max": float("nan"),
                        "C_random_mean": float("nan"),
                        "C_global": row_global_C,
                        "input_sensitivity_noise_max": float("nan"),
                        "input_sensitivity_noise_mean": float("nan"),
                        "sum_rho_t_i": sum(
                            effective_rho.get((target_step, source), 0.0)
                            for source in range(1, target_step)
                        ),
                        "bound_proxy_t": raw_bound,
                        **stats,
                        "segment_calibrated_bound_t": float("nan"),
                        "random_max_calibrated_bound_t": float("nan"),
                        "global_calibrated_bound_t": first_bound,
                        "pred_calibrated_bound_t": first_bound,
                        "segment_second_order_bound_t": float("nan"),
                        "random_max_second_order_bound_t": float("nan"),
                        "global_second_order_bound_t": second_bound,
                        "segment_total_bound_t": float("nan"),
                        "random_max_total_bound_t": float("nan"),
                        "global_total_bound_t": (
                            first_bound + second_bound
                            if np.isfinite(first_bound) and np.isfinite(second_bound)
                            else float("nan")
                        ),
                        "generated_step_correct": step_correct,
                        "generated_step_edit_distance": step_edit,
                        "gold_step": gold_step,
                        "generated_step": generated_step,
                    }
                )

            final_stats = second_stats[terminal_step]
            raw_bound = bound_by_step[terminal_step]
            first_bound = global_C * raw_bound
            second_amount = final_stats["second_order_geometry_amount_t"]
            second_bound = (
                global_C * second_amount
                if np.isfinite(second_amount)
                else float("nan")
            )
            example_rows.append(
                {
                    **common,
                    "teacher_final_loss": teacher_final_loss,
                    "generated_final_loss": generated_final_loss,
                    "loss_gap": generated_final_loss - teacher_final_loss,
                    "generated_final_correct": generated_final_correct,
                    "generated_edit_distance": generated_edit_distance,
                    "bound_proxy": raw_bound,
                    "log_bound_proxy": math.log(max(raw_bound, EPS)),
                    "bound_proxy_valid": True,
                    "embedding_bound_proxy": raw_bound,
                    "embedding_log_bound_proxy": math.log(max(raw_bound, EPS)),
                    "calibrated_bound_proxy": first_bound,
                    "segment_calibrated_bound_proxy": float("nan"),
                    "random_max_calibrated_bound_proxy": float("nan"),
                    "global_calibrated_bound_proxy": first_bound,
                    "pred_calibrated_bound_proxy": first_bound,
                    "M_random_max": final_stats["M_random_max_t"],
                    "M_random_mean": final_stats["M_random_mean_t"],
                    "gamma_T": final_stats["gamma_t"],
                    "rho_remainder_multiplier": final_stats[
                        "rho_remainder_multiplier_t"
                    ],
                    "test_error_square_sum": final_stats["test_error_square_sum_t"],
                    "second_order_weighted_test_error_square_sum": final_stats[
                        "second_order_weighted_test_error_square_sum_t"
                    ],
                    "second_order_geometry_coefficient": final_stats[
                        "second_order_geometry_coefficient_t"
                    ],
                    "second_order_geometry_amount": second_amount,
                    "segment_second_order_bound": float("nan"),
                    "random_max_second_order_bound": float("nan"),
                    "global_second_order_bound": second_bound,
                    "segment_total_bound": float("nan"),
                    "random_max_total_bound": float("nan"),
                    "global_total_bound": (
                        first_bound + second_bound
                        if np.isfinite(second_bound)
                        else float("nan")
                    ),
                    "state_space": "hidden_state",
                    "seed": args.seed,
                    "source_truncated": False,
                    "final_state_delta_norm": final_state_delta_norm,
                    "empirical_lipschitz_C": empirical_C,
                    "C_segment": float("nan"),
                    "C_point_teacher": float("nan"),
                    "C_random_max": float("nan"),
                    "C_random_mean": float("nan"),
                    "C_global": global_C,
                    "input_sensitivity_noise_max": float("nan"),
                    "input_sensitivity_noise_mean": float("nan"),
                    "max_local_log_amp": float("nan"),
                    "mean_adjacent_jac_norm": float(
                        np.mean(
                            [
                                latent_jacobians[(step, step - 1)]
                                for step in range(2, latent_count + 1)
                            ]
                        )
                    )
                    if latent_count > 1
                    else float("nan"),
                    "mean_train_error": float(np.mean(train_errors)),
                    "train_error_type": "contextual_hidden_l2_proxy",
                    "input_jac_norm": float("nan"),
                    "log_input_jac_norm": float("nan"),
                    "gold_final": proxy.final_answer,
                    "pred_final": generated_answer,
                    "raw_gold_answer": _normalise_coconut_text(raw_answer),
                    "generated_answer": _normalise_coconut_text(generated_answer),
                }
            )
            processed += 1
            progress.update(1)

        if processed >= args.max_examples:
            break

    progress.close()
    metadata_fields = [
        "evaluation_mode",
        "intermediate_state_space",
        "terminal_state_space",
        "proxy_target",
        "num_latent_steps",
    ]
    write_csv_with_metadata(
        os.path.join(args.output_dir, "jacobian_rows.csv"),
        jacobian_rows,
        ["task", "checkpoint", "jacobian_output", "jacobian_granularity", "embedding_final_state"]
        + metadata_fields
        + [
            "split", "length", "sample_idx", "i", "j", "num_steps", "num_gold_steps",
            "final_is_gold_final", "jac_norm", "log_jac_norm", "rho_T_i", "local_log_amp_i",
            "step_loss_i", "train_error_i", "train_error_type", "teacher_final_loss",
            "generated_final_loss", "generated_final_correct", "computed_full_jacobian",
        ],
    )
    write_csv_with_metadata(
        os.path.join(args.output_dir, "terminal_jacobian_rows.csv"),
        terminal_jacobian_rows,
        ["task", "checkpoint", "jacobian_output", "jacobian_granularity", "embedding_final_state"]
        + metadata_fields
        + [
            "split", "length", "sample_idx", "i", "j", "num_steps", "num_gold_steps",
            "final_is_gold_final", "terminal_jac_norm", "terminal_log_jac_norm",
            "terminal_rho_T_i", "train_error_i", "train_error_type", "computed_full_jacobian",
        ],
    )
    write_csv_with_metadata(
        os.path.join(args.output_dir, "second_order_rows.csv"),
        second_order_rows,
        ["task", "checkpoint", "jacobian_output", "jacobian_granularity", "embedding_final_state"]
        + metadata_fields
        + [
            "split", "length", "sample_idx", "i", "j", "num_steps", "num_gold_steps",
            "final_is_gold_final", "map_type", "M_random_max", "M_random_mean",
            "second_order_checks", "second_order_epsilon", "second_order_pairs",
        ],
    )
    write_csv_with_metadata(
        os.path.join(args.output_dir, "rho_prefix_rows.csv"),
        rho_prefix_rows,
        ["task", "checkpoint", "jacobian_output", "jacobian_granularity", "embedding_final_state"]
        + metadata_fields
        + [
            "split", "length", "sample_idx", "t", "num_steps", "num_gold_steps",
            "final_is_gold_final", "adjacent_rho_t_1", "adjacent_log_rho_t_1",
            "recursive_rho_t_1", "recursive_log_rho_t_1", "embedding_adjacent_rho_t_1",
            "embedding_adjacent_log_rho_t_1", "embedding_recursive_rho_t_1",
            "embedding_recursive_log_rho_t_1", "computed_full_jacobian",
        ],
    )
    write_csv_with_metadata(
        os.path.join(args.output_dir, "generated_step_loss_rows.csv"),
        generated_step_loss_rows,
        ["task", "checkpoint", "jacobian_output", "jacobian_granularity", "embedding_final_state"]
        + metadata_fields
        + [
            "split", "length", "sample_idx", "t", "num_steps", "num_gold_steps",
            "final_is_gold_final", "state_space", "teacher_step_loss",
            "generated_prefix_step_loss", "step_loss_gap", "teacher_step_loss_capped",
            "generated_step_loss_capped", "step_loss_gap_capped", "state_delta_norm",
            "test_error_norm", "test_error_square", "empirical_lipschitz_C", "C_segment",
            "C_point_teacher", "C_random_max", "C_random_mean", "C_global",
            "input_sensitivity_noise_max", "input_sensitivity_noise_mean", "sum_rho_t_i",
            "bound_proxy_t", "M_random_max_t", "M_random_mean_t", "gamma_t",
            "rho_remainder_multiplier_t", "test_error_square_sum_t",
            "second_order_weighted_test_error_square_sum_t",
            "second_order_geometry_coefficient_t", "second_order_geometry_amount_t",
            "segment_calibrated_bound_t", "random_max_calibrated_bound_t",
            "global_calibrated_bound_t", "pred_calibrated_bound_t",
            "segment_second_order_bound_t", "random_max_second_order_bound_t",
            "global_second_order_bound_t", "segment_total_bound_t", "random_max_total_bound_t",
            "global_total_bound_t", "generated_step_correct", "generated_step_edit_distance",
            "gold_step", "generated_step",
        ],
    )
    write_csv_with_metadata(
        os.path.join(args.output_dir, "example_metrics.csv"),
        example_rows,
        list(example_rows[0].keys()) if example_rows else ["task"] + metadata_fields,
    )
    write_jsonl(os.path.join(args.output_dir, "example_metrics.jsonl"), example_rows)
    write_csv_with_metadata(
        os.path.join(args.output_dir, "jacobian_token_rows.csv"),
        [],
        ["task", "checkpoint"] + metadata_fields + ["i", "j"],
    )
    if finite_difference_rows:
        write_csv_with_metadata(
            os.path.join(args.output_dir, "finite_difference_checks.csv"),
            finite_difference_rows,
            list(finite_difference_rows[0].keys()),
        )
    write_analysis_outputs(args.output_dir, example_rows, jacobian_rows)
    print(f"Processed {processed} Coconut examples.")
    print(f"Wrote Coconut text-proxy outputs to {args.output_dir}")


def evaluate(args) -> None:
    if args.use_coconut:
        evaluate_coconut(args)
        return
    if args.jacobian_output is None:
        args.jacobian_output = "token_distribution"
    if args.jacobian_output == "hidden_state":
        raise ValueError(
            "--jacobian_output hidden_state is only valid with --use_coconut."
        )
    validate_quant_args(args)
    if args.jacobian_granularity == "token_mean" and args.jacobian_output == "selected_logprob":
        raise ValueError("--jacobian_granularity token_mean requires expected_embedding or token_distribution")
    embedding_final_state_explicit = "--embedding_final_state" in sys.argv
    if (
        embedding_final_state_explicit
        and args.embedding_final_state == "hidden_state"
        and args.jacobian_output != "expected_embedding"
    ):
        raise ValueError("--embedding_final_state hidden_state is only valid with --jacobian_output expected_embedding")
    effective_embedding_final_state = (
        args.embedding_final_state if args.jacobian_output == "expected_embedding" else "embedding"
    )
    hidden_terminal = args.jacobian_output == "expected_embedding" and effective_embedding_final_state == "hidden_state"
    os.makedirs(args.output_dir, exist_ok=True)

    # The data module picks the retained gold steps with np.random.choice whenever
    # reduce_steps_ratio < 1, so without a seed each invocation evaluates a
    # different subsequence and the numbers are not reproducible.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    args.devices = [args.device]
    model, tokenizer, _, _, _ = initialize_model(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint_path = load_checkpoint(
        model, args.checkpoint_path, device, use_coconut=args.use_coconut
    )
    model.to(device)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

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
        only_answer_output=args.only_answer_output,
        reduce_steps_ratio=args.reduce_steps_ratio,
        reduce_steps_equally_spaced=args.reduce_steps_equally_spaced,
    )
    data_module.setup(stage="fit")
    if args.split == "train":
        dataloader = data_module.train_dataloader()
    elif args.split == "test":
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()

    input_embeddings = resolve_input_embeddings(model)
    output_embeddings = resolve_output_embeddings(model)
    hidden_ce_global_C = hidden_ce_global_lipschitz(output_embeddings) if hidden_terminal else float("nan")

    example_rows: List[Dict] = []
    jacobian_rows: List[Dict] = []
    terminal_jacobian_rows: List[Dict] = []
    jacobian_token_rows: List[Dict] = []
    rho_prefix_rows: List[Dict] = []
    generated_step_loss_rows: List[Dict] = []
    finite_difference_rows: List[Dict] = []
    second_order_rows: List[Dict] = []
    processed = 0
    finite_difference_count = 0

    progress = tqdm(total=args.max_examples, desc="Evaluating CoT Jacobians")
    for batch in dataloader:
        task_name = batch["task_name"]
        data = batch["data"]
        batch_size = data["input_ids"].shape[0]
        for b in range(batch_size):
            if processed >= args.max_examples:
                break

            sample_idx = int(data["sample_idx"][b]) if "sample_idx" in data else processed
            length = int(data["length"][b]) if "length" in data else -1
            seq = decode_sample(
                tokenizer,
                data["input_ids"][b].cpu(),
                data["labels"][b].cpu(),
                data["attention_mask"][b].cpu(),
            )
            raw_input = data["raw_inputs"][b] if "raw_inputs" in data else None
            check_source_not_truncated(raw_input, seq.source, args.max_length)

            if len(seq.steps) < args.min_jacobian_steps:
                continue
            if len(seq.steps) == 0:
                continue

            # The gold sequence is never truncated: its length fixes where the
            # " | " final-answer marker goes. max_jacobian_steps only limits how
            # many prefixes we analyse, so a capped horizon cannot make an
            # intermediate step look like the final answer.
            num_gold_steps = len(seq.steps)
            total_steps = (
                min(num_gold_steps, args.max_jacobian_steps)
                if args.max_jacobian_steps > 0
                else num_gold_steps
            )
            final_is_gold_final = total_steps == num_gold_steps

            teacher_encodings: Dict[int, Optional[StepEncoding]] = {
                target_step: build_encoding(
                    tokenizer,
                    seq.source,
                    seq.steps,
                    seq.steps[target_step - 1],
                    target_step,
                    num_gold_steps,
                    device,
                )
                for target_step in range(1, total_steps + 1)
            }
            step_dims = compute_step_dims(
                teacher_encodings, total_steps, args.max_target_tokens_for_jacobian
            )

            def step_dim(step: int) -> int:
                if step <= 0:
                    return args.max_target_tokens_for_jacobian
                return step_dims.get(step, args.max_target_tokens_for_jacobian)

            with torch.no_grad():
                step_losses = teacher_forced_step_losses(model, teacher_encodings, total_steps)
            teacher_final_loss = metric_float(step_losses[-1])
            if args.jacobian_output in ("expected_embedding", "token_distribution"):
                train_error_type = (
                    "prob_l2" if args.jacobian_output == "token_distribution" else "expected_embedding_l2"
                )
                train_errors = [
                    metric_float(
                        state_step_error(
                            model,
                            input_embeddings,
                            teacher_encodings.get(target_step),
                            step_dim(target_step),
                            args.jacobian_output,
                            args.jacobian_granularity,
                        )
                    )
                    for target_step in range(1, total_steps + 1)
                ]
            else:
                train_error_type = "sqrt_ce_diagnostic"
                train_errors = [
                    math.sqrt(max(loss, 0.0)) if loss is not None and np.isfinite(loss) else float("nan")
                    for loss in step_losses
                ]

            generated_answer = generate_answer(
                model,
                tokenizer,
                seq.source,
                device,
                max_new_tokens=args.generation_max_new_tokens or args.max_output_length,
            )
            generated_steps, pred_final, _ = split_generated_answer(generated_answer)
            generated_sequence_steps = list(generated_steps) + ([pred_final] if pred_final else [])
            gold_final = seq.steps[total_steps - 1]
            teacher_final_enc = teacher_encodings.get(total_steps)
            generated_final_enc = build_encoding(
                tokenizer,
                seq.source,
                generated_sequence_steps,
                gold_final,
                total_steps,
                num_gold_steps,
                device,
            )
            generated_final_loss = metric_float(step_ce_loss(model, generated_final_enc))
            final_conditioning_step = total_steps - 1 if total_steps > 1 else 0
            terminal_state_output = "hidden_state" if hidden_terminal else args.jacobian_output
            final_ce_lipschitz_span = None
            if teacher_final_enc is not None and final_conditioning_step in teacher_final_enc.spans:
                final_ce_lipschitz_span = teacher_final_enc.spans[final_conditioning_step]
                if args.jacobian_output != "selected_logprob" and final_conditioning_step > 0:
                    final_ce_lipschitz_span = cap_step_span(
                        final_ce_lipschitz_span,
                        step_dim(final_conditioning_step),
                    )
            final_ce_lipschitz_noise = estimate_ce_embedding_lipschitz_by_noise(
                model,
                input_embeddings,
                teacher_final_enc,
                final_ce_lipschitz_span,
                args.ce_lipschitz_noise_checks,
                args.ce_lipschitz_noise_epsilon,
            )
            final_capped_target_ids = (
                capped_target_ids(teacher_final_enc, step_dim(total_steps))
                if teacher_final_enc is not None
                else torch.empty(0, dtype=torch.long, device=device)
            )
            final_probs_teacher = step_target_probs(model, teacher_final_enc, step_dim(total_steps))
            final_probs_generated = step_target_probs(model, generated_final_enc, step_dim(total_steps))
            final_hidden_teacher = (
                step_target_hidden_states(model, teacher_final_enc, step_dim(total_steps))
                if hidden_terminal
                else None
            )
            final_hidden_generated = (
                step_target_hidden_states(model, generated_final_enc, step_dim(total_steps))
                if hidden_terminal
                else None
            )
            final_pred_lipschitz = (
                estimate_hidden_ce_lipschitz(
                    output_embeddings,
                    final_hidden_teacher,
                    final_hidden_generated,
                    final_capped_target_ids,
                    args.jacobian_granularity,
                    args.pred_lipschitz_checks,
                    args.pred_lipschitz_epsilon,
                    hidden_ce_global_C,
                )
                if hidden_terminal
                else estimate_ce_lipschitz(
                    final_probs_teacher,
                    final_probs_generated,
                    final_capped_target_ids,
                    args.pred_lipschitz_checks,
                    args.pred_lipschitz_epsilon,
                )
            )
            final_state_delta_norm = state_delta_norm(
                model,
                input_embeddings,
                teacher_final_enc,
                generated_final_enc,
                step_dim(total_steps),
                terminal_state_output,
                args.jacobian_granularity,
            )
            final_loss_gap = (
                generated_final_loss - teacher_final_loss
                if np.isfinite(generated_final_loss) and np.isfinite(teacher_final_loss)
                else float("nan")
            )
            empirical_lipschitz_C = empirical_lipschitz_constant(
                ce_from_probs(final_probs_teacher, final_capped_target_ids),
                ce_from_probs(final_probs_generated, final_capped_target_ids),
                final_state_delta_norm,
            )
            final_metrics = compute_accuracy([pred_final], [[gold_final]])
            generated_final_correct = final_metrics["accuracy"] / 100.0
            generated_edit_distance = final_metrics["edit_distance"]

            test_errors: List[float] = []
            sample_step_loss_row_start = len(generated_step_loss_rows)
            for target_step in range(1, total_steps + 1):
                step_teacher_enc = teacher_encodings.get(target_step)
                step_generated_enc = build_encoding(
                    tokenizer,
                    seq.source,
                    generated_sequence_steps,
                    seq.steps[target_step - 1],
                    target_step,
                    num_gold_steps,
                    device,
                )
                generated_step_loss = metric_float(step_ce_loss(model, step_generated_enc))
                teacher_step_loss = metric_float(step_losses[target_step - 1])
                step_loss_gap = (
                    generated_step_loss - teacher_step_loss
                    if np.isfinite(generated_step_loss) and np.isfinite(teacher_step_loss)
                    else float("nan")
                )
                step_state_delta_norm = state_delta_norm(
                    model,
                    input_embeddings,
                    step_teacher_enc,
                    step_generated_enc,
                    step_dim(target_step),
                    terminal_state_output,
                    args.jacobian_granularity,
                )
                if args.jacobian_output in ("expected_embedding", "token_distribution"):
                    step_test_error_norm = metric_float(
                        state_step_error(
                            model,
                            input_embeddings,
                            step_generated_enc,
                            step_dim(target_step),
                            args.jacobian_output,
                            args.jacobian_granularity,
                        )
                    )
                else:
                    step_test_error_norm = float("nan")
                test_errors.append(step_test_error_norm)
                # C, the error term and Delta p all live on the capped token window,
                # so the secant must be taken on the same window rather than on the
                # full-length reported CE.
                step_capped_target_ids = (
                    capped_target_ids(step_teacher_enc, step_dim(target_step))
                    if step_teacher_enc is not None
                    else torch.empty(0, dtype=torch.long, device=device)
                )
                step_probs_teacher = step_target_probs(model, step_teacher_enc, step_dim(target_step))
                step_probs_generated = step_target_probs(model, step_generated_enc, step_dim(target_step))
                step_hidden_teacher = (
                    step_target_hidden_states(model, step_teacher_enc, step_dim(target_step))
                    if hidden_terminal
                    else None
                )
                step_hidden_generated = (
                    step_target_hidden_states(model, step_generated_enc, step_dim(target_step))
                    if hidden_terminal
                    else None
                )
                step_teacher_loss_capped = ce_from_probs(step_probs_teacher, step_capped_target_ids)
                step_generated_loss_capped = ce_from_probs(step_probs_generated, step_capped_target_ids)
                step_empirical_lipschitz_C = empirical_lipschitz_constant(
                    step_teacher_loss_capped,
                    step_generated_loss_capped,
                    step_state_delta_norm,
                )
                step_conditioning_step = target_step - 1 if target_step > 1 else 0
                step_ce_lipschitz_span = None
                if step_teacher_enc is not None and step_conditioning_step in step_teacher_enc.spans:
                    step_ce_lipschitz_span = step_teacher_enc.spans[step_conditioning_step]
                    if args.jacobian_output != "selected_logprob" and step_conditioning_step > 0:
                        step_ce_lipschitz_span = cap_step_span(
                            step_ce_lipschitz_span,
                            step_dim(step_conditioning_step),
                        )
                step_ce_lipschitz_noise = estimate_ce_embedding_lipschitz_by_noise(
                    model,
                    input_embeddings,
                    step_teacher_enc,
                    step_ce_lipschitz_span,
                    args.ce_lipschitz_noise_checks,
                    args.ce_lipschitz_noise_epsilon,
                )
                step_pred_lipschitz = (
                    estimate_hidden_ce_lipschitz(
                        output_embeddings,
                        step_hidden_teacher,
                        step_hidden_generated,
                        step_capped_target_ids,
                        args.jacobian_granularity,
                        args.pred_lipschitz_checks,
                        args.pred_lipschitz_epsilon,
                        hidden_ce_global_C,
                    )
                    if hidden_terminal
                    else estimate_ce_lipschitz(
                        step_probs_teacher,
                        step_probs_generated,
                        step_capped_target_ids,
                        args.pred_lipschitz_checks,
                        args.pred_lipschitz_epsilon,
                    )
                )
                # Probability-space C_segment is a segment bound; hidden-state mode
                # uses the global row-norm bound as the conservative ceiling.
                for label, value in (
                    ("empirical_lipschitz_C", step_empirical_lipschitz_C),
                    ("C_random_max", step_pred_lipschitz["C_random_max"]),
                ):
                    ceiling = (
                        step_pred_lipschitz["C_global"]
                        if hidden_terminal
                        else step_pred_lipschitz["C_segment"]
                    )
                    if np.isfinite(value) and np.isfinite(ceiling) and value > ceiling * (1 + 1e-3):
                        print(
                            f"WARNING sample {sample_idx} t={target_step}: {label}={value:.4g} "
                            f"exceeds C ceiling={ceiling:.4g}"
                        )
                generated_step_pred = (
                    generated_sequence_steps[target_step - 1]
                    if target_step - 1 < len(generated_sequence_steps)
                    else ""
                )
                generated_step_metrics = compute_accuracy(
                    [generated_step_pred],
                    [[seq.steps[target_step - 1]]],
                )
                generated_step_loss_rows.append(
                    {
                        "task": task_name,
                        "checkpoint": checkpoint_path or "",
                        "jacobian_output": args.jacobian_output,
                        "jacobian_granularity": args.jacobian_granularity,
                        "embedding_final_state": effective_embedding_final_state,
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "t": target_step,
                        "num_steps": total_steps,
                        "num_gold_steps": num_gold_steps,
                        "final_is_gold_final": final_is_gold_final,
                        "state_space": args.jacobian_output,
                        "teacher_step_loss": teacher_step_loss,
                        "generated_prefix_step_loss": generated_step_loss,
                        "step_loss_gap": step_loss_gap,
                        "teacher_step_loss_capped": step_teacher_loss_capped,
                        "generated_step_loss_capped": step_generated_loss_capped,
                        "step_loss_gap_capped": step_generated_loss_capped - step_teacher_loss_capped,
                        "state_delta_norm": step_state_delta_norm,
                        "test_error_norm": step_test_error_norm,
                        "test_error_square": (
                            step_test_error_norm ** 2
                            if np.isfinite(step_test_error_norm)
                            else float("nan")
                        ),
                        "empirical_lipschitz_C": step_empirical_lipschitz_C,
                        "C_segment": step_pred_lipschitz["C_segment"],
                        "C_point_teacher": step_pred_lipschitz["C_point_teacher"],
                        "C_random_max": step_pred_lipschitz["C_random_max"],
                        "C_random_mean": step_pred_lipschitz["C_random_mean"],
                        "C_global": step_pred_lipschitz.get("C_global", float("nan")),
                        "input_sensitivity_noise_max": step_ce_lipschitz_noise["ce_lipschitz_noise_max"],
                        "input_sensitivity_noise_mean": step_ce_lipschitz_noise["ce_lipschitz_noise_mean"],
                        "generated_step_correct": generated_step_metrics["accuracy"] / 100.0,
                        "generated_step_edit_distance": generated_step_metrics["edit_distance"],
                        "gold_step": seq.steps[target_step - 1],
                        "generated_step": generated_step_pred,
                    }
                )

            pairs: List[Tuple[int, int]] = []
            if args.include_input_jacobian or total_steps == 1:
                pairs.extend((t, 0) for t in range(1, total_steps + 1))
            if args.full_jacobian:
                pairs.extend((t, j) for t in range(2, total_steps + 1) for j in range(1, t))
            else:
                pairs.extend((t, t - 1) for t in range(2, total_steps + 1))

            jac_norms: Dict[Tuple[int, int], float] = {}
            terminal_jac_norms: Dict[Tuple[int, int], float] = {}
            sample_second_order_values: List[Tuple[int, float, float]] = []
            with attention_kernel_context(args.attention_backend):
                for target_step, source_step in pairs:
                    enc = teacher_encodings.get(target_step)
                    if enc is None or source_step not in enc.spans:
                        jac_norm = 0.0
                        terminal_jac_norm = 0.0 if hidden_terminal else float("nan")
                    else:
                        span = enc.spans[source_step]
                        if args.jacobian_output in ("expected_embedding", "token_distribution") and source_step > 0:
                            span = cap_step_span(span, step_dim(source_step))
                        token_norm_rows: List[Dict[str, float]] = []
                        if (
                            args.jacobian_granularity == "token_mean"
                            and args.jacobian_output in ("expected_embedding", "token_distribution")
                            and source_step > 0
                        ):
                            jac_norm, token_norm_rows = estimate_token_mean_jacobian_norm(
                                model,
                                input_embeddings,
                                enc,
                                span,
                                target_width=step_dim(target_step),
                                power_iters=args.power_iters,
                                max_target_tokens=step_dim(target_step),
                                method=args.jacobian_method,
                                jacobian_output=args.jacobian_output,
                            )
                            for token_row in token_norm_rows:
                                jacobian_token_rows.append(
                                    {
                                        "task": task_name,
                                        "checkpoint": checkpoint_path or "",
                                        "jacobian_output": args.jacobian_output,
                                        "jacobian_granularity": args.jacobian_granularity,
                                        "embedding_final_state": effective_embedding_final_state,
                                        "split": args.split,
                                        "length": length,
                                        "sample_idx": sample_idx,
                                        "i": target_step,
                                        "j": source_step,
                                        "num_steps": total_steps,
                                        "num_gold_steps": num_gold_steps,
                                        "final_is_gold_final": final_is_gold_final,
                                        **token_row,
                                    }
                                )
                        else:
                            jac_norm = estimate_jacobian_spectral_norm(
                                model,
                                input_embeddings,
                                enc,
                                span,
                                power_iters=args.power_iters,
                                max_target_tokens=step_dim(target_step),
                                method=args.jacobian_method,
                                jacobian_output=args.jacobian_output,
                            )
                        if hidden_terminal:
                            if (
                                args.jacobian_granularity == "token_mean"
                                and source_step > 0
                            ):
                                terminal_jac_norm, _ = estimate_token_mean_jacobian_norm(
                                    model,
                                    input_embeddings,
                                    enc,
                                    span,
                                    target_width=step_dim(target_step),
                                    power_iters=args.power_iters,
                                    max_target_tokens=step_dim(target_step),
                                    method=args.jacobian_method,
                                    jacobian_output="hidden_state",
                                )
                            else:
                                terminal_jac_norm = estimate_jacobian_spectral_norm(
                                    model,
                                    input_embeddings,
                                    enc,
                                    span,
                                    power_iters=args.power_iters,
                                    max_target_tokens=step_dim(target_step),
                                    method=args.jacobian_method,
                                    jacobian_output="hidden_state",
                                )
                        if finite_difference_count < args.finite_difference_checks:
                            check = finite_difference_directional_check(
                                model,
                                input_embeddings,
                                enc,
                                span,
                                max_target_tokens=step_dim(target_step),
                                epsilon=args.finite_difference_epsilon,
                                jacobian_output=args.jacobian_output,
                            )
                            finite_difference_rows.append(
                                {
                                    "task": task_name,
                                    "checkpoint": checkpoint_path or "",
                                    "jacobian_output": args.jacobian_output,
                                    "jacobian_granularity": args.jacobian_granularity,
                                    "embedding_final_state": effective_embedding_final_state,
                                    "split": args.split,
                                    "length": length,
                                    "sample_idx": sample_idx,
                                    "i": target_step,
                                    "j": source_step,
                                    **check,
                                }
                            )
                            finite_difference_count += 1
                    jac_norms[(target_step, source_step)] = jac_norm
                    if hidden_terminal:
                        terminal_jac_norms[(target_step, source_step)] = terminal_jac_norm

                if args.second_order_checks > 0:
                    for target_step, source_step in pairs:
                        if source_step <= 0:
                            continue
                        if args.second_order_pairs == "adjacent" and source_step != target_step - 1:
                            continue
                        if args.second_order_pairs == "terminal" and target_step != total_steps:
                            continue
                        enc = teacher_encodings.get(target_step)
                        if enc is None or source_step not in enc.spans:
                            continue

                        span = enc.spans[source_step]
                        if args.jacobian_output in ("expected_embedding", "token_distribution"):
                            span = cap_step_span(span, step_dim(source_step))

                        state_second_order = estimate_jacobian_lipschitz_by_jvp_difference(
                            model,
                            input_embeddings,
                            enc,
                            span,
                            max_target_tokens=step_dim(target_step),
                            epsilon=args.second_order_epsilon,
                            jacobian_output=args.jacobian_output,
                            num_checks=args.second_order_checks,
                        )
                        if np.isfinite(state_second_order["M_random_max"]):
                            sample_second_order_values.append(
                                (
                                    target_step,
                                    state_second_order["M_random_max"],
                                    state_second_order["M_random_mean"],
                                )
                            )
                        second_order_rows.append(
                            {
                                "task": task_name,
                                "checkpoint": checkpoint_path or "",
                                "jacobian_output": args.jacobian_output,
                                "jacobian_granularity": args.jacobian_granularity,
                                "embedding_final_state": effective_embedding_final_state,
                                "split": args.split,
                                "length": length,
                                "sample_idx": sample_idx,
                                "i": target_step,
                                "j": source_step,
                                "num_steps": total_steps,
                                "num_gold_steps": num_gold_steps,
                                "final_is_gold_final": final_is_gold_final,
                                "map_type": "state",
                                "M_random_max": state_second_order["M_random_max"],
                                "M_random_mean": state_second_order["M_random_mean"],
                                "second_order_checks": args.second_order_checks,
                                "second_order_epsilon": args.second_order_epsilon,
                                "second_order_pairs": args.second_order_pairs,
                            }
                        )

                        if hidden_terminal:
                            terminal_second_order = estimate_jacobian_lipschitz_by_jvp_difference(
                                model,
                                input_embeddings,
                                enc,
                                span,
                                max_target_tokens=step_dim(target_step),
                                epsilon=args.second_order_epsilon,
                                jacobian_output="hidden_state",
                                num_checks=args.second_order_checks,
                            )
                            if np.isfinite(terminal_second_order["M_random_max"]):
                                sample_second_order_values.append(
                                    (
                                        target_step,
                                        terminal_second_order["M_random_max"],
                                        terminal_second_order["M_random_mean"],
                                    )
                                )
                            second_order_rows.append(
                                {
                                    "task": task_name,
                                    "checkpoint": checkpoint_path or "",
                                    "jacobian_output": args.jacobian_output,
                                    "jacobian_granularity": args.jacobian_granularity,
                                    "embedding_final_state": effective_embedding_final_state,
                                    "split": args.split,
                                    "length": length,
                                    "sample_idx": sample_idx,
                                    "i": target_step,
                                    "j": source_step,
                                    "num_steps": total_steps,
                                    "num_gold_steps": num_gold_steps,
                                    "final_is_gold_final": final_is_gold_final,
                                    "map_type": "terminal",
                                    "M_random_max": terminal_second_order["M_random_max"],
                                    "M_random_mean": terminal_second_order["M_random_mean"],
                                    "second_order_checks": args.second_order_checks,
                                    "second_order_epsilon": args.second_order_epsilon,
                                    "second_order_pairs": args.second_order_pairs,
                                }
                            )

            embedding_rho = compute_rho(jac_norms, total_steps) if args.full_jacobian else {}
            terminal_rho = (
                compute_terminal_rho(embedding_rho, terminal_jac_norms, total_steps)
                if args.full_jacobian and hidden_terminal
                else {}
            )
            effective_rho = terminal_rho if hidden_terminal else embedding_rho
            M_prefix_max: Dict[int, float] = {}
            M_prefix_mean: Dict[int, float] = {}
            for prefix_t in range(1, total_steps + 1):
                prefix_values = [
                    value
                    for target_step, value, _ in sample_second_order_values
                    if target_step <= prefix_t and np.isfinite(value)
                ]
                prefix_means = [
                    value
                    for target_step, _, value in sample_second_order_values
                    if target_step <= prefix_t and np.isfinite(value)
                ]
                M_prefix_max[prefix_t] = max(prefix_values) if prefix_values else float("nan")
                M_prefix_mean[prefix_t] = float(np.mean(prefix_means)) if prefix_means else float("nan")
            second_order_prefix_stats = compute_second_order_prefix_stats(
                effective_rho,
                test_errors,
                M_prefix_max,
                M_prefix_mean,
                total_steps,
            )
            embedding_local_logs = local_log_amplifications(jac_norms, total_steps)
            local_logs = (
                local_log_amplifications_with_terminal(jac_norms, terminal_jac_norms, total_steps)
                if hidden_terminal
                else embedding_local_logs
            )
            embedding_prefix_logs = prefix_log_amplifications_from_first(jac_norms, total_steps)
            prefix_logs = (
                prefix_log_amplifications_from_first_with_terminal(jac_norms, terminal_jac_norms, total_steps)
                if hidden_terminal
                else embedding_prefix_logs
            )
            adjacent_norms = [jac_norms.get((t, t - 1), float("nan")) for t in range(2, total_steps + 1)]
            input_jac_norms = [
                jac_norms.get((t, 0), float("nan"))
                for t in range(1, total_steps + 1)
                if (t, 0) in jac_norms
            ]
            input_jac_norm = float(np.nanmean(input_jac_norms)) if input_jac_norms else float("nan")
            log_input_jac_norm = math.log(max(input_jac_norm, EPS)) if np.isfinite(input_jac_norm) else float("nan")

            if args.jacobian_output in ("expected_embedding", "token_distribution"):
                if args.full_jacobian:
                    bound_terms = [
                        effective_rho.get((total_steps, i), 0.0) * train_errors[i - 1]
                        for i in range(1, total_steps)
                        if np.isfinite(train_errors[i - 1])
                    ]
                    embedding_bound_terms = [
                        embedding_rho.get((total_steps, i), 0.0) * train_errors[i - 1]
                        for i in range(1, total_steps)
                        if np.isfinite(train_errors[i - 1])
                    ]
                else:
                    bound_terms = [
                        math.exp(local_logs[i]) * train_errors[i - 1]
                        for i in range(1, total_steps)
                        if np.isfinite(train_errors[i - 1]) and np.isfinite(local_logs[i])
                    ]
                    embedding_bound_terms = [
                        math.exp(embedding_local_logs[i]) * train_errors[i - 1]
                        for i in range(1, total_steps)
                        if np.isfinite(train_errors[i - 1]) and np.isfinite(embedding_local_logs[i])
                    ]
                bound_proxy = float(np.sum(bound_terms)) if bound_terms else float("nan")
                log_bound_proxy = math.log(max(bound_proxy, EPS)) if np.isfinite(bound_proxy) else float("nan")
                embedding_bound_proxy = (
                    float(np.sum(embedding_bound_terms)) if embedding_bound_terms else float("nan")
                )
                if (
                    not np.isfinite(bound_proxy)
                    and total_steps == 1
                    and len(train_errors) > 0
                    and np.isfinite(input_jac_norm)
                    and np.isfinite(train_errors[0])
                ):
                    bound_proxy = input_jac_norm * train_errors[0]
                    log_bound_proxy = math.log(max(bound_proxy, EPS))
                    embedding_bound_proxy = bound_proxy
            else:
                bound_proxy = float("nan")
                log_bound_proxy = float("nan")
                embedding_bound_proxy = float("nan")
            bound_proxy_valid = bool(
                args.jacobian_output in ("expected_embedding", "token_distribution")
                and np.isfinite(bound_proxy)
            )
            calibrated_bound_proxy = (
                empirical_lipschitz_C * bound_proxy
                if np.isfinite(empirical_lipschitz_C) and np.isfinite(bound_proxy)
                else float("nan")
            )
            segment_calibrated_bound_proxy = (
                final_pred_lipschitz["C_segment"] * bound_proxy
                if np.isfinite(final_pred_lipschitz["C_segment"]) and np.isfinite(bound_proxy)
                else float("nan")
            )
            random_max_calibrated_bound_proxy = (
                final_pred_lipschitz["C_random_max"] * bound_proxy
                if np.isfinite(final_pred_lipschitz["C_random_max"]) and np.isfinite(bound_proxy)
                else float("nan")
            )
            global_calibrated_bound_proxy = (
                final_pred_lipschitz.get("C_global", float("nan")) * bound_proxy
                if np.isfinite(final_pred_lipschitz.get("C_global", float("nan"))) and np.isfinite(bound_proxy)
                else float("nan")
            )
            pred_calibrated_bound_proxy = segment_calibrated_bound_proxy

            prefix_bound_stats: Dict[int, Dict[str, float]] = {}
            embedding_prefix_bound_stats: Dict[int, Dict[str, float]] = {}
            for prefix_t in range(1, total_steps + 1):
                sum_rho_t_i = 0.0
                bound_proxy_t = 0.0
                embedding_sum_rho_t_i = 0.0
                embedding_bound_proxy_t = 0.0
                for source_step in range(1, prefix_t):
                    rho_value = effective_rho.get((prefix_t, source_step), 0.0)
                    embedding_rho_value = embedding_rho.get((prefix_t, source_step), 0.0)
                    sum_rho_t_i += rho_value
                    embedding_sum_rho_t_i += embedding_rho_value
                    train_error = train_errors[source_step - 1] if source_step - 1 < len(train_errors) else float("nan")
                    if np.isfinite(train_error):
                        bound_proxy_t += rho_value * train_error
                        embedding_bound_proxy_t += embedding_rho_value * train_error
                prefix_bound_stats[prefix_t] = {
                    "sum_rho_t_i": sum_rho_t_i,
                    "bound_proxy_t": bound_proxy_t,
                }
                embedding_prefix_bound_stats[prefix_t] = {
                    "sum_rho_t_i": embedding_sum_rho_t_i,
                    "bound_proxy_t": embedding_bound_proxy_t,
                }

            for prefix_t in range(1, total_steps + 1):
                adjacent_log_rho_t_1 = prefix_logs.get(prefix_t, float("nan"))
                adjacent_rho_t_1 = math.exp(adjacent_log_rho_t_1) if np.isfinite(adjacent_log_rho_t_1) else float("nan")
                recursive_rho_t_1 = 1.0 if prefix_t == 1 else effective_rho.get((prefix_t, 1), float("nan"))
                recursive_log_rho_t_1 = (
                    math.log(max(recursive_rho_t_1, EPS)) if np.isfinite(recursive_rho_t_1) else float("nan")
                )
                embedding_adjacent_log_rho_t_1 = embedding_prefix_logs.get(prefix_t, float("nan"))
                embedding_adjacent_rho_t_1 = (
                    math.exp(embedding_adjacent_log_rho_t_1)
                    if np.isfinite(embedding_adjacent_log_rho_t_1)
                    else float("nan")
                )
                embedding_recursive_rho_t_1 = (
                    1.0 if prefix_t == 1 else embedding_rho.get((prefix_t, 1), float("nan"))
                )
                embedding_recursive_log_rho_t_1 = (
                    math.log(max(embedding_recursive_rho_t_1, EPS))
                    if np.isfinite(embedding_recursive_rho_t_1)
                    else float("nan")
                )
                rho_prefix_rows.append(
                    {
                        "task": task_name,
                        "checkpoint": checkpoint_path or "",
                        "jacobian_output": args.jacobian_output,
                        "jacobian_granularity": args.jacobian_granularity,
                        "embedding_final_state": effective_embedding_final_state,
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "t": prefix_t,
                        "num_steps": total_steps,
                        "num_gold_steps": num_gold_steps,
                        "final_is_gold_final": final_is_gold_final,
                        "adjacent_rho_t_1": adjacent_rho_t_1,
                        "adjacent_log_rho_t_1": adjacent_log_rho_t_1,
                        "recursive_rho_t_1": recursive_rho_t_1,
                        "recursive_log_rho_t_1": recursive_log_rho_t_1,
                        "embedding_adjacent_rho_t_1": embedding_adjacent_rho_t_1,
                        "embedding_adjacent_log_rho_t_1": embedding_adjacent_log_rho_t_1,
                        "embedding_recursive_rho_t_1": embedding_recursive_rho_t_1,
                        "embedding_recursive_log_rho_t_1": embedding_recursive_log_rho_t_1,
                        "computed_full_jacobian": bool(args.full_jacobian),
                    }
                )

            for row in generated_step_loss_rows[sample_step_loss_row_start:]:
                prefix_t = int(row["t"])
                prefix_stats = prefix_bound_stats.get(prefix_t, {})
                second_stats = second_order_prefix_stats.get(prefix_t, {})
                bound_proxy_t = prefix_stats.get("bound_proxy_t", float("nan"))
                C_segment = metric_float(row.get("C_segment"))
                C_random_max = metric_float(row.get("C_random_max"))
                C_global = metric_float(row.get("C_global"))
                second_order_amount = second_stats.get("second_order_geometry_amount_t", float("nan"))

                segment_first = (
                    C_segment * bound_proxy_t
                    if np.isfinite(C_segment) and np.isfinite(bound_proxy_t)
                    else float("nan")
                )
                random_max_first = (
                    C_random_max * bound_proxy_t
                    if np.isfinite(C_random_max) and np.isfinite(bound_proxy_t)
                    else float("nan")
                )
                global_first = (
                    C_global * bound_proxy_t
                    if np.isfinite(C_global) and np.isfinite(bound_proxy_t)
                    else float("nan")
                )
                segment_second = (
                    C_segment * second_order_amount
                    if np.isfinite(C_segment) and np.isfinite(second_order_amount)
                    else float("nan")
                )
                random_max_second = (
                    C_random_max * second_order_amount
                    if np.isfinite(C_random_max) and np.isfinite(second_order_amount)
                    else float("nan")
                )
                global_second = (
                    C_global * second_order_amount
                    if np.isfinite(C_global) and np.isfinite(second_order_amount)
                    else float("nan")
                )
                row.update(
                    {
                        "sum_rho_t_i": prefix_stats.get("sum_rho_t_i", float("nan")),
                        "bound_proxy_t": bound_proxy_t,
                        "M_random_max_t": second_stats.get("M_random_max_t", float("nan")),
                        "M_random_mean_t": second_stats.get("M_random_mean_t", float("nan")),
                        "gamma_t": second_stats.get("gamma_t", float("nan")),
                        "rho_remainder_multiplier_t": second_stats.get(
                            "rho_remainder_multiplier_t",
                            float("nan"),
                        ),
                        "test_error_square_sum_t": second_stats.get(
                            "test_error_square_sum_t",
                            float("nan"),
                        ),
                        "second_order_weighted_test_error_square_sum_t": second_stats.get(
                            "second_order_weighted_test_error_square_sum_t",
                            float("nan"),
                        ),
                        "second_order_geometry_coefficient_t": second_stats.get(
                            "second_order_geometry_coefficient_t",
                            float("nan"),
                        ),
                        "second_order_geometry_amount_t": second_order_amount,
                        "segment_calibrated_bound_t": segment_first,
                        "random_max_calibrated_bound_t": random_max_first,
                        "global_calibrated_bound_t": global_first,
                        "pred_calibrated_bound_t": segment_first,
                        "segment_second_order_bound_t": segment_second,
                        "random_max_second_order_bound_t": random_max_second,
                        "global_second_order_bound_t": global_second,
                        "segment_total_bound_t": (
                            segment_first + segment_second
                            if np.isfinite(segment_first) and np.isfinite(segment_second)
                            else float("nan")
                        ),
                        "random_max_total_bound_t": (
                            random_max_first + random_max_second
                            if np.isfinite(random_max_first) and np.isfinite(random_max_second)
                            else float("nan")
                        ),
                        "global_total_bound_t": (
                            global_first + global_second
                            if np.isfinite(global_first) and np.isfinite(global_second)
                            else float("nan")
                        ),
                    }
                )

            for (target_step, source_step), jac_norm in sorted(jac_norms.items()):
                jacobian_rows.append(
                    {
                        "task": task_name,
                        "checkpoint": checkpoint_path or "",
                        "jacobian_output": args.jacobian_output,
                        "jacobian_granularity": args.jacobian_granularity,
                        "embedding_final_state": effective_embedding_final_state,
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "i": target_step,
                        "j": source_step,
                        "num_steps": total_steps,
                        "num_gold_steps": num_gold_steps,
                        "final_is_gold_final": final_is_gold_final,
                        "jac_norm": jac_norm,
                        "log_jac_norm": math.log(max(jac_norm, EPS)) if np.isfinite(jac_norm) else float("nan"),
                        "rho_T_i": embedding_rho.get((total_steps, source_step), float("nan")),
                        "local_log_amp_i": embedding_local_logs.get(source_step, float("nan")),
                        "step_loss_i": metric_float(step_losses[source_step - 1]) if source_step > 0 else float("nan"),
                        "train_error_i": train_errors[source_step - 1] if source_step > 0 else float("nan"),
                        "train_error_type": train_error_type,
                        "teacher_final_loss": teacher_final_loss,
                        "generated_final_loss": generated_final_loss,
                        "generated_final_correct": generated_final_correct,
                        "computed_full_jacobian": bool(args.full_jacobian),
                    }
                )

            for (target_step, source_step), jac_norm in sorted(terminal_jac_norms.items()):
                terminal_jacobian_rows.append(
                    {
                        "task": task_name,
                        "checkpoint": checkpoint_path or "",
                        "jacobian_output": args.jacobian_output,
                        "jacobian_granularity": args.jacobian_granularity,
                        "embedding_final_state": effective_embedding_final_state,
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "i": target_step,
                        "j": source_step,
                        "num_steps": total_steps,
                        "num_gold_steps": num_gold_steps,
                        "final_is_gold_final": final_is_gold_final,
                        "terminal_jac_norm": jac_norm,
                        "terminal_log_jac_norm": (
                            math.log(max(jac_norm, EPS)) if np.isfinite(jac_norm) else float("nan")
                        ),
                        "terminal_rho_T_i": terminal_rho.get((total_steps, source_step), float("nan")),
                        "train_error_i": train_errors[source_step - 1] if source_step > 0 else float("nan"),
                        "train_error_type": train_error_type,
                        "computed_full_jacobian": bool(args.full_jacobian),
                    }
                )

            final_second_stats = second_order_prefix_stats.get(total_steps, {})
            final_second_order_amount = final_second_stats.get(
                "second_order_geometry_amount_t",
                float("nan"),
            )
            segment_second_order_bound = (
                final_pred_lipschitz["C_segment"] * final_second_order_amount
                if np.isfinite(final_pred_lipschitz["C_segment"]) and np.isfinite(final_second_order_amount)
                else float("nan")
            )
            random_max_second_order_bound = (
                final_pred_lipschitz["C_random_max"] * final_second_order_amount
                if np.isfinite(final_pred_lipschitz["C_random_max"]) and np.isfinite(final_second_order_amount)
                else float("nan")
            )
            global_second_order_bound = (
                final_pred_lipschitz.get("C_global", float("nan")) * final_second_order_amount
                if np.isfinite(final_pred_lipschitz.get("C_global", float("nan")))
                and np.isfinite(final_second_order_amount)
                else float("nan")
            )

            example_rows.append(
                {
                    "task": task_name,
                    "checkpoint": checkpoint_path or "",
                    "jacobian_output": args.jacobian_output,
                    "jacobian_granularity": args.jacobian_granularity,
                    "embedding_final_state": effective_embedding_final_state,
                    "split": args.split,
                    "length": length,
                    "sample_idx": sample_idx,
                    "num_steps": total_steps,
                    "num_gold_steps": num_gold_steps,
                    "final_is_gold_final": final_is_gold_final,
                    "teacher_final_loss": teacher_final_loss,
                    "generated_final_loss": generated_final_loss,
                    "loss_gap": final_loss_gap,
                    "generated_final_correct": generated_final_correct,
                    "generated_edit_distance": generated_edit_distance,
                    "bound_proxy": bound_proxy,
                    "log_bound_proxy": log_bound_proxy,
                    "bound_proxy_valid": bound_proxy_valid,
                    "embedding_bound_proxy": embedding_bound_proxy,
                    "embedding_log_bound_proxy": (
                        math.log(max(embedding_bound_proxy, EPS))
                        if np.isfinite(embedding_bound_proxy)
                        else float("nan")
                    ),
                    "calibrated_bound_proxy": calibrated_bound_proxy,
                    "segment_calibrated_bound_proxy": segment_calibrated_bound_proxy,
                    "random_max_calibrated_bound_proxy": random_max_calibrated_bound_proxy,
                    "global_calibrated_bound_proxy": global_calibrated_bound_proxy,
                    "pred_calibrated_bound_proxy": pred_calibrated_bound_proxy,
                    "M_random_max": final_second_stats.get("M_random_max_t", float("nan")),
                    "M_random_mean": final_second_stats.get("M_random_mean_t", float("nan")),
                    "gamma_T": final_second_stats.get("gamma_t", float("nan")),
                    "rho_remainder_multiplier": final_second_stats.get(
                        "rho_remainder_multiplier_t",
                        float("nan"),
                    ),
                    "test_error_square_sum": final_second_stats.get(
                        "test_error_square_sum_t",
                        float("nan"),
                    ),
                    "second_order_weighted_test_error_square_sum": final_second_stats.get(
                        "second_order_weighted_test_error_square_sum_t",
                        float("nan"),
                    ),
                    "second_order_geometry_coefficient": final_second_stats.get(
                        "second_order_geometry_coefficient_t",
                        float("nan"),
                    ),
                    "second_order_geometry_amount": final_second_order_amount,
                    "segment_second_order_bound": segment_second_order_bound,
                    "random_max_second_order_bound": random_max_second_order_bound,
                    "global_second_order_bound": global_second_order_bound,
                    "segment_total_bound": (
                        segment_calibrated_bound_proxy + segment_second_order_bound
                        if np.isfinite(segment_calibrated_bound_proxy)
                        and np.isfinite(segment_second_order_bound)
                        else float("nan")
                    ),
                    "random_max_total_bound": (
                        random_max_calibrated_bound_proxy + random_max_second_order_bound
                        if np.isfinite(random_max_calibrated_bound_proxy)
                        and np.isfinite(random_max_second_order_bound)
                        else float("nan")
                    ),
                    "global_total_bound": (
                        global_calibrated_bound_proxy + global_second_order_bound
                        if np.isfinite(global_calibrated_bound_proxy)
                        and np.isfinite(global_second_order_bound)
                        else float("nan")
                    ),
                    "state_space": args.jacobian_output,
                    "seed": args.seed,
                    "source_truncated": False,
                    "final_state_delta_norm": final_state_delta_norm,
                    "empirical_lipschitz_C": empirical_lipschitz_C,
                    "C_segment": final_pred_lipschitz["C_segment"],
                    "C_point_teacher": final_pred_lipschitz["C_point_teacher"],
                    "C_random_max": final_pred_lipschitz["C_random_max"],
                    "C_random_mean": final_pred_lipschitz["C_random_mean"],
                    "C_global": final_pred_lipschitz.get("C_global", float("nan")),
                    "input_sensitivity_noise_max": final_ce_lipschitz_noise["ce_lipschitz_noise_max"],
                    "input_sensitivity_noise_mean": final_ce_lipschitz_noise["ce_lipschitz_noise_mean"],
                    "max_local_log_amp": max(local_logs.values()) if local_logs else float("nan"),
                    "mean_adjacent_jac_norm": float(np.nanmean(adjacent_norms)) if adjacent_norms else float("nan"),
                    "mean_train_error": float(np.nanmean(train_errors)) if train_errors else float("nan"),
                    "train_error_type": train_error_type,
                    "input_jac_norm": input_jac_norm,
                    "log_input_jac_norm": log_input_jac_norm,
                    "raw_gold_answer": seq.raw_answer,
                    "generated_answer": " ".join(generated_answer.replace("\n", " ").split()),
                    "gold_final": gold_final,
                    "pred_final": pred_final,
                }
            )

            processed += 1
            progress.update(1)

        if processed >= args.max_examples:
            break

    progress.close()

    write_csv(
        os.path.join(args.output_dir, "jacobian_rows.csv"),
        jacobian_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "i",
            "j",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "jac_norm",
            "log_jac_norm",
            "rho_T_i",
            "local_log_amp_i",
            "step_loss_i",
            "train_error_i",
            "train_error_type",
            "teacher_final_loss",
            "generated_final_loss",
            "generated_final_correct",
            "computed_full_jacobian",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "terminal_jacobian_rows.csv"),
        terminal_jacobian_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "i",
            "j",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "terminal_jac_norm",
            "terminal_log_jac_norm",
            "terminal_rho_T_i",
            "train_error_i",
            "train_error_type",
            "computed_full_jacobian",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "second_order_rows.csv"),
        second_order_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "i",
            "j",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "map_type",
            "M_random_max",
            "M_random_mean",
            "second_order_checks",
            "second_order_epsilon",
            "second_order_pairs",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "jacobian_token_rows.csv"),
        jacobian_token_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "i",
            "j",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "target_token_idx",
            "source_token_idx",
            "token_jac_norm",
            "token_log_jac_norm",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "rho_prefix_rows.csv"),
        rho_prefix_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "t",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "adjacent_rho_t_1",
            "adjacent_log_rho_t_1",
            "recursive_rho_t_1",
            "recursive_log_rho_t_1",
            "embedding_adjacent_rho_t_1",
            "embedding_adjacent_log_rho_t_1",
            "embedding_recursive_rho_t_1",
            "embedding_recursive_log_rho_t_1",
            "computed_full_jacobian",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "generated_step_loss_rows.csv"),
        generated_step_loss_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "t",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "state_space",
            "teacher_step_loss",
            "generated_prefix_step_loss",
            "step_loss_gap",
            "teacher_step_loss_capped",
            "generated_step_loss_capped",
            "step_loss_gap_capped",
            "state_delta_norm",
            "test_error_norm",
            "test_error_square",
            "empirical_lipschitz_C",
            "C_segment",
            "C_point_teacher",
            "C_random_max",
            "C_random_mean",
            "C_global",
            "input_sensitivity_noise_max",
            "input_sensitivity_noise_mean",
            "sum_rho_t_i",
            "bound_proxy_t",
            "M_random_max_t",
            "M_random_mean_t",
            "gamma_t",
            "rho_remainder_multiplier_t",
            "test_error_square_sum_t",
            "second_order_weighted_test_error_square_sum_t",
            "second_order_geometry_coefficient_t",
            "second_order_geometry_amount_t",
            "segment_calibrated_bound_t",
            "random_max_calibrated_bound_t",
            "global_calibrated_bound_t",
            "pred_calibrated_bound_t",
            "segment_second_order_bound_t",
            "random_max_second_order_bound_t",
            "global_second_order_bound_t",
            "segment_total_bound_t",
            "random_max_total_bound_t",
            "global_total_bound_t",
            "generated_step_correct",
            "generated_step_edit_distance",
            "gold_step",
            "generated_step",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "example_metrics.csv"),
        example_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "jacobian_granularity",
            "embedding_final_state",
            "split",
            "length",
            "sample_idx",
            "num_steps",
            "num_gold_steps",
            "final_is_gold_final",
            "teacher_final_loss",
            "generated_final_loss",
            "loss_gap",
            "generated_final_correct",
            "generated_edit_distance",
            "bound_proxy",
            "log_bound_proxy",
            "bound_proxy_valid",
            "embedding_bound_proxy",
            "embedding_log_bound_proxy",
            "calibrated_bound_proxy",
            "segment_calibrated_bound_proxy",
            "random_max_calibrated_bound_proxy",
            "global_calibrated_bound_proxy",
            "pred_calibrated_bound_proxy",
            "M_random_max",
            "M_random_mean",
            "gamma_T",
            "rho_remainder_multiplier",
            "test_error_square_sum",
            "second_order_weighted_test_error_square_sum",
            "second_order_geometry_coefficient",
            "second_order_geometry_amount",
            "segment_second_order_bound",
            "random_max_second_order_bound",
            "global_second_order_bound",
            "segment_total_bound",
            "random_max_total_bound",
            "global_total_bound",
            "state_space",
            "seed",
            "source_truncated",
            "final_state_delta_norm",
            "empirical_lipschitz_C",
            "C_segment",
            "C_point_teacher",
            "C_random_max",
            "C_random_mean",
            "C_global",
            "input_sensitivity_noise_max",
            "input_sensitivity_noise_mean",
            "max_local_log_amp",
            "mean_adjacent_jac_norm",
            "mean_train_error",
            "train_error_type",
            "input_jac_norm",
            "log_input_jac_norm",
            "gold_final",
            "pred_final",
            "raw_gold_answer",
            "generated_answer",
        ],
    )
    write_jsonl(os.path.join(args.output_dir, "example_metrics.jsonl"), example_rows)
    if finite_difference_rows:
        write_csv(
            os.path.join(args.output_dir, "finite_difference_checks.csv"),
            finite_difference_rows,
            [
                "task",
                "checkpoint",
                "jacobian_output",
                "jacobian_granularity",
                "embedding_final_state",
                "split",
                "length",
                "sample_idx",
                "i",
                "j",
                "jvp_directional_norm",
                "finite_difference_directional_norm",
                "relative_norm_error",
                "directional_cosine",
            ],
        )
    write_analysis_outputs(args.output_dir, example_rows, jacobian_rows)

    print(f"Processed {processed} examples.")
    print(f"Wrote outputs to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CLRS CoT Jacobian amplification.")

    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seeds gold-step sampling, which is random whenever --reduce_steps_ratio < 1.",
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", type=str, default="32")

    parser.add_argument("--task_names", type=str, nargs="+", default=["dfs"])
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--train_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--test_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1_000_000)
    parser.add_argument("--minimum_samples_validation", type=int, default=1_000_000)
    parser.add_argument("--few_shot_k", type=int, default=0)
    parser.add_argument("--only_answer_output", action="store_true")
    parser.add_argument("--reduce_steps_ratio", type=float, default=1.0)
    parser.add_argument("--reduce_steps_equally_spaced", action="store_true")

    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--min_jacobian_steps", type=int, default=0)
    parser.add_argument(
        "--max_jacobian_steps",
        type=int,
        default=0,
        help=(
            "Longest prefix T to analyse; 0 means the whole gold sequence. The gold "
            "sequence itself is never shortened, so the ' | ' final-answer marker "
            "stays where the data puts it."
        ),
    )
    parser.add_argument("--power_iters", type=int, default=5)
    parser.add_argument("--max_target_tokens_for_jacobian", type=int, default=8)
    parser.add_argument("--jacobian_method", type=str, choices=["jvp", "rowgrad"], default="jvp")
    parser.add_argument(
        "--jacobian_granularity",
        type=str,
        choices=["block", "token_mean"],
        default="block",
        help=(
            "block estimates one spectral norm for each multi-token step map. token_mean "
            "estimates token-to-token Jacobian norms and averages them into the step map."
        ),
    )
    parser.add_argument(
        "--jacobian_output",
        type=str,
        choices=[
            "selected_logprob",
            "expected_embedding",
            "token_distribution",
            "hidden_state",
        ],
        default=None,
        help=(
            "token_distribution is the formulation the bound is stated in: the state is "
            "the next-step distribution, ground truth is a one-hot, and cross-entropy is a "
            "genuine function of it. expected_embedding projects through the embedding "
            "matrix, which loses information and leaves the loss undefined on the state. "
            "selected_logprob is the original diagnostic and does not compose across steps. "
            "hidden_state is reserved for Coconut's latent text-proxy evaluator."
        ),
    )
    parser.add_argument(
        "--embedding_final_state",
        type=str,
        choices=["embedding", "hidden_state"],
        default="hidden_state",
        help=(
            "For --jacobian_output expected_embedding, choose whether the terminal "
            "loss-gap bound ends at the expected embedding or at the pre-unembedding "
            "hidden state. The flag is ignored for other jacobian_output modes unless "
            "explicitly set to hidden_state, which is invalid."
        ),
    )
    parser.add_argument("--attention_backend", type=str, choices=["math", "auto"], default="math")
    parser.add_argument("--include_input_jacobian", action="store_true")
    parser.add_argument("--full_jacobian", action="store_true")
    parser.add_argument("--finite_difference_checks", type=int, default=0)
    parser.add_argument("--finite_difference_epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--second_order_checks",
        type=int,
        default=0,
        help="Randomized JVP-difference checks for estimating local Jacobian Lipschitz M.",
    )
    parser.add_argument("--second_order_epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--second_order_pairs",
        type=str,
        choices=["all", "adjacent", "terminal"],
        default="all",
        help="Which step-pair maps to use when estimating the second-order M term.",
    )
    parser.add_argument("--ce_lipschitz_noise_checks", type=int, default=0)
    parser.add_argument("--ce_lipschitz_noise_epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--pred_lipschitz_checks",
        type=int,
        default=16,
        help="Random simplex-tangent perturbations used to cross-check the analytic C_segment.",
    )
    parser.add_argument("--pred_lipschitz_epsilon", type=float, default=1e-4)
    parser.add_argument("--generation_max_new_tokens", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")
    parser.add_argument("--use_graph_llama", action="store_true")
    parser.add_argument("--only_train_graph", action="store_true")
    parser.add_argument("--test_classifier_before_cross_attn", action="store_true")
    parser.add_argument("--freeze_graph_tower", action="store_true")
    parser.add_argument("--use_cross_attn", action="store_true")
    parser.add_argument("--add_output_projection", action="store_true")
    parser.add_argument("--alignment_loss_weight", type=float, default=0.0)

    parser.add_argument("--intrinsic_dim", type=int, default=0)
    parser.add_argument("--intrinsic_mode", type=str, default="rdkronqr")

    parser.add_argument("--use_quant", action="store_true")
    parser.add_argument("--quant_training_mode", type=str, choices=["lora", "full"], default="lora")
    parser.add_argument("--quant_w_bits", type=int, default=2)
    parser.add_argument("--quant_modules", type=str, nargs="+", default=DEFAULT_QUANT_MODULES)
    parser.add_argument("--quant_noise_injection", action="store_true")
    parser.add_argument("--quant_pre_quantization_noise", action="store_true")
    parser.add_argument("--quant_post_quantization_noise", action="store_true")
    parser.add_argument("--quant_sigma_weights", type=float, default=0.0)
    parser.add_argument("--quant_sigma_clipvals", type=float, default=0.0)
    parser.add_argument("--quant_initialize_noise", action="store_true")
    parser.add_argument("--quant_trainable_noise_scale", action="store_true")
    parser.add_argument("--quant_reinitialize_steps", type=int, default=0)
    parser.add_argument("--quant_reinitialize_alpha", type=float, default=0.2)
    
    # just to keep the command consistent
    parser.add_argument("--use_coconut", action="store_true") # train with Chain of Continuous Thought latent reasoning
    parser.add_argument("--coconut_steps_to_full_latent", type=int, default=0) # optimizer steps used to replace all reasoning steps with latent thoughts; 0 means full training run
    parser.add_argument("--coconut_latents_per_step", type=int, default=1) # continuous thoughts inserted per removed language reasoning step
    parser.add_argument("--coconut_max_train_latent_thoughts", type=int, default=-1) # cap training latent thoughts per sample; -1 means no cap
    parser.add_argument("--coconut_eval_latent_thoughts", type=int, default=-1) # explicit number of latent thoughts for Coconut generation
    parser.add_argument("--coconut_gradient_checkpointing", action="store_true") # reduce Coconut activation memory at the cost of speed
    parser.add_argument("--coconut_bot_token", type=str, default="<bot>")
    parser.add_argument("--coconut_eot_token", type=str, default="<eot>")
    parser.add_argument("--eval_test_during_fit", action="store_true") # evaluate test split during each fit validation epoch
    parser.add_argument("--use_forward_noise_injection", action="store_true")  # inject noise during forward pass
    parser.add_argument("--forward_noise_std", type=float, default=0.01)  # std of forward noise
    parser.add_argument("--forward_noise_lora_only", action="store_true")  # apply noise only to LoRA layers

    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
