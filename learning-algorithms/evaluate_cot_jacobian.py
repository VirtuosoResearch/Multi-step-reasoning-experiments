"""
Offline evaluator for CLRS chain-of-thought error amplification.

The evaluator estimates step-to-step Jacobian spectral norms in embedding
space and compares their amplification proxies with the teacher-forced versus
autoregressive final-step loss gap.
"""

import argparse
import csv
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_clrs_text import DEFAULT_QUANT_MODULES, initialize_model, validate_quant_args
from src.custom.clrs_text_task_data_module import TextCLRSDataModule
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


def generated_prefix_for_step(generated_steps: Sequence[str], target_step: int, total_steps: int) -> str:
    if target_step <= 1:
        return ""

    pieces: List[str] = []
    for prev_step in range(1, target_step):
        if prev_step > len(generated_steps):
            break
        pieces.append(generated_steps[prev_step - 1])
        if prev_step == total_steps - 1:
            pieces.append(" | ")
        else:
            pieces.append(", ")
    return "".join(pieces)


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


def encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_step_encoding(
    tokenizer,
    source: str,
    steps: Sequence[str],
    target_step: int,
    device: torch.device,
) -> Optional[StepEncoding]:
    """Build token ids for predicting 1-indexed target_step from gold prefix."""
    if target_step < 1 or target_step > len(steps):
        return None

    source_ids = encode_text(tokenizer, source + " ")
    ids: List[int] = list(source_ids)
    spans: Dict[int, Tuple[int, int]] = {0: (0, len(source_ids))}

    for prev_step in range(1, target_step):
        step_text = steps[prev_step - 1]
        step_ids = encode_text(tokenizer, step_text)
        start = len(ids)
        ids.extend(step_ids)
        end = len(ids)
        spans[prev_step] = (start, end)

        if prev_step == len(steps) - 1:
            ids.extend(encode_text(tokenizer, " | "))
        else:
            ids.extend(encode_text(tokenizer, ", "))

    target_ids = encode_text(tokenizer, steps[target_step - 1])
    if len(target_ids) == 0 or len(ids) == 0:
        return None

    target_start = len(ids)
    ids.extend(target_ids)

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    return StepEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_start=target_start,
        target_ids=target_tensor,
        spans=spans,
    )


def build_free_prefix_encoding(
    tokenizer,
    source: str,
    generated_prefix: str,
    target: str,
    device: torch.device,
) -> Optional[StepEncoding]:
    prompt = source + " " + generated_prefix
    if generated_prefix and not generated_prefix.endswith(" "):
        prompt += " "
    prompt_ids = encode_text(tokenizer, prompt)
    target_ids = encode_text(tokenizer, target)
    if len(prompt_ids) == 0 or len(target_ids) == 0:
        return None
    ids = prompt_ids + target_ids
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return StepEncoding(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, device=device),
        target_start=len(prompt_ids),
        target_ids=torch.tensor(target_ids, dtype=torch.long, device=device),
        spans={},
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


def jacobian_output_vector(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    embeds: torch.Tensor,
    max_target_tokens: int,
    jacobian_output: str,
) -> torch.Tensor:
    """Continuous step map used as the Jacobian output.

    selected_logprob is the original diagnostic map. expected_embedding maps
    the next-step distribution back into token-embedding space, so rho composes
    embedding-space perturbations with embedding-space outputs.
    """
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return torch.empty(0, device=embeds.device)

    outputs = model(inputs_embeds=embeds, attention_mask=enc.attention_mask.detach())
    positions = torch.arange(
        enc.target_start - 1,
        enc.target_start - 1 + target_ids.numel(),
        device=embeds.device,
    )
    logits = outputs.logits[0, positions, :].float()

    if jacobian_output == "selected_logprob":
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[torch.arange(target_ids.numel(), device=embeds.device), target_ids]

    if jacobian_output == "expected_embedding":
        probs = F.softmax(logits, dim=-1)
        return (probs @ input_embeddings.weight.float()).reshape(-1)

    raise ValueError(f"Unknown jacobian_output: {jacobian_output}")


@torch.no_grad()
def expected_embedding_vector(
    model,
    input_embeddings: torch.nn.Module,
    enc: Optional[StepEncoding],
    max_target_tokens: int,
) -> Optional[torch.Tensor]:
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
        jacobian_output="expected_embedding",
    ).float().detach()


@torch.no_grad()
def expected_embedding_delta_norm(
    model,
    input_embeddings: torch.nn.Module,
    enc_a: Optional[StepEncoding],
    enc_b: Optional[StepEncoding],
    max_target_tokens: int,
) -> float:
    vec_a = expected_embedding_vector(model, input_embeddings, enc_a, max_target_tokens)
    vec_b = expected_embedding_vector(model, input_embeddings, enc_b, max_target_tokens)
    if vec_a is None or vec_b is None or vec_a.numel() != vec_b.numel():
        return float("nan")
    return float((vec_a - vec_b).float().norm().detach().cpu())


def empirical_lipschitz_constant(loss_a: float, loss_b: float, delta_norm: float) -> float:
    if not (np.isfinite(loss_a) and np.isfinite(loss_b) and np.isfinite(delta_norm)):
        return float("nan")
    if delta_norm < EPS:
        return 0.0 if abs(loss_b - loss_a) < EPS else float("nan")
    return abs(loss_b - loss_a) / delta_norm


@torch.no_grad()
def expected_embedding_step_error(
    model,
    input_embeddings: torch.nn.Module,
    enc: Optional[StepEncoding],
    max_target_tokens: int,
) -> Optional[float]:
    if enc is None:
        return None
    target_ids = capped_target_ids(enc, max_target_tokens)
    if target_ids.numel() == 0:
        return None

    pred_vec = expected_embedding_vector(model, input_embeddings, enc, max_target_tokens)
    if pred_vec is None:
        return None
    pred = pred_vec.view(target_ids.numel(), -1)
    gold = input_embeddings(target_ids).float()
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


def teacher_forced_step_losses(model, tokenizer, seq: StepSequence, device: torch.device) -> List[Optional[float]]:
    losses: List[Optional[float]] = []
    for target_step in range(1, len(seq.steps) + 1):
        enc = build_step_encoding(tokenizer, seq.source, seq.steps, target_step, device)
        losses.append(step_ce_loss(model, enc))
    return losses


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
) -> float:
    """Estimate the spectral norm of the selected step map with JVP/VJP power iteration."""
    start, end = span
    if end <= start:
        return 0.0

    base_embeds = input_embeddings(enc.input_ids).detach()
    base_attention = enc.attention_mask.detach()
    span_shape = base_embeds[:, start:end, :].shape
    x0 = base_embeds[:, start:end, :].detach().clone()

    def fn(x: torch.Tensor) -> torch.Tensor:
        embeds = base_embeds.clone()
        embeds[:, start:end, :] = x.view(span_shape).to(dtype=embeds.dtype)
        return jacobian_output_vector(
            model,
            input_embeddings,
            enc,
            embeds,
            max_target_tokens=max_target_tokens,
            jacobian_output=jacobian_output,
        )

    if fn(x0).numel() == 0:
        return 0.0

    v = _normalize_like(torch.randn_like(x0))
    sigma = torch.tensor(0.0, device=x0.device)

    for _ in range(max(1, power_iters)):
        _, jv = torch.autograd.functional.jvp(fn, (x0,), (v,), create_graph=False, strict=False)
        sigma = jv.float().norm()
        if not torch.isfinite(sigma) or sigma.item() < EPS:
            return 0.0

        u = (jv.float() / sigma).detach()
        x_req = x0.detach().clone().requires_grad_(True)
        y = fn(x_req)
        (jt_u,) = torch.autograd.grad(y, x_req, grad_outputs=u.to(y.dtype), retain_graph=False)
        v_norm = jt_u.float().norm()
        if not torch.isfinite(v_norm) or v_norm.item() < EPS:
            return float(sigma.detach().cpu())
        v = (jt_u.float() / v_norm).to(dtype=x0.dtype).detach()

    _, jv = torch.autograd.functional.jvp(fn, (x0,), (v,), create_graph=False, strict=False)
    sigma = jv.float().norm()
    if not torch.isfinite(sigma):
        return float("nan")
    return float(sigma.detach().cpu())


def estimate_jacobian_spectral_norm_rowgrad(
    model,
    input_embeddings: torch.nn.Module,
    enc: StepEncoding,
    span: Tuple[int, int],
    power_iters: int,
    max_target_tokens: int,
    jacobian_output: str,
) -> float:
    """Compute the selected Jacobian rows with first-order reverse-mode gradients.

    This is exact but only practical for low-dimensional outputs. For
    expected_embedding, prefer the JVP method because the output dimension is
    max_target_tokens * hidden_size.
    """
    if jacobian_output == "expected_embedding":
        raise ValueError("--jacobian_method rowgrad is too expensive for --jacobian_output expected_embedding; use --jacobian_method jvp")

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
        )
    raise ValueError(f"Unknown Jacobian method: {method}")


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
    base_attention = enc.attention_mask.detach()
    span_shape = base_embeds[:, start:end, :].shape
    x0 = base_embeds[:, start:end, :].detach().clone()

    def fn(x: torch.Tensor) -> torch.Tensor:
        embeds = base_embeds.clone()
        embeds[:, start:end, :] = x.view(span_shape).to(dtype=embeds.dtype)
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


def compute_rho(jac_norms: Dict[Tuple[int, int], float], total_steps: int) -> Dict[Tuple[int, int], float]:
    rho: Dict[Tuple[int, int], float] = {}
    for t in range(2, total_steps + 1):
        for i in range(1, t):
            val = jac_norms.get((t, i), 0.0)
            for k in range(i + 1, t):
                val += jac_norms.get((t, k), 0.0) * rho.get((k, i), 0.0)
            rho[(t, i)] = val
    return rho


def local_log_amplifications(jac_norms: Dict[Tuple[int, int], float], total_steps: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for i in range(1, total_steps):
        log_amp = 0.0
        for k in range(i + 1, total_steps + 1):
            log_amp += math.log(max(jac_norms.get((k, k - 1), 0.0), EPS))
        result[i] = log_amp
    return result


def prefix_log_amplifications_from_first(jac_norms: Dict[Tuple[int, int], float], total_steps: int) -> Dict[int, float]:
    result: Dict[int, float] = {1: 0.0}
    log_amp = 0.0
    for t in range(2, total_steps + 1):
        log_amp += math.log(max(jac_norms.get((t, t - 1), 0.0), EPS))
        result[t] = log_amp
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
        summary_rows.append(
            {
                "task": task,
                "length": length,
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
        plt.figure()
        plt.scatter([r["log_bound_proxy"] for r in finite_rows], [r["loss_gap"] for r in finite_rows], s=14)
        plt.xlabel("log bound proxy")
        plt.ylabel("generated final loss - teacher final loss")
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
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "first_full_jacobian_heatmap.png"), dpi=180)
        plt.close()


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


def load_checkpoint(model, checkpoint_path: Optional[str], device: torch.device) -> Optional[str]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if checkpoint_path is None:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        checkpoint = {k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()}

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    model.to(device)
    return checkpoint_path


def evaluate(args) -> None:
    validate_quant_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    args.devices = [args.device]
    model, tokenizer, _, _, _ = initialize_model(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint_path = load_checkpoint(model, args.checkpoint_path, device)
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

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("Model does not expose input embeddings.")

    example_rows: List[Dict] = []
    jacobian_rows: List[Dict] = []
    rho_prefix_rows: List[Dict] = []
    generated_step_loss_rows: List[Dict] = []
    finite_difference_rows: List[Dict] = []
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
            if len(seq.steps) < args.min_jacobian_steps:
                continue
            if args.max_jacobian_steps > 0 and len(seq.steps) > args.max_jacobian_steps:
                seq.steps = seq.steps[: args.max_jacobian_steps]
            if len(seq.steps) == 0:
                continue

            total_steps = len(seq.steps)
            with torch.no_grad():
                step_losses = teacher_forced_step_losses(model, tokenizer, seq, device)
            teacher_final_loss = metric_float(step_losses[-1])
            if args.jacobian_output == "expected_embedding":
                train_error_type = "expected_embedding_l2"
                train_errors = []
                for target_step in range(1, total_steps + 1):
                    enc = build_step_encoding(tokenizer, seq.source, seq.steps, target_step, device)
                    train_errors.append(
                        metric_float(
                            expected_embedding_step_error(
                                model,
                                input_embeddings,
                                enc,
                                args.max_target_tokens_for_jacobian,
                            )
                        )
                    )
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
            generated_steps, pred_final, generated_prefix = split_generated_answer(generated_answer)
            generated_sequence_steps = list(generated_steps) + ([pred_final] if pred_final else [])
            gold_final = seq.steps[-1]
            teacher_final_enc = build_step_encoding(tokenizer, seq.source, seq.steps, total_steps, device)
            generated_final_enc = build_free_prefix_encoding(
                tokenizer,
                seq.source,
                generated_prefix,
                gold_final,
                device,
            )
            generated_final_loss = metric_float(step_ce_loss(model, generated_final_enc))
            final_conditioning_step = total_steps - 1 if total_steps > 1 else 0
            final_ce_lipschitz_span = None
            if teacher_final_enc is not None and final_conditioning_step in teacher_final_enc.spans:
                final_ce_lipschitz_span = teacher_final_enc.spans[final_conditioning_step]
                if args.jacobian_output == "expected_embedding" and final_conditioning_step > 0:
                    final_ce_lipschitz_span = cap_step_span(
                        final_ce_lipschitz_span,
                        args.max_target_tokens_for_jacobian,
                    )
            final_ce_lipschitz_noise = estimate_ce_embedding_lipschitz_by_noise(
                model,
                input_embeddings,
                teacher_final_enc,
                final_ce_lipschitz_span,
                args.ce_lipschitz_noise_checks,
                args.ce_lipschitz_noise_epsilon,
            )
            final_expected_embedding_delta_norm = expected_embedding_delta_norm(
                model,
                input_embeddings,
                teacher_final_enc,
                generated_final_enc,
                args.max_target_tokens_for_jacobian,
            )
            final_loss_gap = (
                generated_final_loss - teacher_final_loss
                if np.isfinite(generated_final_loss) and np.isfinite(teacher_final_loss)
                else float("nan")
            )
            empirical_lipschitz_C = empirical_lipschitz_constant(
                teacher_final_loss,
                generated_final_loss,
                final_expected_embedding_delta_norm,
            )
            final_metrics = compute_accuracy([pred_final], [[gold_final]])
            generated_final_correct = final_metrics["accuracy"] / 100.0
            generated_edit_distance = final_metrics["edit_distance"]

            for target_step in range(1, total_steps + 1):
                step_teacher_enc = build_step_encoding(tokenizer, seq.source, seq.steps, target_step, device)
                step_generated_prefix = generated_prefix_for_step(
                    generated_sequence_steps,
                    target_step,
                    total_steps,
                )
                step_generated_enc = build_free_prefix_encoding(
                    tokenizer,
                    seq.source,
                    step_generated_prefix,
                    seq.steps[target_step - 1],
                    device,
                )
                generated_step_loss = metric_float(step_ce_loss(model, step_generated_enc))
                teacher_step_loss = metric_float(step_losses[target_step - 1])
                step_loss_gap = (
                    generated_step_loss - teacher_step_loss
                    if np.isfinite(generated_step_loss) and np.isfinite(teacher_step_loss)
                    else float("nan")
                )
                step_expected_embedding_delta_norm = expected_embedding_delta_norm(
                    model,
                    input_embeddings,
                    step_teacher_enc,
                    step_generated_enc,
                    args.max_target_tokens_for_jacobian,
                )
                step_empirical_lipschitz_C = empirical_lipschitz_constant(
                    teacher_step_loss,
                    generated_step_loss,
                    step_expected_embedding_delta_norm,
                )
                step_conditioning_step = target_step - 1 if target_step > 1 else 0
                step_ce_lipschitz_span = None
                if step_teacher_enc is not None and step_conditioning_step in step_teacher_enc.spans:
                    step_ce_lipschitz_span = step_teacher_enc.spans[step_conditioning_step]
                    if args.jacobian_output == "expected_embedding" and step_conditioning_step > 0:
                        step_ce_lipschitz_span = cap_step_span(
                            step_ce_lipschitz_span,
                            args.max_target_tokens_for_jacobian,
                        )
                step_ce_lipschitz_noise = estimate_ce_embedding_lipschitz_by_noise(
                    model,
                    input_embeddings,
                    step_teacher_enc,
                    step_ce_lipschitz_span,
                    args.ce_lipschitz_noise_checks,
                    args.ce_lipschitz_noise_epsilon,
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
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "t": target_step,
                        "num_steps": total_steps,
                        "teacher_step_loss": teacher_step_loss,
                        "generated_prefix_step_loss": generated_step_loss,
                        "step_loss_gap": step_loss_gap,
                        "expected_embedding_delta_norm": step_expected_embedding_delta_norm,
                        "empirical_lipschitz_C": step_empirical_lipschitz_C,
                        "ce_lipschitz_noise_max": step_ce_lipschitz_noise["ce_lipschitz_noise_max"],
                        "ce_lipschitz_noise_mean": step_ce_lipschitz_noise["ce_lipschitz_noise_mean"],
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
            with attention_kernel_context(args.attention_backend):
                for target_step, source_step in pairs:
                    enc = build_step_encoding(tokenizer, seq.source, seq.steps, target_step, device)
                    if enc is None or source_step not in enc.spans:
                        jac_norm = 0.0
                    else:
                        span = enc.spans[source_step]
                        if args.jacobian_output == "expected_embedding" and source_step > 0:
                            span = cap_step_span(span, args.max_target_tokens_for_jacobian)
                        jac_norm = estimate_jacobian_spectral_norm(
                            model,
                            input_embeddings,
                            enc,
                            span,
                            power_iters=args.power_iters,
                            max_target_tokens=args.max_target_tokens_for_jacobian,
                            method=args.jacobian_method,
                            jacobian_output=args.jacobian_output,
                        )
                        if finite_difference_count < args.finite_difference_checks:
                            check = finite_difference_directional_check(
                                model,
                                input_embeddings,
                                enc,
                                span,
                                max_target_tokens=args.max_target_tokens_for_jacobian,
                                epsilon=args.finite_difference_epsilon,
                                jacobian_output=args.jacobian_output,
                            )
                            finite_difference_rows.append(
                                {
                                    "task": task_name,
                                    "checkpoint": checkpoint_path or "",
                                    "jacobian_output": args.jacobian_output,
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

            rho = compute_rho(jac_norms, total_steps) if args.full_jacobian else {}
            local_logs = local_log_amplifications(jac_norms, total_steps)
            prefix_logs = prefix_log_amplifications_from_first(jac_norms, total_steps)
            adjacent_norms = [jac_norms.get((t, t - 1), float("nan")) for t in range(2, total_steps + 1)]
            input_jac_norms = [
                jac_norms.get((t, 0), float("nan"))
                for t in range(1, total_steps + 1)
                if (t, 0) in jac_norms
            ]
            input_jac_norm = float(np.nanmean(input_jac_norms)) if input_jac_norms else float("nan")
            log_input_jac_norm = math.log(max(input_jac_norm, EPS)) if np.isfinite(input_jac_norm) else float("nan")

            if args.jacobian_output == "expected_embedding":
                if args.full_jacobian:
                    bound_terms = [
                        rho.get((total_steps, i), 0.0) * train_errors[i - 1]
                        for i in range(1, total_steps)
                        if np.isfinite(train_errors[i - 1])
                    ]
                else:
                    bound_terms = [
                        math.exp(local_logs[i]) * train_errors[i - 1]
                        for i in range(1, total_steps)
                        if np.isfinite(train_errors[i - 1]) and np.isfinite(local_logs[i])
                    ]
                bound_proxy = float(np.sum(bound_terms)) if bound_terms else float("nan")
                log_bound_proxy = math.log(max(bound_proxy, EPS)) if np.isfinite(bound_proxy) else float("nan")
                if (
                    not np.isfinite(bound_proxy)
                    and total_steps == 1
                    and len(train_errors) > 0
                    and np.isfinite(input_jac_norm)
                    and np.isfinite(train_errors[0])
                ):
                    bound_proxy = input_jac_norm * train_errors[0]
                    log_bound_proxy = math.log(max(bound_proxy, EPS))
            else:
                bound_proxy = float("nan")
                log_bound_proxy = float("nan")
            bound_proxy_valid = bool(args.jacobian_output == "expected_embedding" and np.isfinite(bound_proxy))
            calibrated_bound_proxy = (
                empirical_lipschitz_C * bound_proxy
                if np.isfinite(empirical_lipschitz_C) and np.isfinite(bound_proxy)
                else float("nan")
            )
            noise_calibrated_bound_proxy = (
                final_ce_lipschitz_noise["ce_lipschitz_noise_max"] * bound_proxy
                if np.isfinite(final_ce_lipschitz_noise["ce_lipschitz_noise_max"]) and np.isfinite(bound_proxy)
                else float("nan")
            )

            for prefix_t in range(1, total_steps + 1):
                adjacent_log_rho_t_1 = prefix_logs.get(prefix_t, float("nan"))
                adjacent_rho_t_1 = math.exp(adjacent_log_rho_t_1) if np.isfinite(adjacent_log_rho_t_1) else float("nan")
                recursive_rho_t_1 = 1.0 if prefix_t == 1 else rho.get((prefix_t, 1), float("nan"))
                recursive_log_rho_t_1 = (
                    math.log(max(recursive_rho_t_1, EPS)) if np.isfinite(recursive_rho_t_1) else float("nan")
                )
                rho_prefix_rows.append(
                    {
                        "task": task_name,
                        "checkpoint": checkpoint_path or "",
                        "jacobian_output": args.jacobian_output,
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "t": prefix_t,
                        "num_steps": total_steps,
                        "adjacent_rho_t_1": adjacent_rho_t_1,
                        "adjacent_log_rho_t_1": adjacent_log_rho_t_1,
                        "recursive_rho_t_1": recursive_rho_t_1,
                        "recursive_log_rho_t_1": recursive_log_rho_t_1,
                        "computed_full_jacobian": bool(args.full_jacobian),
                    }
                )

            for (target_step, source_step), jac_norm in sorted(jac_norms.items()):
                jacobian_rows.append(
                    {
                        "task": task_name,
                        "checkpoint": checkpoint_path or "",
                        "jacobian_output": args.jacobian_output,
                        "split": args.split,
                        "length": length,
                        "sample_idx": sample_idx,
                        "i": target_step,
                        "j": source_step,
                        "jac_norm": jac_norm,
                        "log_jac_norm": math.log(max(jac_norm, EPS)) if np.isfinite(jac_norm) else float("nan"),
                        "rho_T_i": rho.get((total_steps, source_step), float("nan")),
                        "local_log_amp_i": local_logs.get(source_step, float("nan")),
                        "step_loss_i": metric_float(step_losses[source_step - 1]) if source_step > 0 else float("nan"),
                        "train_error_i": train_errors[source_step - 1] if source_step > 0 else float("nan"),
                        "train_error_type": train_error_type,
                        "teacher_final_loss": teacher_final_loss,
                        "generated_final_loss": generated_final_loss,
                        "generated_final_correct": generated_final_correct,
                        "computed_full_jacobian": bool(args.full_jacobian),
                    }
                )

            example_rows.append(
                {
                    "task": task_name,
                    "checkpoint": checkpoint_path or "",
                    "jacobian_output": args.jacobian_output,
                    "split": args.split,
                    "length": length,
                    "sample_idx": sample_idx,
                    "num_steps": total_steps,
                    "teacher_final_loss": teacher_final_loss,
                    "generated_final_loss": generated_final_loss,
                    "loss_gap": final_loss_gap,
                    "generated_final_correct": generated_final_correct,
                    "generated_edit_distance": generated_edit_distance,
                    "bound_proxy": bound_proxy,
                    "log_bound_proxy": log_bound_proxy,
                    "bound_proxy_valid": bound_proxy_valid,
                    "calibrated_bound_proxy": calibrated_bound_proxy,
                    "noise_calibrated_bound_proxy": noise_calibrated_bound_proxy,
                    "final_expected_embedding_delta_norm": final_expected_embedding_delta_norm,
                    "empirical_lipschitz_C": empirical_lipschitz_C,
                    "ce_lipschitz_noise_max": final_ce_lipschitz_noise["ce_lipschitz_noise_max"],
                    "ce_lipschitz_noise_mean": final_ce_lipschitz_noise["ce_lipschitz_noise_mean"],
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
            "split",
            "length",
            "sample_idx",
            "i",
            "j",
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
        os.path.join(args.output_dir, "rho_prefix_rows.csv"),
        rho_prefix_rows,
        [
            "task",
            "checkpoint",
            "jacobian_output",
            "split",
            "length",
            "sample_idx",
            "t",
            "num_steps",
            "adjacent_rho_t_1",
            "adjacent_log_rho_t_1",
            "recursive_rho_t_1",
            "recursive_log_rho_t_1",
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
            "split",
            "length",
            "sample_idx",
            "t",
            "num_steps",
            "teacher_step_loss",
            "generated_prefix_step_loss",
            "step_loss_gap",
            "expected_embedding_delta_norm",
            "empirical_lipschitz_C",
            "ce_lipschitz_noise_max",
            "ce_lipschitz_noise_mean",
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
            "split",
            "length",
            "sample_idx",
            "num_steps",
            "teacher_final_loss",
            "generated_final_loss",
            "loss_gap",
            "generated_final_correct",
            "generated_edit_distance",
            "bound_proxy",
            "log_bound_proxy",
            "bound_proxy_valid",
            "calibrated_bound_proxy",
            "noise_calibrated_bound_proxy",
            "final_expected_embedding_delta_norm",
            "empirical_lipschitz_C",
            "ce_lipschitz_noise_max",
            "ce_lipschitz_noise_mean",
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
    parser.add_argument("--max_jacobian_steps", type=int, default=0)
    parser.add_argument("--power_iters", type=int, default=5)
    parser.add_argument("--max_target_tokens_for_jacobian", type=int, default=8)
    parser.add_argument("--jacobian_method", type=str, choices=["jvp", "rowgrad"], default="jvp")
    parser.add_argument(
        "--jacobian_output",
        type=str,
        choices=["selected_logprob", "expected_embedding"],
        default="selected_logprob",
        help=(
            "selected_logprob reproduces the original log-probability diagnostic. "
            "expected_embedding maps each predicted step to expected token embeddings, "
            "so recursive rho composes maps in the same embedding space."
        ),
    )
    parser.add_argument("--attention_backend", type=str, choices=["math", "auto"], default="math")
    parser.add_argument("--include_input_jacobian", action="store_true")
    parser.add_argument("--full_jacobian", action="store_true")
    parser.add_argument("--finite_difference_checks", type=int, default=0)
    parser.add_argument("--finite_difference_epsilon", type=float, default=1e-3)
    parser.add_argument("--ce_lipschitz_noise_checks", type=int, default=0)
    parser.add_argument("--ce_lipschitz_noise_epsilon", type=float, default=1e-3)
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
