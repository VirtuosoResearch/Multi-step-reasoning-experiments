import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
    _CallbackBase = pl.Callback
except ModuleNotFoundError:
    _CallbackBase = object


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """Straight-through LSQ-style quantizer used by Quant."""

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            qn = -1
            qp = 1
        else:
            qn = -(2 ** (num_bits - 1))
            qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        grad_scale = 1.0 / math.sqrt(input.numel()) if not qp else 1.0 / math.sqrt(input.numel() * qp)

        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, qn, qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(qn, qp)
        return q_w * alpha

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, qn, qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < qn).float()
        indicate_big = (q_w > qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        if ctx.num_bits == 1:
            grad_alpha = input_.sign() * grad_output * grad_scale
        else:
            grad_alpha = (
                indicate_small * qn
                + indicate_big * qp
                + indicate_middle * (-q_w + q_w.round())
            ) * grad_output * grad_scale

        if layerwise:
            grad_alpha = grad_alpha.sum().unsqueeze(dim=0)
        else:
            grad_alpha = grad_alpha.sum(dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """Quant's stretched elastic quantizer for ternary / 2-bit settings."""

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            qn = -1
            qp = 1
        else:
            qn = -(2 ** (num_bits - 1))
            qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        grad_scale = 1.0 / math.sqrt(input.numel()) if not qp else 1.0 / math.sqrt(input.numel() * qp)

        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        qp = (n_levels - shift) / n_levels
        qn = -qp
        ctx.other = grad_scale, qn, qp, layerwise

        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift)
                + shift
            ) / n_levels
        return q_w * alpha

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, qn, qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5

        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            grad_alpha = input_.sign() * grad_output * grad_scale
        else:
            quantized = (
                torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift)
                + shift
            ) / n_levels
            grad_alpha = (
                indicate_small * qn
                + indicate_big * qp
                + indicate_middle * (-q_w + quantized)
            ) * grad_output * grad_scale

        if layerwise:
            grad_alpha = grad_alpha.sum().unsqueeze(dim=0)
        else:
            grad_alpha = grad_alpha.sum(dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


@dataclass
class QuantConfig:
    w_bits: int = 2
    noise_injection: bool = False
    pre_quantization_noise: bool = False
    post_quantization_noise: bool = False
    noise_sigma_weights: float = 0.0
    noise_sigma_clipvals: float = 0.0
    initialize_noise: bool = False
    trainable_noise_scale: bool = False
    weight_layerwise: bool = False


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        config: Optional[QuantConfig] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.w_bits = int(w_bits)
        self.weight_layerwise = weight_layerwise

        if self.w_bits < 16:
            clip_shape = (1,) if weight_layerwise else (out_features, 1)
            self.weight_clip_val = nn.Parameter(torch.empty(clip_shape, device=device, dtype=dtype))

        self.noise_injection = False
        self.pre_quantization_noise = False
        self.post_quantization_noise = False
        self.initialize_noise = False
        self.noise_sigma_weights = 0.0
        self.noise_sigma_clipvals = 0.0
        self.trainable_noise_scale = False
        self._frozen_weight_noise = None
        self._frozen_clip_noise = None
        self._frozen_post_noise = None
        self._cached_post_sigma = None

        if config is not None:
            self._apply_noise_config(config)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuantConfig) -> "QuantizeLinear":
        quantized = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            w_bits=config.w_bits,
            weight_layerwise=config.weight_layerwise,
            config=config,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        quantized.weight.data.copy_(linear.weight.data)
        quantized.weight.requires_grad = linear.weight.requires_grad
        if linear.bias is not None:
            quantized.bias.data.copy_(linear.bias.data)
            quantized.bias.requires_grad = linear.bias.requires_grad
        quantized._initialize_clip_values()
        return quantized

    def _apply_noise_config(self, config: QuantConfig):
        self.noise_injection = bool(config.noise_injection)
        self.pre_quantization_noise = bool(config.pre_quantization_noise)
        self.post_quantization_noise = bool(config.post_quantization_noise)
        self.initialize_noise = bool(config.initialize_noise)
        self.noise_sigma_weights = float(config.noise_sigma_weights)
        self.noise_sigma_clipvals = float(config.noise_sigma_clipvals)
        self.trainable_noise_scale = bool(config.trainable_noise_scale)

        if not self.noise_injection:
            return

        weight_sigma_init = torch.tensor(self.noise_sigma_weights, dtype=torch.float32)
        clip_sigma_init = torch.tensor(self.noise_sigma_clipvals, dtype=torch.float32)
        if self.trainable_noise_scale:
            self.weight_noise_scale = nn.Parameter(weight_sigma_init)
            self.clip_noise_scale = nn.Parameter(clip_sigma_init)
        else:
            self.register_buffer("weight_noise_scale", weight_sigma_init, persistent=False)
            self.register_buffer("clip_noise_scale", clip_sigma_init, persistent=False)

    def _initialize_clip_values(self):
        if self.w_bits >= 16 or not hasattr(self, "weight_clip_val"):
            return
        with torch.no_grad():
            weight = self.weight.detach()
            if self.w_bits == 1:
                scale = torch.mean(weight.abs(), dim=-1, keepdim=True)
            elif self.w_bits == 0 or self.w_bits == 2:
                scale, _ = torch.max(weight.abs(), dim=-1, keepdim=True)
            elif self.w_bits == 3 or self.w_bits == 4:
                xmax, _ = torch.max(weight.abs(), dim=-1, keepdim=True)
                maxq = 2 ** (self.w_bits - 1) - 1
                scale = xmax / maxq
            else:
                raise NotImplementedError(f"Unsupported Quant bit width: {self.w_bits}")

            if self.weight_layerwise:
                scale = scale.mean().view(1)
            self.weight_clip_val.copy_(scale.clamp_min(1e-5).to(self.weight_clip_val.dtype))

    def reset_noise_buffers(self):
        self._frozen_weight_noise = None
        self._frozen_clip_noise = None
        self._frozen_post_noise = None
        self._cached_post_sigma = None

    def _get_sigma_tensor(self, attr, default, device, dtype):
        if attr is None:
            if default == 0.0:
                return None
            return torch.tensor(default, device=device, dtype=dtype)
        if isinstance(attr, torch.Tensor):
            return attr.to(device=device, dtype=dtype)
        return torch.tensor(float(default), device=device, dtype=dtype)

    def _maybe_apply_noise(self, tensor, sigma_tensor, frozen_attr_name):
        if tensor is None or sigma_tensor is None:
            return tensor
        if sigma_tensor.abs().item() == 0.0:
            return tensor

        if self.initialize_noise:
            frozen_noise = getattr(self, frozen_attr_name)
            if frozen_noise is None or frozen_noise.shape != tensor.shape:
                frozen_noise = torch.randn_like(tensor)
                setattr(self, frozen_attr_name, frozen_noise)
            noise = frozen_noise
        else:
            noise = torch.randn_like(tensor)
        return tensor + noise * sigma_tensor

    def _quantize_weight(self, weight, clip_val, dtype):
        if self.w_bits >= 16:
            return weight.to(dtype)
        if self.w_bits == 2 or self.w_bits == 0:
            return StretchedElasticQuant.apply(
                weight,
                clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(dtype)
        if self.w_bits <= 4:
            return LsqBinaryTernaryExtension.apply(
                weight,
                clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(dtype)
        raise NotImplementedError(f"Unsupported Quant bit width: {self.w_bits}")

    def _prepare_quant_inputs(self, apply_noise: bool):
        weight = self.weight
        clip_val = getattr(self, "weight_clip_val", None)

        if self.noise_injection and apply_noise and self.training:
            device = weight.device
            dtype = weight.dtype
            weight_sigma = self._get_sigma_tensor(
                self.weight_noise_scale,
                self.noise_sigma_weights,
                device,
                dtype,
            )
            clip_sigma = None
            if clip_val is not None:
                clip_sigma = self._get_sigma_tensor(
                    self.clip_noise_scale,
                    self.noise_sigma_clipvals,
                    device,
                    clip_val.dtype,
                )

            if self.pre_quantization_noise:
                weight = self._maybe_apply_noise(weight, weight_sigma, "_frozen_weight_noise")
                if clip_val is not None:
                    clip_val = self._maybe_apply_noise(clip_val, clip_sigma, "_frozen_clip_noise")

            self._cached_post_sigma = weight_sigma
        else:
            self._cached_post_sigma = None

        return weight, clip_val

    def quantized_weight(self, dtype=None, apply_noise=False):
        weight, clip_val = self._prepare_quant_inputs(apply_noise)
        q_weight = self._quantize_weight(weight, clip_val, dtype or weight.dtype)
        if self.noise_injection and apply_noise and self.training and self.post_quantization_noise:
            sigma = self._cached_post_sigma
            if sigma is not None and sigma.abs().item() != 0.0:
                if self.initialize_noise:
                    post_noise = self._frozen_post_noise
                    if post_noise is None or post_noise.shape != q_weight.shape:
                        post_noise = torch.randn_like(q_weight)
                        self._frozen_post_noise = post_noise
                else:
                    post_noise = torch.randn_like(q_weight)
                q_weight = q_weight + post_noise * sigma.to(q_weight.dtype)
        return q_weight

    def forward(self, input_):
        weight = self.quantized_weight(dtype=input_.dtype, apply_noise=True)
        return nn.functional.linear(input_, weight, self.bias)


def normalize_quant_module_names(module_names: Sequence[str]) -> List[str]:
    if module_names is None:
        return []
    names = []
    for item in module_names:
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
                names.append(piece)
    return names


def _matches_module_name(full_name: str, leaf_name: str, target_modules: Iterable[str]) -> bool:
    for target in target_modules:
        if leaf_name == target or full_name.endswith(f".{target}") or full_name == target:
            return True
    return False


def _replace_named_linears(module: nn.Module, target_modules: Sequence[str], config: QuantConfig, prefix="") -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, QuantizeLinear):
            continue
        if isinstance(child, nn.Linear) and _matches_module_name(full_name, child_name, target_modules):
            setattr(module, child_name, QuantizeLinear.from_linear(child, config))
            replaced += 1
        else:
            replaced += _replace_named_linears(child, target_modules, config, full_name)
    return replaced


def apply_quant_full_model_quantization(
    model: nn.Module,
    target_modules: Sequence[str],
    config: QuantConfig,
) -> int:
    replaced = _replace_named_linears(model, target_modules, config)
    for module in model.modules():
        if isinstance(module, QuantizeLinear):
            module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
            if hasattr(module, "weight_clip_val"):
                module.weight_clip_val.requires_grad = True
    return replaced


def apply_quant_lora_quantization(model: nn.Module, config: QuantConfig) -> int:
    replaced = 0
    for module in model.modules():
        for attr_name in ("lora_A", "lora_B"):
            lora_layers = getattr(module, attr_name, None)
            if lora_layers is None or not hasattr(lora_layers, "items"):
                continue
            for adapter_name, layer in list(lora_layers.items()):
                if isinstance(layer, QuantizeLinear):
                    continue
                if not isinstance(layer, nn.Linear):
                    continue
                quantized = QuantizeLinear.from_linear(layer, config)
                quantized.weight.requires_grad = True
                if hasattr(quantized, "weight_clip_val"):
                    quantized.weight_clip_val.requires_grad = True
                lora_layers[adapter_name] = quantized
                replaced += 1
            print(f"Applied {config.w_bits}-bit quantization in {attr_name} of module {module._get_name()}")
    return replaced


class QuantReinitializeCallback(_CallbackBase):
    def __init__(self, reinitialize_steps: int, reinitialize_alpha: float = 0.2):
        self.reinitialize_steps = int(reinitialize_steps)
        self.reinitialize_alpha = float(reinitialize_alpha)
        self._last_reinitialized_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.reinitialize_steps <= 0:
            return
        step = int(trainer.global_step)
        if step == 0 or step == self._last_reinitialized_step:
            return
        if step % self.reinitialize_steps != 0:
            return
        self._interpolate_weights(pl_module.model)
        self._last_reinitialized_step = step

    def _interpolate_weights(self, model):
        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, QuantizeLinear):
                    continue
                if not module.weight.requires_grad:
                    continue
                quantized_weight = module.quantized_weight(
                    dtype=module.weight.dtype,
                    apply_noise=False,
                )
                module.weight.data.mul_(self.reinitialize_alpha).add_(
                    quantized_weight * (1 - self.reinitialize_alpha)
                )
                module.reset_noise_buffers()
