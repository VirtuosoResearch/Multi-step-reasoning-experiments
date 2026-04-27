import torch
import torch.nn as nn

from src.quantization import (
    QuantizeLinear,
    QuantConfig,
    apply_quant_full_model_quantization,
    apply_quant_lora_quantization,
)


def test_quant_linear_forward_backward_and_clip_grad():
    base = nn.Linear(8, 4, bias=True)
    quantized = QuantizeLinear.from_linear(base, QuantConfig(w_bits=2))

    x = torch.randn(3, 8)
    out = quantized(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert out.shape == (3, 4)
    assert quantized.weight.grad is not None
    assert quantized.weight_clip_val.grad is not None
    assert quantized.weight_clip_val.shape == (4, 1)


def test_quant_lora_quantization_replaces_trainable_lora_layers_only():
    class FakeLoraLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(8, 8)
            self.base_layer.requires_grad_(False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(8, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 8, bias=False)})

    model = FakeLoraLayer()
    replaced = apply_quant_lora_quantization(model, QuantConfig(w_bits=4))

    assert replaced == 2
    assert isinstance(model.lora_A["default"], QuantizeLinear)
    assert isinstance(model.lora_B["default"], QuantizeLinear)
    assert not isinstance(model.base_layer, QuantizeLinear)
    assert not model.base_layer.weight.requires_grad
    assert model.lora_A["default"].weight.requires_grad
    assert model.lora_A["default"].weight_clip_val.requires_grad


def test_quant_full_model_quantization_targets_named_projection_layers():
    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.q_proj = nn.Linear(8, 8)
            self.gate_proj = nn.Linear(8, 16)
            self.lm_head = nn.Linear(8, 16)
            self.norm = nn.LayerNorm(8)

    model = ToyBlock()
    replaced = apply_quant_full_model_quantization(
        model,
        target_modules=["q_proj", "gate_proj"],
        config=QuantConfig(w_bits=2),
    )

    assert replaced == 2
    assert isinstance(model.q_proj, QuantizeLinear)
    assert isinstance(model.gate_proj, QuantizeLinear)
    assert not isinstance(model.lm_head, QuantizeLinear)
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.norm, nn.LayerNorm)
    assert model.q_proj.weight.requires_grad
    assert model.q_proj.weight_clip_val.requires_grad
