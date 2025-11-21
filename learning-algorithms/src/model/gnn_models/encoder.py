import torch.nn as nn
import torch
import torch_scatter
from loguru import logger
from .utils import NaNException
## Base encoders and decoders

class NodeBaseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.lin(x)
        return x
    
class EdgeBaseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.lin(x)
        return x
    
class EdgeMaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Parameter(torch.randn(2, hidden_dim))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.lin[x]
        return x

_ENCODER_MAP = {
    ('node', 'scalar'): NodeBaseEncoder,
    ('node', 'mask'): NodeBaseEncoder,
    ('node', 'mask_one'): NodeBaseEncoder,
    ('edge', 'scalar'): EdgeBaseEncoder,
    ('edge', 'mask'): EdgeMaskEncoder,
    ('edge', 'mask_one'): EdgeMaskEncoder
}

class Encoder(nn.Module):
    def __init__(self, specs, hidden_dim=128):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.encoder = nn.ModuleDict()
        for k, v in specs.items():
            if k == "randomness": # randomness is not encoded
                continue
            stage, loc, type_, cat_dim = v
            if type_ == 'pointer':
                continue
            elif loc == 'edge':
                # logger.debug(f'Ignoring edge encoder for {k}')
                # continue
                self.encoder[k] = _ENCODER_MAP[(loc, type_)](1, hidden_dim)
            elif stage == 'hint':
                # logger.debug(f'Ignoring hint encoder for {k}')
                continue
            elif stage == 'output':
                # logger.debug(f'Ignoring output encoder for {k}')
                continue
            else:
                # Input DIM currently hardcoded to 1
                self.encoder[k] = _ENCODER_MAP[(loc, type_)](1, hidden_dim)

    def forward(self, batch):
        hidden = None; edge_attr = None
        for key in batch.inputs:
            stage, loc, type_, cat_dim = self.specs[key]
            if key == "randomness":
                continue
            # logger.debug(f"Encoding {key}")
            if type_ == 'pointer':
                continue
            if loc == 'edge':
                encoding = self.encoder[key](batch[key].reshape(-1))
                if edge_attr is None:
                    edge_attr = encoding
                else:
                    edge_attr += encoding
            else:
                encoding = self.encoder[key](batch[key])
                # check of nan
                if torch.isnan(encoding).any():
                    # logger.warning(f"NaN in encoded hidden state")
                    raise NaNException(f"NaN in encoded hidden state")
                if hidden is None:
                    hidden = encoding
                else:
                    hidden += encoding

        if edge_attr is not None:
            batch.weights = edge_attr
        randomness = batch.randomness if "randomness" in batch.inputs else None
        return hidden, randomness
