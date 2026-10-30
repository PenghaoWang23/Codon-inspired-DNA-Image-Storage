from __future__ import annotations
import math
from pathlib import Path
from typing import Sequence, Union
import torch
import torch.nn as nn
BASE_TO_INDEX = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': 4}
INDEX_TO_BASE = {value: key for key, value in BASE_TO_INDEX.items()}
MODEL_EMBED_DIM = 64
MODEL_HEADS = 2
MODEL_LAYERS = 2
MODEL_HIDDEN_DIM = 256

class PositionalEncoding(nn.Module):

    def __init__(self, dimensions: int, max_length: int=1000, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimensions, 2) * (-math.log(10000.0) / dimensions))
        encoding = torch.zeros(max_length, 1, dimensions)
        encoding[:, 0, 0::2] = torch.sin(positions * scale)
        encoding[:, 0, 1::2] = torch.cos(positions * scale)
        self.register_buffer('pe', encoding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.pe[:inputs.size(1)].permute(1, 0, 2)
        return self.dropout(inputs)

class TransformerDNA(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, MODEL_EMBED_DIM, padding_idx=BASE_TO_INDEX['-'])
        self.conv_block = nn.Sequential(nn.Conv1d(MODEL_EMBED_DIM, 2 * MODEL_EMBED_DIM, kernel_size=3, padding=1, groups=MODEL_EMBED_DIM, bias=False), nn.BatchNorm1d(2 * MODEL_EMBED_DIM), nn.ReLU(), nn.Dropout(0.3), nn.Conv1d(2 * MODEL_EMBED_DIM, 2 * MODEL_EMBED_DIM, kernel_size=5, padding=2, groups=2 * MODEL_EMBED_DIM, bias=False), nn.BatchNorm1d(2 * MODEL_EMBED_DIM), nn.ReLU(), nn.Dropout(0.3), nn.Conv1d(2 * MODEL_EMBED_DIM, MODEL_EMBED_DIM, kernel_size=1, bias=False), nn.BatchNorm1d(MODEL_EMBED_DIM), nn.ReLU())
        self.copy_attention = nn.MultiheadAttention(MODEL_EMBED_DIM, MODEL_HEADS, batch_first=True)
        self.pos_encoder = PositionalEncoding(MODEL_EMBED_DIM)
        layer = nn.TransformerEncoderLayer(d_model=MODEL_EMBED_DIM, nhead=MODEL_HEADS, dim_feedforward=MODEL_HIDDEN_DIM, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, MODEL_LAYERS)
        self.fc = nn.Linear(MODEL_EMBED_DIM, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, copies, sequence_length = inputs.shape
        embedded = self.embedding(inputs)
        convolved = embedded.reshape(batch_size * copies, sequence_length, MODEL_EMBED_DIM).permute(0, 2, 1)
        convolved = self.conv_block(convolved).permute(0, 2, 1)
        convolved = convolved.reshape(batch_size, copies, sequence_length, MODEL_EMBED_DIM)
        fused_positions = []
        for position in range(sequence_length):
            features = convolved[:, :, position, :]
            attended, _ = self.copy_attention(features, features, features)
            fused_positions.append(attended.mean(dim=1))
        fused = torch.stack(fused_positions, dim=1)
        encoded = self.transformer_encoder(self.pos_encoder(fused))
        return self.fc(encoded)

def select_device(requested: str='auto') -> torch.device:
    if requested == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is not available')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)

def load_model(path: Union[str, Path], device: Union[str, torch.device]='auto') -> TransformerDNA:
    device = select_device(device) if isinstance(device, str) else device
    model = TransformerDNA().to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def reconstruct(model: TransformerDNA, copies: Sequence[str], device: Union[str, torch.device]='auto') -> str:
    device = select_device(device) if isinstance(device, str) else device
    if not copies:
        raise ValueError('At least one noisy copy is required')
    max_length = max((len(copy) for copy in copies))
    indexed = [[BASE_TO_INDEX.get(base, BASE_TO_INDEX['-']) for base in copy] + [BASE_TO_INDEX['-']] * (max_length - len(copy)) for copy in copies]
    inputs = torch.tensor([indexed], dtype=torch.long, device=device)
    with torch.inference_mode():
        prediction = model(inputs).argmax(dim=2).squeeze(0).cpu().tolist()
    return ''.join((INDEX_TO_BASE[index] for index in prediction if index != 4))
__all__ = ['BASE_TO_INDEX', 'INDEX_TO_BASE', 'MODEL_EMBED_DIM', 'MODEL_HEADS', 'MODEL_HIDDEN_DIM', 'MODEL_LAYERS', 'TransformerDNA', 'load_model', 'reconstruct', 'select_device']
