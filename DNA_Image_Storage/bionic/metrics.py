from __future__ import annotations
import math
import numpy as np
from skimage.metrics import structural_similarity

def psnr(reference: np.ndarray, decoded: np.ndarray, per_channel: bool=True) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if reference.shape != decoded.shape:
        raise ValueError(f'Shape mismatch: {reference.shape} vs {decoded.shape}')
    if not per_channel:
        mse = float(np.mean((reference - decoded) ** 2))
        return float('inf') if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    values = []
    for channel in range(reference.shape[-1]):
        mse = float(np.mean((reference[..., channel] - decoded[..., channel]) ** 2))
        values.append(float('inf') if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse))
    return float(np.mean(values))

def ssim(reference: np.ndarray, decoded: np.ndarray, win_size: int=7) -> float:
    reference = np.asarray(reference, dtype=np.uint8)
    decoded = np.asarray(decoded, dtype=np.uint8)
    if reference.shape != decoded.shape:
        raise ValueError(f'Shape mismatch: {reference.shape} vs {decoded.shape}')
    return float(structural_similarity(reference, decoded, data_range=255, channel_axis=-1, win_size=win_size))
__all__ = ['psnr', 'ssim']
