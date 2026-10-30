from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .codec import ADDRESS_NT, PAYLOAD_NT, Codebook, build_inner_rs_positions, decode_address, decode_inner_rs, encode_address, encode_inner_rs, split_payload
PIXELS_PER_STRAND = PAYLOAD_NT // 5

def encode_image(image: np.ndarray, codebook: Codebook, rs_bytes: int=4, use_address: bool=True) -> List[str]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an (H, W, 3) RGB image')
    image = np.asarray(image, dtype=np.uint8)
    positions = build_inner_rs_positions(PAYLOAD_NT)
    strands: List[str] = []
    address = 0
    for channel in range(3):
        pixels = list(image[..., channel].ravel())
        padding = -len(pixels) % PIXELS_PER_STRAND
        if padding:
            pixels.extend([0] * padding)
        dna = codebook.encode_pixels(pixels)
        for segment in split_payload(dna, PAYLOAD_NT):
            protected = encode_inner_rs(segment, positions, rs_bytes)
            prefix = encode_address(address) if use_address else ''
            strands.append(prefix + protected)
            address += 1
    return strands

def _next_valid_fill(items: Sequence[Tuple[int, str]], max_address: Optional[int]=None) -> List[Tuple[int, str]]:
    if not items:
        return []
    by_address: Dict[int, str] = dict(items)
    top = max(by_address) if max_address is None else int(max_address)
    top = max(top, max(by_address))
    first_valid = items[0][1]
    filled: Dict[int, str] = {}
    next_valid: Optional[str] = None
    for address in range(top, -1, -1):
        if address in by_address:
            next_valid = by_address[address]
        if next_valid is not None:
            filled[address] = next_valid
    for address in range(top + 1):
        filled.setdefault(address, first_valid)
    return [(address, filled[address]) for address in range(top + 1)]

def decode_image(strands: Sequence[str], codebook: Codebook, shape: Tuple[int, int], rs_bytes: int=4, use_address: bool=True, impute_missing: bool=True, max_address: Optional[int]=None) -> np.ndarray:
    positions = build_inner_rs_positions(PAYLOAD_NT)
    items: List[Tuple[int, str]] = []
    for order, strand in enumerate(strands):
        strand = strand.strip().upper()
        if not strand:
            continue
        if use_address:
            if len(strand) <= ADDRESS_NT:
                continue
            address = decode_address(strand[:ADDRESS_NT])
            if max_address is not None and (not 0 <= address <= int(max_address)):
                continue
            body = strand[ADDRESS_NT:]
        else:
            address = order
            body = strand
        payload = decode_inner_rs(body, positions, rs_bytes, PAYLOAD_NT)
        items.append((address, payload))
    if not items:
        raise ValueError('No decodable strands were provided')
    items.sort(key=lambda item: item[0])
    if impute_missing and use_address:
        items = _next_valid_fill(items, max_address=max_address)
    flat: List[int] = []
    for _, payload in items:
        flat.extend(codebook.decode_payload(payload))
    height, width = (int(shape[0]), int(shape[1]))
    channel_size = height * width
    padded_channel_size = ((channel_size + PIXELS_PER_STRAND - 1) // PIXELS_PER_STRAND) * PIXELS_PER_STRAND
    channels = []
    for channel in range(3):
        # Each channel was padded independently during encoding.
        start = channel * padded_channel_size
        chunk = flat[start:start + channel_size]
        if len(chunk) < channel_size:
            chunk = chunk + [0] * (channel_size - len(chunk))
        channels.append(np.asarray(chunk[:channel_size], dtype=np.uint8).reshape(height, width))
    return np.stack(channels, axis=-1)
__all__ = ['PIXELS_PER_STRAND', 'decode_image', 'encode_image']
