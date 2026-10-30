from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bionic import Codebook

def main() -> None:
    payload_path = ROOT / 'assets' / 'payload_sequences.txt'
    codebook = Codebook.from_csv(ROOT / 'assets' / 'codebook.csv')
    payloads = [line.strip() for line in payload_path.read_text(encoding='ascii').splitlines() if line.strip()]
    if len(payloads) != 13200:
        raise ValueError(f'Expected 13,200 payloads, got {len(payloads)}')
    flat: list[int] = []
    for payload in payloads:
        if len(payload) != 120:
            raise ValueError(f'Expected 120-nt payload, got {len(payload)}')
        flat.extend(codebook.decode_payload(payload))
    height, width = (352, 300)
    channel_size = height * width
    image = np.stack([np.asarray(flat[channel * channel_size:(channel + 1) * channel_size], dtype=np.uint8).reshape(height, width) for channel in range(3)], axis=-1)
    reference = np.asarray(Image.open(ROOT / 'assets' / 'ground_truth.jpg').convert('RGB'), dtype=np.uint8)
    if not np.array_equal(image, reference):
        diff = np.abs(image.astype(int) - reference.astype(int))
        raise SystemExit(f'Decoded image differs from ground truth: max diff {diff.max()}')
    print(f'payloads: {len(payloads)}')
    print(f'image: {width} x {height} RGB')
    print('decoded image is pixel-identical to assets/ground_truth.jpg')
if __name__ == '__main__':
    main()
