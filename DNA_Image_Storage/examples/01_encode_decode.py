from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bionic import Codebook, decode_image, encode_image, psnr, ssim
from bionic.noise import mutate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Encode an RGB image into DNA strands and decode it back.')
    parser.add_argument('--image', type=Path, default=ROOT / 'assets' / 'ground_truth.jpg')
    parser.add_argument('--codebook', type=Path, default=ROOT / 'assets' / 'codebook.csv')
    parser.add_argument('--rs-bytes', type=int, default=4, help='Inner RS parity bytes (default: 4).')
    parser.add_argument('--noise', type=float, default=0.0, help='Substitution probability per base.')
    parser.add_argument('--insertion', type=float, default=0.0, help='Insertion probability per base.')
    parser.add_argument('--deletion', type=float, default=0.0, help='Deletion probability per base.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-address', action='store_true', help='Omit the 8-nt address prefix.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'decoded.png')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    image = np.asarray(Image.open(args.image).convert('RGB'), dtype=np.uint8)
    height, width = image.shape[:2]
    codebook = Codebook.from_csv(args.codebook)
    use_address = not args.no_address
    strands = encode_image(image, codebook, rs_bytes=args.rs_bytes, use_address=use_address)
    strands_per_channel = math.ceil(height * width / 24)
    max_address = 3 * strands_per_channel - 1
    print(f'image: {width} x {height} RGB')
    print(f'strands: {len(strands)} ({strands_per_channel} per channel), length: {len(strands[0])} nt')
    print(f"layout: {('8-nt address + ' if use_address else '')}120-nt payload + {4 * args.rs_bytes}-nt RS")
    if args.noise or args.insertion or args.deletion:
        import random
        rng = random.Random(args.seed)
        strands = [strand[:8] + mutate(strand[8:], args.noise, args.insertion, args.deletion, rng) if use_address else mutate(strand, args.noise, args.insertion, args.deletion, rng) for strand in strands]
        print(f'channel: sub={args.noise:g}, ins={args.insertion:g}, del={args.deletion:g}')
    decoded = decode_image(strands, codebook, shape=(height, width), rs_bytes=args.rs_bytes, use_address=use_address, max_address=max_address if use_address else None)
    p = psnr(image, decoded)
    s = ssim(image, decoded)
    print(f'PSNR: {p:.4f} dB' if math.isfinite(p) else 'PSNR: inf (lossless)')
    print(f'SSIM: {s:.6f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(decoded).save(args.output)
    print(f'decoded image: {args.output}')
if __name__ == '__main__':
    main()
