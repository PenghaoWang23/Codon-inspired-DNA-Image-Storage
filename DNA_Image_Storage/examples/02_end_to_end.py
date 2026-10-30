from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib.pyplot as plt
import numpy as np
import reedsolo
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
BASES = 'ATCG'
BASE_TO_BITS = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
BITS_TO_BASE = {value: key for key, value in BASE_TO_BITS.items()}
BASE_TO_INDEX = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': 4}
INDEX_TO_BASE = {value: key for key, value in BASE_TO_INDEX.items()}
PAYLOAD_NT = 120
CODING_UNIT_NT = 5
PIXELS_PER_STRAND = PAYLOAD_NT // CODING_UNIT_NT
FULL_STRANDS_PER_CHANNEL = 4400
FULL_IMAGE_SHAPE = (352, 300)
MODEL_EMBED_DIM = 64
MODEL_HEADS = 2
MODEL_LAYERS = 2
MODEL_HIDDEN_DIM = 256

@dataclass(frozen=True)
class ErrorCondition:
    substitution: float
    insertion: float
    deletion: float

    @property
    def label(self) -> str:
        return f'sub={self.substitution:g}, ins={self.insertion:g}, del={self.deletion:g}'

class TrieNode:

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.pixel: int | None = None

class CodebookDecoder:

    def __init__(self, rows: Iterable[tuple[str, int]]) -> None:
        self.root = TrieNode()
        self.exact: dict[str, int] = {}
        for codeword, pixel in rows:
            if len(codeword) != CODING_UNIT_NT or set(codeword) - set(BASES):
                raise ValueError(f'Invalid codeword in codebook: {codeword!r}')
            node = self.root
            for base in codeword:
                node = node.children.setdefault(base, TrieNode())
            node.pixel = pixel
            self.exact[codeword] = pixel
        pixels = {pixel for pixel in self.exact.values()}
        if len(self.exact) != 640 or pixels != set(range(256)):
            raise ValueError('The released codebook must contain 640 leaves covering pixels 0-255.')

    @classmethod
    def from_csv(cls, path: Path) -> 'CodebookDecoder':
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            rows = [(row['codeword'], int(row['pixel_value'])) for row in reader]
        return cls(rows)

    @staticmethod
    def _first_leaf(node: TrieNode) -> int | None:
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current.pixel is not None:
                return current.pixel
            queue.extend(current.children.values())
        return None

    def decode_payload(self, payload: str) -> list[int]:
        pixels: list[int] = []
        for start in range(0, len(payload), CODING_UNIT_NT):
            word = payload[start:start + CODING_UNIT_NT]
            if len(word) != CODING_UNIT_NT or word[0] not in BASES:
                pixels.append(pixels[-1] if pixels else 0)
                continue
            exact_pixel = self.exact.get(word)
            if exact_pixel is not None:
                pixels.append(exact_pixel)
                continue
            node = self.root.children.get(word[0])
            if node is not None:
                for base in word[1:]:
                    next_node = node.children.get(base)
                    if next_node is None:
                        break
                    node = next_node
            pixel = self._first_leaf(node) if node is not None else None
            pixels.append(pixel if pixel is not None else pixels[-1] if pixels else 128)
        return pixels

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

def parse_condition(text: str) -> ErrorCondition:
    values: dict[str, float] = {}
    aliases = {'sub': 'substitution', 'substitution': 'substitution', 'ins': 'insertion', 'insertion': 'insertion', 'del': 'deletion', 'deletion': 'deletion'}
    for item in text.split(','):
        if '=' not in item:
            raise argparse.ArgumentTypeError('Conditions must use sub=...,ins=...,del=... syntax.')
        key, raw_value = (part.strip().lower() for part in item.split('=', 1))
        if key not in aliases:
            raise argparse.ArgumentTypeError(f'Unknown error type: {key}')
        try:
            value = float(raw_value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f'Invalid probability: {raw_value}') from error
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError('Error probabilities must be in [0, 1].')
        values[aliases[key]] = value
    return ErrorCondition(substitution=values.get('substitution', 0.0), insertion=values.get('insertion', 0.0), deletion=values.get('deletion', 0.0))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the end-to-end codon-inspired decoding experiment.')
    parser.add_argument('--condition', action='append', type=parse_condition, help='Error condition; repeat this option for multiple conditions. Example: --condition sub=0.02,ins=0.002,del=0.002')
    parser.add_argument('--rs-bytes', nargs='+', type=int, default=[4], help='Numbers of RS parity bytes to compare; 0 disables RS (default: 4).')
    parser.add_argument('--copies', type=int, default=5, help='Noisy copies per strand (Fig. 4a/b default: 5).')
    parser.add_argument('--repeats', type=int, default=3, help='Independent noise repeats.')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed.')
    parser.add_argument('--strands-per-channel', type=int, default=FULL_STRANDS_PER_CHANNEL, help='Default: all 4400 strands per RGB channel. Use fewer for an audit run.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--save-images', action='store_true', help='Save each reconstructed image.')
    args = parser.parse_args()
    if not args.condition:
        args.condition = [ErrorCondition(rate, rate / 10, rate / 10) for rate in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)]
    if args.copies < 1 or args.repeats < 1:
        parser.error('--copies and --repeats must be positive.')
    if not 1 <= args.strands_per_channel <= FULL_STRANDS_PER_CHANNEL:
        parser.error(f'--strands-per-channel must be between 1 and {FULL_STRANDS_PER_CHANNEL}.')
    if any((value < 0 or value > 32 for value in args.rs_bytes)):
        parser.error('Each --rs-bytes value must be between 0 and 32.')
    args.rs_bytes = list(dict.fromkeys(args.rs_bytes))
    return args

def select_device(requested: str) -> torch.device:
    if requested == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is not available.')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)

def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(path: Path, device: torch.device) -> TransformerDNA:
    model = TransformerDNA().to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def protected_positions() -> list[int]:
    positions = []
    for start in range(0, PAYLOAD_NT, CODING_UNIT_NT):
        positions.extend((start, start + 1))
    return positions

def bases_to_bytes(sequence: str) -> bytearray:
    bits = ''.join((BASE_TO_BITS[base] for base in sequence))
    if len(bits) % 8:
        raise ValueError('The protected sequence must contain a whole number of bytes.')
    return bytearray((int(bits[index:index + 8], 2) for index in range(0, len(bits), 8)))

def bytes_to_bases(values: Sequence[int]) -> str:
    bits = ''.join((f'{value:08b}' for value in values))
    return ''.join((BITS_TO_BASE[bits[index:index + 2]] for index in range(0, len(bits), 2)))

def add_rs(payload: str, parity_bytes: int) -> str:
    if parity_bytes == 0:
        return payload
    protected = ''.join((payload[index] for index in protected_positions()))
    encoded = reedsolo.RSCodec(parity_bytes).encode(bases_to_bytes(protected))
    parity = bytes_to_bases(encoded[-parity_bytes:])
    return payload + parity

def decode_rs(reconstructed: str, parity_bytes: int) -> tuple[str, bool]:
    expected_length = PAYLOAD_NT + 4 * parity_bytes
    normalized = reconstructed[:expected_length].ljust(expected_length, 'A')
    payload = normalized[:PAYLOAD_NT]
    if parity_bytes == 0:
        return (payload, True)
    positions = protected_positions()
    protected = ''.join((payload[index] for index in positions))
    parity = normalized[PAYLOAD_NT:expected_length]
    try:
        decoded = reedsolo.RSCodec(parity_bytes).decode(bases_to_bytes(protected + parity))[0]
    except (reedsolo.ReedSolomonError, ValueError, KeyError):
        return (payload, False)
    corrected = bytes_to_bases(decoded)
    corrected_payload = list(payload)
    for index, base in zip(positions, corrected):
        corrected_payload[index] = base
    return (''.join(corrected_payload), True)

def add_noise(sequence: str, condition: ErrorCondition, rng: random.Random) -> str:
    noisy: list[str] = []
    for original in sequence:
        if rng.random() < condition.deletion:
            continue
        current = original
        if rng.random() < condition.substitution:
            current = rng.choice([base for base in BASES if base != original])
        noisy.append(current)
        if rng.random() < condition.insertion:
            noisy.append(rng.choice(BASES))
    return ''.join(noisy)

def stream_seed(base_seed: int, condition_index: int, repeat: int, strand: int, copy: int) -> int:
    material = f'{base_seed}:{condition_index}:{repeat}:{strand}:{copy}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'big')

def reconstruct(model: TransformerDNA, copies: Sequence[str], device: torch.device) -> str:
    max_length = max(map(len, copies))
    indexed = [[BASE_TO_INDEX.get(base, BASE_TO_INDEX['-']) for base in copy] + [BASE_TO_INDEX['-']] * (max_length - len(copy)) for copy in copies]
    inputs = torch.tensor([indexed], dtype=torch.long, device=device)
    with torch.inference_mode():
        prediction = model(inputs).argmax(dim=2).squeeze(0).cpu().tolist()
    return ''.join((INDEX_TO_BASE[index] for index in prediction if index != 4))

def decode_strands(payloads: Sequence[str], decoder: CodebookDecoder, model: TransformerDNA, condition: ErrorCondition, condition_index: int, parity_bytes: int, copies: int, repeat: int, base_seed: int, device: torch.device) -> tuple[np.ndarray, int, int]:
    decoded_pixels: list[int] = []
    rs_successes = 0
    rs_failures = 0
    for strand_index, payload in enumerate(payloads):
        encoded = add_rs(payload, parity_bytes)
        noisy_copies = []
        for copy_index in range(copies):
            rng = random.Random(stream_seed(base_seed, condition_index, repeat, strand_index, copy_index))
            noisy_copies.append(add_noise(encoded, condition, rng))
        reconstructed = reconstruct(model, noisy_copies, device)
        decoded_payload, success = decode_rs(reconstructed, parity_bytes)
        rs_successes += int(success)
        rs_failures += int(not success)
        decoded_pixels.extend(decoder.decode_payload(decoded_payload))
        if (strand_index + 1) % 250 == 0 or strand_index + 1 == len(payloads):
            print(f'    strands {strand_index + 1}/{len(payloads)}', end='\r' if strand_index + 1 < len(payloads) else '\n', flush=True)
    return (np.asarray(decoded_pixels, dtype=np.uint8), rs_successes, rs_failures)

def build_rgb(channel_pixels: Sequence[np.ndarray], full_size: bool, image_shape: tuple[int, int]=FULL_IMAGE_SHAPE) -> np.ndarray:
    pixel_count = min(map(len, channel_pixels))
    if full_size:
        shape = image_shape
        if pixel_count < shape[0] * shape[1]:
            raise ValueError('Not enough decoded pixels for the full input image.')
        pixel_count = shape[0] * shape[1]
    else:
        width = max(1, int(math.sqrt(pixel_count)))
        shape = (math.ceil(pixel_count / width), width)
    padded_channels = []
    size = shape[0] * shape[1]
    for channel in channel_pixels:
        padded = np.zeros(size, dtype=np.uint8)
        padded[:pixel_count] = channel[:pixel_count]
        padded_channels.append(padded.reshape(shape))
    return np.stack(padded_channels, axis=-1)

def calculate_metrics(reference: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    # Match the Fig. 4a/b source workflow after JPEG decoding.
    ref = reference.astype(np.float32)
    test = decoded.astype(np.float32)
    if ref.shape != test.shape:
        raise ValueError('Reference and reconstructed images must have the same shape.')
    window = min(7, min(ref.shape[:2]) if min(ref.shape[:2]) % 2 else min(ref.shape[:2]) - 1)
    if window < 3:
        raise ValueError('At least three pixels per image dimension are required for SSIM.')
    channel_psnr = [float(peak_signal_noise_ratio(ref[..., c], test[..., c], data_range=255)) for c in range(3)]
    return {'mse': round(float(mean_squared_error(ref, test)), 4), 'psnr': round(float(np.mean(channel_psnr)), 4), 'psnr_red': round(channel_psnr[0], 4), 'psnr_green': round(channel_psnr[1], 4), 'psnr_blue': round(channel_psnr[2], 4), 'ssim': round(float(structural_similarity(ref, test, data_range=255, channel_axis=-1, win_size=window)), 4)}

def jpeg_roundtrip(image: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='JPEG')
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert('RGB'))

def save_reference(reference: np.ndarray, output: Path) -> str:
    # Preserve the exact evaluated pixels, including any earlier JPEG decoding.
    name = 'ground_truth.png'
    Image.fromarray(reference).save(output / name)
    return name

def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(raw_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((int(row['condition_id']), int(row['rs_bytes'])), []).append(row)
    summary_rows = []
    for (condition_id, rs_bytes), rows in sorted(grouped.items()):
        first = rows[0]
        summary: dict[str, object] = {'condition_id': condition_id, 'condition': first['condition'], 'substitution': first['substitution'], 'insertion': first['insertion'], 'deletion': first['deletion'], 'rs_bytes': rs_bytes, 'copies': first['copies'], 'repeats': len(rows)}
        for metric in ('psnr', 'ssim', 'rs_failure_fraction'):
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows

def plot_summary(rows: Sequence[dict[str, object]], output: Path) -> None:
    condition_ids = sorted({int(row['condition_id']) for row in rows})
    labels = [next((str(row['condition']) for row in rows if int(row['condition_id']) == value)) for value in condition_ids]
    rs_values = sorted({int(row['rs_bytes']) for row in rows})
    x = np.arange(len(condition_ids))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(rs_values)))
    for rs_bytes, color in zip(rs_values, colors):
        selected = [row for row in rows if int(row['rs_bytes']) == rs_bytes]
        selected.sort(key=lambda row: int(row['condition_id']))
        label = 'No RS' if rs_bytes == 0 else f'RS parity: {rs_bytes} bytes'
        for axis, metric, ylabel in ((axes[0], 'psnr', 'PSNR (dB)'), (axes[1], 'ssim', 'SSIM')):
            means = [float(row[f'{metric}_mean']) for row in selected]
            errors = [float(row[f'{metric}_sd']) for row in selected]
            axis.errorbar(x, means, yerr=errors, marker='o', linewidth=1.6, capsize=3, color=color, label=label)
            axis.set_ylabel(ylabel)
            axis.set_xticks(x, labels, rotation=18, ha='right')
            axis.grid(axis='y', color='#d8d8d8', linewidth=0.7)
            axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, ncol=min(3, len(rs_values)))
    fig.savefig(output / 'quality_comparison.png', dpi=300)
    fig.savefig(output / 'quality_comparison.pdf')
    plt.close(fig)

def load_payloads(path: Path, strands_per_channel: int=FULL_STRANDS_PER_CHANNEL) -> list[str]:
    payloads = [line.strip() for line in path.read_text(encoding='ascii').splitlines() if line.strip()]
    if len(payloads) != 3 * strands_per_channel:
        raise ValueError(f'Expected {3 * strands_per_channel} payload strands.')
    if any((len(sequence) != PAYLOAD_NT or set(sequence) - set(BASES) for sequence in payloads)):
        raise ValueError('Every payload must contain exactly 120 A/T/C/G nucleotides.')
    return payloads

def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    assets = script_dir.parent / 'assets'
    full_count = FULL_STRANDS_PER_CHANNEL
    image_shape = FULL_IMAGE_SHAPE
    reference_path = assets / 'ground_truth.jpg'
    payload_path = assets / 'payload_sequences.txt'
    model_path = assets / 'reconstruction_model.pth'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    seed_torch(args.seed)
    decoder = CodebookDecoder.from_csv(assets / 'codebook.csv')
    all_payloads = load_payloads(payload_path, full_count)
    model = load_model(model_path, device)
    count = args.strands_per_channel
    selected_payloads = []
    for channel in range(3):
        start = channel * full_count
        selected_payloads.extend(all_payloads[start:start + count])
    reference_channels = []
    for channel in range(3):
        start = channel * count
        pixels: list[int] = []
        for payload in selected_payloads[start:start + count]:
            pixels.extend(decoder.decode_payload(payload))
        reference_channels.append(np.asarray(pixels, dtype=np.uint8))
    clean_reference = build_rgb(reference_channels, count == full_count, image_shape)
    if count == full_count:
        reference = np.asarray(Image.open(reference_path).convert('RGB'))
        if reference.shape[:2] != image_shape:
            raise ValueError('Reference image shape must be 352 rows by 300 columns.')
    else:
        reference = jpeg_roundtrip(clean_reference)
    reference_name = save_reference(reference, output)
    configuration = {'conditions': [asdict(condition) for condition in args.condition], 'rs_bytes': args.rs_bytes, 'copies': args.copies, 'repeats': args.repeats, 'seed': args.seed, 'strands_per_channel': count, 'input_profile': 'figure4ab', 'full_strands_per_channel': full_count, 'full_image_shape_height_width': list(image_shape), 'reference_file': reference_name, 'reference_source_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(), 'payload_sha256': hashlib.sha256(payload_path.read_bytes()).hexdigest(), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'metric_image_processing': 'JPEG roundtrip of reconstructed image, float32 metric inputs, four-decimal metrics', 'evaluation_profile': 'figure4ab_jpeg', 'psnr_definition': 'arithmetic mean of RGB channel PSNRs; individual channels also exported', 'device': str(device), 'payload_nt': PAYLOAD_NT, 'protected_positions_per_5nt_unit': [1, 2]}
    (output / 'run_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    print(f'Device: {device}')
    print(f'Fig. 4a/b input: {image_shape[1]} x {image_shape[0]}; JPEG evaluation')
    print(f'Payload strands: {len(selected_payloads)} ({count} per RGB channel)')
    print(f'Conditions: {len(args.condition)}; RS settings: {args.rs_bytes}')
    raw_rows: list[dict[str, object]] = []
    started = time.time()
    for condition_index, condition in enumerate(args.condition, 1):
        for parity_bytes in args.rs_bytes:
            for repeat in range(1, args.repeats + 1):
                print(f'Condition {condition_index}/{len(args.condition)} ({condition.label}); RS={parity_bytes}; repeat={repeat}/{args.repeats}')
                decoded_pixels, rs_successes, rs_failures = decode_strands(selected_payloads, decoder, model, condition, condition_index, parity_bytes, args.copies, repeat, args.seed, device)
                channel_size = count * PIXELS_PER_STRAND
                decoded_channels = [decoded_pixels[channel * channel_size:(channel + 1) * channel_size] for channel in range(3)]
                decoded_image = build_rgb(decoded_channels, count == full_count, image_shape)
                evaluated_image = jpeg_roundtrip(decoded_image)
                metrics = calculate_metrics(reference, evaluated_image)
                total_blocks = rs_successes + rs_failures
                row: dict[str, object] = {'condition_id': condition_index, 'condition': condition.label, 'substitution': condition.substitution, 'insertion': condition.insertion, 'deletion': condition.deletion, 'rs_bytes': parity_bytes, 'copies': args.copies, 'repeat': repeat, 'seed': args.seed, 'strands': len(selected_payloads), 'rs_successes': rs_successes, 'rs_failures': rs_failures, 'rs_failure_fraction': rs_failures / total_blocks if total_blocks else 0.0, **metrics}
                raw_rows.append(row)
                print(f"    PSNR={metrics['psnr']:.4f} dB; SSIM={metrics['ssim']:.6f}")
                if args.save_images:
                    name = f'condition_{condition_index:02d}_rs_{parity_bytes}_repeat_{repeat}.jpg'
                    Image.fromarray(decoded_image).save(output / name)
                write_csv(output / 'raw_results.csv', raw_rows)
    summary_rows = summarize(raw_rows)
    write_csv(output / 'summary_results.csv', summary_rows)
    plot_summary(summary_rows, output)
    elapsed = time.time() - started
    print(f'Completed in {elapsed / 60:.2f} min. Results: {output}')
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by user.', file=sys.stderr)
        raise SystemExit(130)
