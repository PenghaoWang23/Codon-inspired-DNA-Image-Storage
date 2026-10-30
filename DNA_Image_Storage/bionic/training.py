from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from .codec import build_inner_rs_positions, encode_inner_rs
from .noise import mutate
from .reconstruction import BASE_TO_INDEX, TransformerDNA, select_device
PAD = BASE_TO_INDEX['-']
TARGET_LENGTH = 136
ROOT = Path(__file__).resolve().parents[1]

def build_split_indices(n: int, train_ratio: float=0.8, seed: int=42) -> tuple[list[int], list[int]]:
    if n < 2:
        raise ValueError(f'At least two sequences are required, got n={n}')
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f'train_ratio must be in (0, 1), got {train_ratio}')
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = max(1, min(n - 1, int(round(n * float(train_ratio)))))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    assert_disjoint(train_indices, val_indices, n)
    return (train_indices, val_indices)

def assert_disjoint(train_indices: Sequence[int], val_indices: Sequence[int], n: Optional[int]=None) -> None:
    train_set = set((int(i) for i in train_indices))
    val_set = set((int(i) for i in val_indices))
    overlap = train_set & val_set
    if overlap:
        raise AssertionError(f'Train/validation overlap: {len(overlap)} shared indices')
    if n is not None and (len(train_set) + len(val_set) != n or train_set | val_set != set(range(n))):
        raise AssertionError('Split is not a complete partition of range(n)')

def _sha256_indices(indices: Sequence[int]) -> str:
    payload = ','.join((str(int(i)) for i in indices)).encode('ascii')
    return hashlib.sha256(payload).hexdigest()

def write_split_manifest(path: Union[str, Path], train_indices: Sequence[int], val_indices: Sequence[int], seed: int, train_ratio: float, extra: Optional[dict]=None) -> Path:
    train_indices = [int(i) for i in train_indices]
    val_indices = [int(i) for i in val_indices]
    n = len(train_indices) + len(val_indices)
    assert_disjoint(train_indices, val_indices, n)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': int(seed), 'train_ratio': float(train_ratio), 'n_sequences': n, 'train_size': len(train_indices), 'val_size': len(val_indices), 'train_indices': train_indices, 'val_indices': val_indices, 'train_indices_sha256': _sha256_indices(train_indices), 'val_indices_sha256': _sha256_indices(val_indices)}
    if extra:
        manifest['extra'] = extra
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output

def load_targets(path: Union[str, Path], count: int, seed: int) -> list[str]:
    payloads = [line.strip() for line in Path(path).read_text(encoding='ascii').splitlines() if line.strip()]
    if len(payloads) < count:
        raise ValueError(f'Requested {count} payloads but the file has {len(payloads)}')
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(payloads)), count))
    positions = build_inner_rs_positions()
    return [encode_inner_rs(payloads[i], positions, check_size=4) for i in indices]

class CopyDataset(torch.utils.data.Dataset):

    def __init__(self, targets: list[str], indices: list[int], copies: int, seed: int, training: bool, substitution: float=0.05, insertion: float=0.01, deletion: float=0.01):
        self.targets = targets
        self.indices = list(indices)
        self.copies = copies
        self.seed = seed
        self.training = training
        self.substitution = substitution
        self.insertion = insertion
        self.deletion = deletion

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        target = self.targets[index]
        rng = random.Random(self.seed + (0 if self.training else 1000000) + index)
        copy_tensors = []
        for _ in range(self.copies):
            noisy = mutate(target, substitution=self.substitution, insertion=self.insertion, deletion=self.deletion, rng=rng)
            noisy = noisy[:TARGET_LENGTH].ljust(TARGET_LENGTH, '-')
            copy_tensors.append([BASE_TO_INDEX.get(base, PAD) for base in noisy])
        target_tensor = [BASE_TO_INDEX.get(base, PAD) for base in target[:TARGET_LENGTH].ljust(TARGET_LENGTH, '-')]
        return (torch.tensor(copy_tensors, dtype=torch.long), torch.tensor(target_tensor, dtype=torch.long))

def train_reconstruction(payloads: Union[str, Path], sequences: int=100, epochs: int=2, copies: int=4, batch_size: int=8, lr: float=0.0005, seed: int=42, device: str='auto', output: Union[str, Path]=ROOT / 'results' / 'training', substitution: float=0.05, insertion: float=0.01, deletion: float=0.01) -> tuple[TransformerDNA, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = load_targets(payloads, sequences, seed)
    train_indices, val_indices = build_split_indices(len(targets), train_ratio=0.8, seed=seed)
    print(f'sequences: {len(targets)}, train: {len(train_indices)}, val: {len(val_indices)}, overlap: 0')
    train_dataset = CopyDataset(targets, train_indices, copies, seed, training=True, substitution=substitution, insertion=insertion, deletion=deletion)
    val_dataset = CopyDataset(targets, val_indices, copies, seed, training=False, substitution=substitution, insertion=insertion, deletion=deletion)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    device_obj = select_device(device)
    model = TransformerDNA().to(device_obj)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    write_split_manifest(output / 'split_manifest.json', train_indices, val_indices, seed=seed, train_ratio=0.8, extra={'sequences': len(targets), 'copies': copies, 'target_length': TARGET_LENGTH})
    best_val = -1.0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0.0
        train_total = 0
        for batch_copies, batch_target in train_loader:
            batch_copies, batch_target = (batch_copies.to(device_obj), batch_target.to(device_obj))
            optimizer.zero_grad()
            logits = model(batch_copies)
            loss = criterion(logits.transpose(1, 2), batch_target)
            loss.backward()
            optimizer.step()
            mask = batch_target != PAD
            train_correct += (logits.argmax(dim=2)[mask] == batch_target[mask]).sum().item()
            train_total += mask.sum().item()
        model.eval()
        val_correct = 0.0
        val_total = 0
        with torch.inference_mode():
            for batch_copies, batch_target in val_loader:
                batch_copies, batch_target = (batch_copies.to(device_obj), batch_target.to(device_obj))
                logits = model(batch_copies)
                mask = batch_target != PAD
                val_correct += (logits.argmax(dim=2)[mask] == batch_target[mask]).sum().item()
                val_total += mask.sum().item()
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history.append({'epoch': epoch, 'train_accuracy': train_acc, 'val_accuracy': val_acc})
        print(f'epoch {epoch}/{epochs}: train={train_acc:.4f} val={val_acc:.4f}')
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), output / 'reconstruction_demo.pth')
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f'best validation accuracy: {best_val:.4f}')
    print(f"saved: {output / 'reconstruction_demo.pth'}")
    print(f"saved: {output / 'split_manifest.json'}")
    return (model, history)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a small cross-copy Transformer on the released payloads.')
    parser.add_argument('--payloads', type=Path, default=ROOT / 'assets' / 'payload_sequences.txt')
    parser.add_argument('--sequences', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--copies', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--substitution', type=float, default=0.05)
    parser.add_argument('--insertion', type=float, default=0.01)
    parser.add_argument('--deletion', type=float, default=0.01)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'training')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    train_reconstruction(payloads=args.payloads, sequences=args.sequences, epochs=args.epochs, copies=args.copies, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device, output=args.output, substitution=args.substitution, insertion=args.insertion, deletion=args.deletion)
__all__ = ['CopyDataset', 'assert_disjoint', 'build_split_indices', 'load_targets', 'main', 'train_reconstruction', 'write_split_manifest']
if __name__ == '__main__':
    main()
