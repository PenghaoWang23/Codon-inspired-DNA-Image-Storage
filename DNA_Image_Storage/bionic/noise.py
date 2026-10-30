from __future__ import annotations
import random
from typing import Optional, Sequence
BASES = ('A', 'T', 'C', 'G')

def mutate(sequence: str, substitution: float=0.0, insertion: float=0.0, deletion: float=0.0, rng: Optional[random.Random]=None) -> str:
    rng = rng or random
    noisy: list[str] = []
    for original in sequence:
        if rng.random() < deletion:
            continue
        current = original
        if rng.random() < substitution:
            current = rng.choice([base for base in BASES if base != original])
        noisy.append(current)
        if rng.random() < insertion:
            noisy.append(rng.choice(BASES))
    return ''.join(noisy)

def mutate_many(sequences: Sequence[str], substitution: float=0.0, insertion: float=0.0, deletion: float=0.0, seed: int=42) -> list[str]:
    rng = random.Random(seed)
    return [mutate(sequence, substitution, insertion, deletion, rng) for sequence in sequences]
__all__ = ['BASES', 'mutate', 'mutate_many']
