from __future__ import annotations
import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union
import reedsolo
DNA_TO_BITS = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
BITS_TO_DNA = {value: key for key, value in DNA_TO_BITS.items()}
BASES = ('A', 'T', 'G', 'C')
PAYLOAD_NT = 120
CODING_UNIT_NT = 5
ADDRESS_NT = 8

@dataclass
class _TrieNode:
    children: Dict[str, '_TrieNode'] = field(default_factory=dict)
    pixel: Optional[int] = None

class Codebook:

    def __init__(self, rows: Iterable[tuple[str, int]]) -> None:
        self.root = _TrieNode()
        self.exact: Dict[str, int] = {}
        self.pixel_to_codeword: Dict[int, str] = {}
        for codeword, pixel in rows:
            codeword = str(codeword).strip().upper()
            pixel = int(pixel)
            if len(codeword) != CODING_UNIT_NT or set(codeword) - set(BASES):
                raise ValueError(f'Invalid 5-nt codeword: {codeword!r}')
            if not 0 <= pixel <= 255:
                raise ValueError(f'Pixel value out of range: {pixel}')
            if codeword in self.exact:
                raise ValueError(f'Duplicate codeword: {codeword}')
            node = self.root
            for base in codeword:
                node = node.children.setdefault(base, _TrieNode())
            node.pixel = pixel
            self.exact[codeword] = pixel
            self.pixel_to_codeword.setdefault(pixel, codeword)
        if len(self.exact) != 640:
            raise ValueError(f'Expected 640 codewords, got {len(self.exact)}')
        if set(self.pixel_to_codeword) != set(range(256)):
            raise ValueError('The codebook must cover every pixel value 0-255')

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> 'Codebook':
        with Path(path).open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            rows = [(row['codeword'], int(row['pixel_value'])) for row in reader]
        return cls(rows)

    @staticmethod
    def _first_leaf(node: _TrieNode) -> Optional[int]:
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current.pixel is not None:
                return current.pixel
            queue.extend(current.children.values())
        return None

    def encode_pixel(self, pixel: int) -> str:
        return self.pixel_to_codeword[int(pixel)]

    def encode_pixels(self, pixels: Iterable[int]) -> str:
        return ''.join((self.pixel_to_codeword[int(p)] for p in pixels))

    def decode_codeword(self, word: str) -> Optional[int]:
        word = str(word).upper()
        if len(word) != CODING_UNIT_NT or word[0] not in BASES:
            return None
        exact = self.exact.get(word)
        if exact is not None:
            return exact
        node = self.root.children.get(word[0])
        if node is None:
            return None
        for base in word[1:]:
            next_node = node.children.get(base)
            if next_node is None:
                break
            node = next_node
        return self._first_leaf(node)

    def decode_payload(self, payload: str, fill: Optional[int]=None) -> List[int]:
        pixels: List[int] = []
        for start in range(0, len(payload), CODING_UNIT_NT):
            word = payload[start:start + CODING_UNIT_NT]
            value = self.decode_codeword(word)
            if value is None:
                value = fill if fill is not None else pixels[-1] if pixels else 0
            pixels.append(int(value))
        return pixels

def dna_to_binary(dna_sequence: str) -> str:
    return ''.join((DNA_TO_BITS[base] for base in dna_sequence))

def binary_to_dna(bit_string: str) -> str:
    if len(bit_string) % 2:
        raise ValueError('Bit string length must be even')
    return ''.join((BITS_TO_DNA[bit_string[i:i + 2]] for i in range(0, len(bit_string), 2)))

def binary_to_byte_list(bit_string: str) -> List[int]:
    if len(bit_string) % 8:
        raise ValueError('Bit string length must be divisible by 8')
    return [int(bit_string[i:i + 8], 2) for i in range(0, len(bit_string), 8)]

def byte_list_to_binary(byte_values: Sequence[int]) -> str:
    return ''.join((format(int(value) & 255, '08b') for value in byte_values))

def build_inner_rs_positions(payload_length: int=PAYLOAD_NT, stride: int=10) -> List[int]:
    positions: List[int] = []
    for start in range(0, payload_length, stride):
        positions.extend([start + 1, start + 2, start + 6, start + 7])
    return [position for position in positions if position <= payload_length]

def encode_inner_rs(dna_sequence: str, positions: Sequence[int], check_size: int) -> str:
    if len(dna_sequence) != PAYLOAD_NT:
        raise ValueError(f'Expected a {PAYLOAD_NT}-nt payload, got {len(dna_sequence)}')
    if check_size == 0:
        return dna_sequence
    extracted = ''.join((dna_sequence[position - 1] for position in positions))
    encoded = list(reedsolo.RSCodec(check_size).encode(binary_to_byte_list(dna_to_binary(extracted))))
    parity_bits = byte_list_to_binary(encoded)[-check_size * 8:]
    return dna_sequence + binary_to_dna(parity_bits)

def decode_inner_rs(mutated_dna: str, positions: Sequence[int], check_size: int, payload_length: int=PAYLOAD_NT) -> str:
    expected = payload_length + 4 * check_size
    normalized = mutated_dna[:expected].ljust(expected, 'A')
    payload = normalized[:payload_length]
    if check_size <= 0:
        return payload
    extracted = ''.join((normalized[position - 1] for position in positions))
    parity = normalized[payload_length:expected]
    try:
        decoded = list(reedsolo.RSCodec(check_size).decode(binary_to_byte_list(dna_to_binary(extracted + parity))))
    except reedsolo.ReedSolomonError:
        return payload
    corrected = decoded[0] if decoded and (not isinstance(decoded[0], int)) else decoded
    corrected_dna = binary_to_dna(byte_list_to_binary(corrected))
    corrected_payload = list(payload)
    for index, position in enumerate(positions):
        corrected_payload[position - 1] = corrected_dna[index]
    return ''.join(corrected_payload)

def encode_address(index: int, length: int=ADDRESS_NT) -> str:
    if not 0 <= index < 4 ** length:
        raise ValueError(f'Address {index} does not fit in {length} nt')
    bits = format(index, f'0{2 * length}b')
    return binary_to_dna(bits)

def decode_address(address: str) -> int:
    return int(dna_to_binary(address.upper()), 2)

def split_payload(dna_sequence: str, payload_length: int=PAYLOAD_NT) -> List[str]:
    segments = [dna_sequence[i:i + payload_length] for i in range(0, len(dna_sequence), payload_length)]
    if not segments or any((len(segment) != payload_length for segment in segments)):
        raise ValueError('Every payload segment must be exactly 120 nt')
    return segments
__all__ = ['ADDRESS_NT', 'BASES', 'CODING_UNIT_NT', 'Codebook', 'DNA_TO_BITS', 'PAYLOAD_NT', 'build_inner_rs_positions', 'binary_to_dna', 'binary_to_byte_list', 'byte_list_to_binary', 'decode_address', 'decode_inner_rs', 'dna_to_binary', 'encode_address', 'encode_inner_rs', 'split_payload']
