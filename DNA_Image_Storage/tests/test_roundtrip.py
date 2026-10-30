from __future__ import annotations
import random
import runpy
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bionic import Codebook, decode_address, decode_image, encode_address, encode_image
from bionic.codec import PAYLOAD_NT, build_inner_rs_positions, decode_inner_rs, encode_inner_rs

def check_codebook(codebook: Codebook) -> None:
    assert len(codebook.exact) == 640
    assert set(codebook.pixel_to_codeword) == set(range(256))
    for pixel in (0, 1, 127, 128, 255):
        word = codebook.encode_pixel(pixel)
        assert len(word) == 5
        assert codebook.decode_codeword(word) == pixel
    print('[ok] codebook: 640 codewords, 256 pixels, exact encode/decode')

def check_address() -> None:
    for value in (0, 1, 255, 12345, 65535):
        encoded = encode_address(value)
        assert len(encoded) == 8
        assert decode_address(encoded) == value
    print('[ok] 8-nt address round-trip')

def check_rs() -> None:
    codebook = Codebook.from_csv(ROOT / 'assets' / 'codebook.csv')
    rng = random.Random(7)
    pixels = [rng.randrange(256) for _ in range(24)]
    payload = codebook.encode_pixels(pixels)
    positions = build_inner_rs_positions(PAYLOAD_NT)
    unprotected = encode_inner_rs(payload, positions, check_size=0)
    assert unprotected == payload and len(unprotected) == PAYLOAD_NT
    assert decode_inner_rs(unprotected, positions, check_size=0) == payload
    encoded = encode_inner_rs(payload, positions, check_size=4)
    assert len(encoded) == PAYLOAD_NT + 16
    assert decode_inner_rs(encoded, positions, check_size=4) == payload
    position = positions[0] - 1
    original = encoded[position]
    replacement = next((base for base in 'ATCG' if base != original))
    damaged = encoded[:position] + replacement + encoded[position + 1:]
    assert decode_inner_rs(damaged, positions, check_size=4) == payload
    print('[ok] inner RS: zero parity, clean round-trip and one-error correction')

def check_image_roundtrip() -> None:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    codebook = Codebook.from_csv(ROOT / 'assets' / 'codebook.csv')
    strands = encode_image(image, codebook, rs_bytes=4, use_address=True)
    assert len(strands) == 3 * 24
    assert all((len(strand) == 144 for strand in strands))
    decoded = decode_image(strands, codebook, image.shape[:2], rs_bytes=4, use_address=True, max_address=len(strands) - 1)
    assert np.array_equal(image, decoded), 'clean image round-trip is not lossless'
    print('[ok] image round-trip: 24x24 RGB, 72 strands, lossless')

def check_padded_image_roundtrip() -> None:
    rng = np.random.default_rng(71)
    codebook = Codebook.from_csv(ROOT / 'assets' / 'codebook.csv')
    for shape in ((1, 1, 3), (7, 7, 3), (17, 19, 3), (24, 24, 3)):
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        for use_address in (False, True):
            for rs_bytes in (0, 2, 4, 8):
                strands = encode_image(image, codebook, rs_bytes=rs_bytes, use_address=use_address)
                expected_length = PAYLOAD_NT + 4 * rs_bytes + (8 if use_address else 0)
                assert all(len(strand) == expected_length for strand in strands)
                decoded = decode_image(strands, codebook, shape[:2], rs_bytes=rs_bytes,
                                       use_address=use_address, max_address=len(strands) - 1)
                assert np.array_equal(image, decoded), (shape, use_address, rs_bytes)
    print('[ok] padded RGB channels: 4 sizes, with/without addresses, RS=0/2/4/8')

def check_reference_export() -> None:
    pipeline = runpy.run_path(str(ROOT / 'examples' / '02_end_to_end.py'))
    reference = np.asarray(Image.open(ROOT / 'assets' / 'ground_truth.jpg').convert('RGB'))
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        for pixels in (reference, pipeline['jpeg_roundtrip'](reference[:23, :29])):
            name = pipeline['save_reference'](pixels, output)
            assert name == 'ground_truth.png'
            saved = np.asarray(Image.open(output / name).convert('RGB'))
            assert np.array_equal(saved, pixels), 'Saved reference pixels changed'
    print('[ok] full and subset reference exports preserve exact evaluated pixels')

def check_figure4ab_configuration() -> None:
    pipeline = runpy.run_path(str(ROOT / 'examples' / '02_end_to_end.py'))
    previous_argv = sys.argv
    try:
        sys.argv = ['02_end_to_end.py']
        args = pipeline['parse_args']()
    finally:
        sys.argv = previous_argv
    assert args.copies == 5 and args.rs_bytes == [4] and args.repeats == 3
    assert args.strands_per_channel == 4400
    assert pipeline['FULL_IMAGE_SHAPE'] == (352, 300)
    expected = [(r, r / 10, r / 10) for r in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)]
    assert [(c.substitution, c.insertion, c.deletion) for c in args.condition] == expected
    assert not hasattr(args, 'input_profile')
    print('[ok] single Fig. 4a/b configuration: 300x352, 5 copies, RS=4, 3 repeats, 6 conditions')

def main() -> None:
    codebook = Codebook.from_csv(ROOT / 'assets' / 'codebook.csv')
    check_codebook(codebook)
    check_address()
    check_rs()
    check_image_roundtrip()
    check_padded_image_roundtrip()
    check_reference_export()
    check_figure4ab_configuration()
    print('\nAll checks passed.')
if __name__ == '__main__':
    main()
