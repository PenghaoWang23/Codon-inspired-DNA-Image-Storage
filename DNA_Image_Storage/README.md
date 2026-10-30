# Bionic DNA Image Storage

Reference implementation of a codon-inspired framework for perceptually robust DNA image storage. This repository provides the 640-codeword codebook, image codec, selective Reed-Solomon (RS) protection, cross-copy neural reconstruction, pretrained checkpoints and example inputs.

## Method Overview

- **Codebook:** two or three valid 5-nt codewords are assigned to each of the 256 pixel values. Encoding deterministically selects the first assigned codeword; decoding retains all 640 codewords.
- **Selective protection:** the first two positions of each 5-nt unit are protected. A 120-nt payload stores 24 pixel values and contains 48 protected nucleotides, corresponding to 12 GF(256) data symbols.
- **Reconstruction:** multiple noisy copies are reconstructed by a cross-copy Transformer.
- **Degenerate decoding:** exact codebook matches are decoded directly. An invalid codeword uses the first assigned leaf below its longest matched prefix.

With four parity bytes, the address-bearing codec emits `8-nt address | 120-nt payload | 16-nt RS parity`: 144 nt excluding primers. The neural evaluation example processes the payload and parity region, not addresses or primers.

## Installation

Use Python 3.10 or later. Run all commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

CPU execution is supported; CUDA is selected automatically when available. Use `--device cpu` to force CPU execution. Commands below use a single line for compatibility with both PowerShell and Bash.

Tested environment: Python 3.10.20, NumPy 2.2.6, Pillow 12.2.0, scikit-image 0.25.2, SciPy 1.15.3, reedsolo 1.7.0, PyTorch 2.5.1+cu121 and Matplotlib 3.10.9. These are the tested versions, not a guarantee of compatibility with every version allowed by `requirements.txt`.

## Repository Layout

```text
bionic/
  codec.py                 Codebook, DNA/byte conversion, RS and addresses
  image_codec.py           RGB image encoding and decoding
  reconstruction.py        Neural architecture and checkpoint loader
  training.py              Optional training demonstration
  noise.py                 Simulated substitution, insertion and deletion
  metrics.py               PSNR and SSIM
examples/
  01_encode_decode.py      Codec demonstration without neural reconstruction
  02_end_to_end.py         Multi-copy reconstruction and image evaluation
  03_decode_payload.py     Verification of released clean payloads
assets/
  codebook.csv
  reconstruction_model.pth
  ground_truth.jpg
  payload_sequences.txt
  CHECKSUMS.sha256
tests/
  test_roundtrip.py
requirements.txt
README.md
```

## Quick Start

### 1. Run the Self-Checks

```bash
python tests/test_roundtrip.py
```

Checks cover the codebook, addresses, single-error RS correction, RS=0 strand length, independently padded RGB channels and lossless reference export. Image round trips are tested at four sizes, with and without addresses, and with 0, 2, 4 and 8 parity bytes.

### 2. Verify the Released Payloads

```bash
python examples/03_decode_payload.py
```

This decodes all 13,200 released payloads and verifies pixel identity with the decoded `assets/ground_truth.jpg` image. No neural model or simulated noise is used.

### 3. Encode and Decode an Image

Clean round trip:

```bash
python examples/01_encode_decode.py --output results/codec_clean.png
```

Add 1% substitutions:

```bash
python examples/01_encode_decode.py --noise 0.01 --seed 42 --output results/codec_noisy.png
```

Use another RGB image without RS:

```bash
python examples/01_encode_decode.py --image path/to/image.png --rs-bytes 0 --output results/codec_no_rs.png
```

This example does not use neural reconstruction. Noise is applied to the payload and parity region while preserving addresses. Use `--no-address` to omit the address prefix entirely.

### 4. Run Multi-Copy Neural Reconstruction

Start with a small subset:

```bash
python examples/02_end_to_end.py --strands-per-channel 200 --copies 5 --repeats 1 --condition sub=0.01,ins=0.001,del=0.001 --rs-bytes 4 --save-images --output results/demo
```


Compare parity budgets and error conditions:

```bash
python examples/02_end_to_end.py --strands-per-channel 200 --copies 5 --repeats 3 --condition sub=0.01,ins=0.001,del=0.001 --condition sub=0.05,ins=0.005,del=0.005 --rs-bytes 0 4 8 --output results/parity_comparison
```

### Configuration

The neural evaluation has one input configuration only:

- Image: **300 x 352 pixels** (width x height), using `assets/ground_truth.jpg`.
- Payloads: **13,200 strands**, with 4,400 per RGB channel.
- Checkpoint: `assets/reconstruction_model.pth`, matching the Fig source implementation.
- Defaults: **5 copies**, **4 parity bytes**, **3 independent repeats**.

Without `--condition`, the six substitution/insertion/deletion probability combinations are evaluated in this order:

```text
0.005 / 0.0005 / 0.0005
0.010 / 0.0010 / 0.0010
0.020 / 0.0020 / 0.0020
0.030 / 0.0030 / 0.0030
0.040 / 0.0040 / 0.0040
0.050 / 0.0050 / 0.0050
```

Processing follows the Fig workflow: selective RS encoding, noisy-copy generation, Transformer reconstruction, sequence normalization, strand-level RS decoding, codebook decoding, RGB assembly, JPEG saving/reloading and metric calculation. Metrics use float32 inputs, RGB-channel mean PSNR and standard RGB SSIM, reported to four decimals.


### Main Evaluation Parameters

| Argument | Description |
| --- | --- |
| `--condition` | `sub=...,ins=...,del=...`; repeat for multiple conditions; defaults to the six Fig. 4a/b conditions |
| `--rs-bytes` | One or more parity budgets from 0 to 32 bytes; 0 disables RS |
| `--copies` | Noisy copies per strand; default 5 |
| `--repeats` | Independent error realizations; default 3 |
| `--seed` | Base random seed; default 42 |
| `--strands-per-channel` | Subset size; defaults to 4,400 |
| `--device` | `auto`, `cpu` or `cuda` |
| `--save-images` | Save each reconstructed image |
| `--output` | Output directory; default `results` relative to the working directory |

Use a distinct output directory for each run. Parity sweeps use the same checkpoint without retraining; changing parity length also changes sequence length relative to the checkpoint's nominal 136-nt configuration.

## Outputs and Metrics

| File | Contents |
| --- | --- |
| `raw_results.csv` | Per-repeat metrics, channel settings, copy count and RS successes/failures |
| `summary_results.csv` | Group means and sample SDs across repeats |
| `run_config.json` | Configuration, input/checkpoint hashes and evaluation processing |
| `ground_truth.png` | Exact evaluated reference pixels, saved losslessly |
| `quality_comparison.png` / `.pdf` | PSNR and SSIM summary plots |
| `condition_*_rs_*_repeat_*.jpg` | Reconstructed images, if `--save-images` is enabled |

- **PSNR:** arithmetic mean of RGB-channel PSNR values with data range 255. A zero-MSE channel contributes infinity, so an infinite channel-mean PSNR alone does not establish pixel identity.
- **SSIM:** calculated using `skimage.metrics.structural_similarity` with the channel axis and a window of at most 7 pixels, adapted to small images.
- **Uncertainty:** sample SD uses denominator `n-1`. For one repeat, zero SD is written as a plotting placeholder, not an uncertainty estimate.
- **Reference export:** `ground_truth.png` preserves the exact reference pixels without an extra JPEG compression. Reconstructed images still undergo the Fig JPEG evaluation step. Saved reference and reconstructed images can be used to recompute the reported metrics.
- **RS failures:** an uncorrectable block retains its reconstructed payload for codebook decoding and does not terminate decoding of the remaining strands.



## Optional Training Demonstration

```bash
python -m bionic.training --sequences 200 --epochs 2 --copies 4 --seed 42 --output results/training
```

This is a small training example, not the exact protocol or training provenance of the released checkpoints. 


## Citation

Please cite the associated codon-inspired DNA image storage manuscript when using this method.
