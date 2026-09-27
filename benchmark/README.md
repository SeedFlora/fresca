# FRESCA-Net — reproduction package

A self-contained pipeline that trains **FRESCA-Net** from scratch (no external
data, no pretraining) and reproduces every number in the paper, under the
official folds of **ESC-50** and **UrbanSound8K**.

Training is **GPU-idle gated**: it only touches the GPU when the GPU is free
(judged by VRAM used by *other* applications, so our own training never counts
as "busy"), and pauses if another app grabs the card.

## Headline results

| Dataset | Protocol | FRESCA-Net |
|---|---|---|
| ESC-50 | official 5-fold, from scratch | **81.55 % ± 2.83** (single), **82.45 % ± 1.60** (3-seed ensemble) |
| UrbanSound8K | official 10-fold, from scratch, recipe unchanged | **74.57 % ± 6.10** (100-epoch budget, 1 seed) |

Full comparison tables and the ablation are in [BENCHMARK.md](BENCHMARK.md).

## The model: FRESCA-Net (2.88 M params for 50 classes)

*Frequency-aware Residual-SE net + Energy-aware Between-Class + multi-Crop +
Attentive-statistics pooling.*

| Stage | Block | Output (C×F×T) at the 3 s crop |
|---|---|---|
| Input | log-mel (128 mels) **+ CoordConv frequency channel** | 2×128×259 |
| Stem | Conv3×3 → BN → ReLU → MaxPool 2×2 | 32×64×129 |
| Stage 1 | 2× SE-BasicBlock, C=32 | 32×64×129 |
| Stage 2 | 2× SE-BasicBlock, C=64, stride 2×2 | 64×32×65 |
| Stage 3 | 2× SE-BasicBlock, C=128, stride **(2,1)** anisotropic | 128×16×65 |
| Stage 4 | 2× SE-BasicBlock, C=256, stride **(2,1)** anisotropic | 256×8×65 |
| Freq collapse | AdaptiveAvgPool over frequency | 256×1×65 |
| Head | **attentive-statistics pooling** (mean+std) → BN → FC | 50 |

Key design choices:

- **Energy-domain Between-Class mixing on the raw waveform**, before the STFT —
  mixing log-mels would break BC's energy-additive semantics.
- **Random 3 s crops (train) + 3-crop TTA (test)** — the strongest regulariser
  in the 1600-clips-per-fold regime.
- **Anisotropic downsampling** (frequency ÷16, time ÷4) so a real temporal
  sequence survives for the attentive-statistics head.
- **CoordConv frequency channel + squeeze-excitation** — frequency is not
  translation-invariant on a mel axis.
- **Leakage-free protocol**: fixed epoch budget + SWA (BN recomputed on the
  training folds only); no early stopping or checkpoint selection on the
  held-out fold; per-instance log-mel standardisation.

## Exact commands behind every number in the paper

All runs use **seed 42** unless stated. Each writes a JSON summary with per-fold
accuracies into `results/`.

```bash
# one-time: cache ESC-50 waveforms  -> cache/waveforms.npy
python benchmark/features_wav.py

# Table I + Table III: the four matched ESC-50 cells (200 epochs, SWA last 30)
python benchmark/train.py --model fresca --aug full --epochs 200   # 81.55
python benchmark/train.py --model fresca --aug none --epochs 200   # 78.65
python benchmark/train.py --model piczak --aug full --epochs 200   # 70.25
python benchmark/train.py --model piczak --aug none --epochs 200   # 62.20

# Table I + Fig. 3: 3-seed ensemble (seeds 0,1,2; 300 epochs; 7-crop TTA)
python benchmark/ensemble.py                                       # 82.45

# Table II: edge cost (MACs, weights, peak activation, latency)
python benchmark/profile_edge.py --runs 60                         # -> results/edge_profile.json

# Section III-C: UrbanSound8K, official 10-fold, recipe unchanged
#   (downloads to data/, extracts, caches, then trains)
bash benchmark/run_us8k.sh

# re-derive every claim in the paper from the result files
python benchmark/verify_claims.py
```

`run_us8k.sh` reduces the budget to 100 epochs with a 15-epoch SWA window —
the *same 15 % tail* of the cosine schedule as 30/200 on ESC-50, so the
averaged weights span the same learning-rate range. Every other
hyper-parameter is identical to the ESC-50 runs.

## Files

| File | Purpose |
|---|---|
| `features_wav.py` | Cache ESC-50 raw waveforms → `cache/waveforms.npy` (memmap) |
| `prepare_us8k.py` | Cache UrbanSound8K → `cache_us8k/` (+ duration statistics) |
| `frontend.py` | On-GPU log-mel front-end + waveform augmentation (BC mixing, crop, shift, gain, noise, SpecAugment) |
| `models.py` | `FRESCANet` (proposed) + `PiczakCNN` (baseline) |
| `train.py` | Fold-wise CV trainer (SWA, multi-crop TTA, GPU-idle gated), `--dataset esc50\|us8k` |
| `ensemble.py` | Multi-seed ensemble over the official folds |
| `profile_edge.py` | MACs / weights / peak activation / latency profiler (Table II) |
| `verify_claims.py` | Re-derives every reported number from `results/*.json` |
| `gpu_gate.py` | Wait-for-GPU-idle gate (external-VRAM based) |
| `report.py` | Builds `BENCHMARK.md` |
| `run_all.py` | Runs the ESC-50 experiment matrix |
| `run_us8k.sh` | End-to-end UrbanSound8K reproduction |

## Datasets

Both are public and used exactly as distributed, with the authors' fold
assignments unmodified:

- **ESC-50** — K. J. Piczak, *ESC: Dataset for Environmental Sound
  Classification*, ACM MM 2015. Included in this repository.
- **UrbanSound8K** — J. Salamon, C. Jacoby, J. P. Bello, *A Dataset and Taxonomy
  for Urban Sound Research*, ACM MM 2014. Downloaded by `run_us8k.sh`
  (~6 GB) from the authors' Zenodo record; not redistributed here.

No other data is used at any point — the from-scratch, no-external-data
constraint is the premise of the paper.

## Requirements

Python 3.10+, PyTorch 2.x with CUDA, `librosa`, `soundfile`, `numpy`, `thop`
(for the MAC cross-check) and `onnxruntime` (for the ONNX/INT8 rows of the edge
profile). Training was run on a single 8 GB RTX 3070 Ti Laptop GPU.

### GPU-idle gate (env vars)

| Var | Default | Meaning |
|---|---|---|
| `ESC_GPU_EXT_MAX` | 4000 | pause if VRAM used by *other* apps exceeds this (MiB) |
| `ESC_GPU_MIN_FREE` | 3000 | free VRAM (MiB) required to resume |
| `ESC_GPU_SUSTAIN` | 15 | seconds the GPU must stay free before resuming |
| `ESC_GPU_DISABLE` | 0 | set `1` to skip the gate and train immediately |
