# FRESCA-Net

Reference implementation for **"FRESCA-Net: A Compact From-Scratch Convolutional
Network with Energy-Domain Between-Class Learning for Environmental Sound
Classification"** (CITISMA 2026).

Budi Juarto, Mochamad Haldi Widianto — Bina Nusantara University, Jakarta, Indonesia.

FRESCA-Net is a 2.88 M-parameter residual network trained **entirely from
scratch** — no external data, no pretraining — and evaluated under a
deliberately leakage-free cross-validation protocol.

## Results

| Dataset | Protocol | Accuracy |
|---|---|---|
| ESC-50 | official 5-fold, from scratch | **81.55 % ± 2.83** (single model) |
| ESC-50 | 3-seed ensemble | **82.45 % ± 1.60** |
| UrbanSound8K | official 10-fold, recipe unchanged | **74.57 % ± 6.10** |

On ESC-50 this exceeds crowdsourced human accuracy (81.3 %) and matches
EnvNet-v2 with Between-Class learning (81.8 %) without any external data. On
UrbanSound8K the same recipe transfers without re-tuning and clears the classic
from-scratch baselines, though it does not reproduce its ESC-50 ranking — the
paper reports this directly rather than as a win.

Measured inference cost (batch 1, one 3 s crop, single core of an i7-12700H):
2.23 GMAC, 11.0 MB of fp32 weights, 45.2 MB peak activation, 100.1 ms.

## Start here

- **[benchmark/README.md](benchmark/README.md)** — the reproduction package: the
  exact command and seed behind every number in the paper.
- **[benchmark/BENCHMARK.md](benchmark/BENCHMARK.md)** — full comparison tables
  and the ablation.
- **[paper/](paper/)** — LaTeX source, figures, and the compiled PDF.

```bash
python benchmark/features_wav.py     # cache ESC-50 waveforms
python benchmark/run_all.py          # ESC-50 experiment matrix
bash   benchmark/run_us8k.sh         # UrbanSound8K 10-fold
python benchmark/profile_edge.py     # inference-cost table
python benchmark/verify_claims.py    # re-derive every reported number
```

## Data

Neither dataset is redistributed here; both are public and are used exactly as
their authors distribute them, with the official fold assignments unmodified.

- **ESC-50** — K. J. Piczak, *ESC: Dataset for Environmental Sound
  Classification*, ACM MM 2015. <https://github.com/karolpiczak/ESC-50>
  (CC BY-NC 3.0). This repository is built on it; the upstream README is kept as
  [README-ESC-50-upstream.md](README-ESC-50-upstream.md).
- **UrbanSound8K** — J. Salamon, C. Jacoby, J. P. Bello, *A Dataset and Taxonomy
  for Urban Sound Research*, ACM MM 2014. Fetched by `benchmark/run_us8k.sh`.

## Requirements

Python 3.10+, PyTorch 2.x with CUDA, `librosa`, `soundfile`, `numpy`, plus
`thop` and `onnxruntime` for the cost profiler. All experiments were run on a
single 8 GB RTX 3070 Ti Laptop GPU.
