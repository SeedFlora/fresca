# ESC-50 Benchmark — FRESCA-Net

Official **5-fold cross-validation**, trained **from scratch on ESC-50 only** (no external data / no pretraining). Each of the 2000 clips is tested exactly once as part of its held-out fold. Accuracy is the **SWA-averaged model with multi-crop test-time augmentation** (label-free), under the leakage-free protocol described below.

## Result in context (from-scratch methods)

| Method | Accuracy | Class |
|---|---|---|
| BEATs (pretrained, external data) | 98.10% | pretrained (out of scope) |
| CLAP (pretrained, external data) | 96.70% | pretrained (out of scope) |
| EnvNet-v2 + aug + BC learning (tokozume2017b) | 84.90% | from-scratch + aug (SOTA) |
| **FRESCA-Net, 3-seed ensemble (ours)** | **82.45%** | proposed (±1.60) |
| EnvNet-v2 + BC learning (tokozume2017b) | 81.80% | from-scratch + aug |
| **FRESCA-Net, single model (ours)** | **81.55%** | proposed (±2.83) |
| Human accuracy (piczak2015a) | 81.30% | reference |
| Multi-scale CNN (zhu2018) | 79.10% | from-scratch + aug |
| Baseline CNN+BN+BC learning (tokozume2017b) | 76.90% | from-scratch + aug |
| EnvNet-v2 raw waveform (tokozume2017a) | 74.40% | from-scratch CNN |
| Piczak CNN (piczak2015b) | 64.50% | from-scratch CNN |
| 3-layer CNN mel-STFT (huzaifah2017) | 56.40% | from-scratch CNN |
| 5-layer CNN on raw audio (SoundNet, ESC-50) | 51.10% | from-scratch CNN |
| Random Forest (MFCC+ZCR), piczak2015a | 44.30% | from-scratch baseline |
| SVM (MFCC+ZCR), piczak2015a | 39.60% | from-scratch baseline |
| k-NN (MFCC+ZCR), piczak2015a | 32.20% | from-scratch baseline |

## Ablation (same pipeline & protocol)

| Architecture | no augmentation | full aug + SWA + 3-crop TTA |
|---|---|---|
| **FRESCA-Net** (2.88M) | 78.65% ± 3.39 | 81.55% ± 2.83 |
| Piczak CNN (0.71M) | 62.20% ± 4.11 | 70.25% ± 1.39 |

**Contribution decomposition (matched recipe):**

- Architecture, no-aug (FRESCA-Net vs Piczak): **+16.45 pt**
- Augmentation on FRESCA-Net (full vs none): **+2.90 pt**
- Augmentation on Piczak (full vs none): **+8.05 pt**

## Protocol (leakage-free)

- Official 5 folds; for fold *k*, train on the other 4, test on *k*.
- **No** early-stopping / checkpoint / EMA selection on the test fold; fixed 200-epoch budget.
- Headline = SWA-averaged weights (last 30 epochs), BN recomputed on training folds only.
- Inference uses **3-crop TTA** (label-free averaging of 3 evenly-spaced 3 s crops) — standard for ESC-50.
- Per-instance log-mel standardization (no train/test statistic sharing).
- Pretrained transformers (CLAP/BEATs/AST, 95–98%) use external data and are out of scope; shown only for orientation.

