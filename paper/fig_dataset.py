"""
Dataset visualization for the FRESCA-Net paper: raw waveforms and the log-mel
spectrograms that the network actually sees, for four ESC-50 classes plus an
energy-domain Between-Class mixture of two of them (computed exactly as in the
training front-end).
"""
import os
import csv
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.6,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "audio")
META = os.path.join(ROOT, "meta", "esc50.csv")
SR = 44100
N_FFT, HOP, N_MELS = 1024, 512, 128

# choose one representative fold-1 clip per class
WANT = ["dog", "rain", "crying_baby", "clock_tick"]
pick = {}
with open(META) as fh:
    for r in csv.DictReader(fh):
        if r["fold"] == "1" and r["category"] in WANT and r["category"] not in pick:
            pick[r["category"]] = r["filename"]

MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20, fmax=22050)


def load(fn):
    w, _ = librosa.load(os.path.join(AUDIO, fn), sr=SR, mono=True)
    return w


def logmel(w):
    S = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP, window="hann")) ** 2
    mel = MEL_FB @ S
    return np.log(mel + 1e-6)


def bc_mix(w1, w2, p=0.6):
    r1 = np.sqrt(np.mean(w1 ** 2)) + 1e-10
    r2 = np.sqrt(np.mean(w2 ** 2)) + 1e-10
    n1, n2 = w1 / r1, w2 / r2
    L = min(len(n1), len(n2))
    mix = (p * n1[:L] + (1 - p) * n2[:L]) / np.sqrt(p * p + (1 - p) * (1 - p))
    return mix


sigs = {c: load(pick[c]) for c in WANT}
mix = bc_mix(sigs["dog"], sigs["rain"], p=0.6)

cols = [
    ("Dog bark", sigs["dog"], "#4C78A8"),
    ("Rain", sigs["rain"], "#54A24B"),
    ("BC mix: dog $\\oplus$ rain", mix, "#F58518"),
]

# Single-column figure: the two sources and their energy-domain BC mixture.
# The wider four-class variant is kept as fig_dataset_wide.pdf.
fig, axes = plt.subplots(2, len(cols), figsize=(3.45, 2.15), dpi=300)
for j, (name, w, c) in enumerate(cols):
    t = np.arange(len(w)) / SR
    ax = axes[0, j]
    ax.plot(t, w, color=c, lw=0.4)
    ax.set_title(name, fontsize=6.8, pad=2)
    ax.set_xlim(0, len(w) / SR)
    ax.set_ylim(-1.05 * np.max(np.abs(w)) - 1e-3, 1.05 * np.max(np.abs(w)) + 1e-3)
    ax.set_yticks([])
    ax.tick_params(labelsize=5, width=0.4, length=1.5)
    if j == 0:
        ax.set_ylabel("amp.", fontsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    lm = logmel(w)
    ax2 = axes[1, j]
    ax2.imshow(lm, origin="lower", aspect="auto", cmap="magma",
               extent=[0, len(w) / SR, 0, N_MELS])
    ax2.set_xlabel("time (s)", fontsize=6)
    ax2.tick_params(labelsize=5, width=0.4, length=1.5)
    if j == 0:
        ax2.set_ylabel("mel band", fontsize=6)
    else:
        ax2.set_yticks([])

fig.tight_layout(pad=0.2, h_pad=0.35, w_pad=0.25)
fig.savefig("fig_dataset.pdf", bbox_inches="tight")
print("wrote fig_dataset.pdf using clips:", pick)
