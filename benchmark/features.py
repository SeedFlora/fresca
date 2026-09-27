"""
Precompute and cache log-mel spectrograms for the full ESC-50 dataset.

Runs on CPU (librosa) across all cores. Produces a single cache file so that
GPU training epochs are fast (features live in RAM, no per-epoch audio decode).

Output: benchmark/cache/logmel_128.npz
    X      float32 [2000, N_MELS, N_FRAMES]   log-mel spectrograms (dB)
    y      int64   [2000]                      class target [0..49]
    fold   int64   [2000]                      official CV fold [1..5]
    files  <U..>   [2000]                      filenames (for debugging)
"""
import os
import sys
import numpy as np
import pandas as pd
import librosa
from concurrent.futures import ProcessPoolExecutor
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "audio")
META_CSV = os.path.join(ROOT, "meta", "esc50.csv")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# --- Feature config (fixed, so the cache is reproducible) ---
SR = 44100
N_FFT = 1024
HOP = 512
N_MELS = 128
FMIN = 0
FMAX = SR // 2
# 5 s @ 44100 -> 220500 samples -> 1 + floor(220500/512) = 431 frames
N_FRAMES = 431


def extract_one(filename):
    path = os.path.join(AUDIO_DIR, filename)
    y, _ = librosa.load(path, sr=SR, mono=True)
    # pad/trim to exact length so every clip has identical frame count
    target_len = SR * 5
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # guard frame count
    if logmel.shape[1] < N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, N_FRAMES - logmel.shape[1])))
    else:
        logmel = logmel[:, :N_FRAMES]
    return logmel


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, "logmel_128.npz")
    if os.path.exists(out_path) and "--force" not in sys.argv:
        print(f"[features] cache exists: {out_path} (use --force to rebuild)")
        return

    df = pd.read_csv(META_CSV).sort_values("filename").reset_index(drop=True)
    files = df["filename"].tolist()
    y = df["target"].to_numpy(dtype=np.int64)
    fold = df["fold"].to_numpy(dtype=np.int64)

    print(f"[features] extracting log-mel for {len(files)} clips "
          f"({N_MELS} mels x {N_FRAMES} frames) ...")
    X = np.zeros((len(files), N_MELS, N_FRAMES), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for i, mel in enumerate(ex.map(extract_one, files, chunksize=8)):
            X[i] = mel
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)}")

    np.savez_compressed(out_path, X=X, y=y, fold=fold, files=np.array(files))
    mb = os.path.getsize(out_path) / 1e6
    print(f"[features] saved {out_path}  ({mb:.1f} MB)")
    print(f"[features] X={X.shape} dtype={X.dtype} "
          f"range=[{X.min():.1f},{X.max():.1f}] mean={X.mean():.2f}")


if __name__ == "__main__":
    main()
