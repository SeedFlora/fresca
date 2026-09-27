"""
Cache raw waveforms for the FRESCA-Net pipeline.

FRESCA-Net does energy-domain Between-Class mixing, random cropping, gain/noise
and the log-mel transform ON THE GPU, so we cache the raw 5 s waveforms (not
log-mels). Loaded via memmap at train time; only the current batch is moved to
the GPU.

Output:
    cache/waveforms.npy   float32 [2000, 220500]   (mmap-friendly, ~1.76 GB)
    cache/meta.npz        y[2000] int64, fold[2000] int64, files[2000]
"""
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "audio")
META_CSV = os.path.join(ROOT, "meta", "esc50.csv")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

SR = 44100
CLIP_LEN = SR * 5  # 220500


def load_one(filename):
    y, sr = sf.read(os.path.join(AUDIO_DIR, filename), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:  # ESC-50 is already 44.1k; guard anyway
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - len(y)))
    else:
        y = y[:CLIP_LEN]
    return y.astype(np.float32)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    wav_path = os.path.join(CACHE_DIR, "waveforms.npy")
    meta_path = os.path.join(CACHE_DIR, "meta.npz")
    if os.path.exists(wav_path) and "--force" not in sys.argv:
        print(f"[wav] cache exists: {wav_path} (use --force to rebuild)")
        return

    df = pd.read_csv(META_CSV).sort_values("filename").reset_index(drop=True)
    files = df["filename"].tolist()
    y = df["target"].to_numpy(dtype=np.int64)
    fold = df["fold"].to_numpy(dtype=np.int64)

    print(f"[wav] loading {len(files)} waveforms @ {SR} Hz ...")
    W = np.lib.format.open_memmap(wav_path, mode="w+", dtype=np.float32,
                                  shape=(len(files), CLIP_LEN))
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for i, w in enumerate(ex.map(load_one, files, chunksize=8)):
            W[i] = w
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)}")
    W.flush()
    np.savez(meta_path, y=y, fold=fold, files=np.array(files))
    gb = os.path.getsize(wav_path) / 1e9
    print(f"[wav] saved {wav_path} ({gb:.2f} GB) and {meta_path}")
    print(f"[wav] W shape={W.shape} dtype={W.dtype} "
          f"range=[{W.min():.3f},{W.max():.3f}]")


if __name__ == "__main__":
    main()
