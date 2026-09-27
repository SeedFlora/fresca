"""
Cache UrbanSound8K waveforms for the FRESCA-Net trainer (reviewer #2, point 2).

Mirrors benchmark/features_wav.py, but for UrbanSound8K:
  * 8732 clips, 10 classes, 10 author-defined folds (used exactly as published)
  * source clips vary in sample rate (8-192 kHz), channel count and duration
    (<=4 s); we resample to 44.1 kHz mono to match the ESC-50 front-end exactly,
    then zero-pad / truncate to a fixed 4.0 s so the cache is one dense array.

Keeping the sample rate and the 3 s crop identical to ESC-50 means the network
sees the *same* input tensor shape (2x128x259) on both datasets, so the
architecture, its MAC cost and the whole recipe transfer unchanged. That is the
point of the experiment: no per-dataset tuning.

Output (mirrors the ESC-50 cache layout):
  benchmark/cache_us8k/waveforms.npy   float32 [N, 4.0*44100]  (memmap-friendly)
  benchmark/cache_us8k/meta.npz        y (int64), fold (int64), duration (float32)

Usage:
    python benchmark/prepare_us8k.py --root data/UrbanSound8K
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

SR = 44100
CLIP_SEC = 4.0
N_SAMPLES = int(SR * CLIP_SEC)
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(HERE),
                                                   "data", "UrbanSound8K"))
    ap.add_argument("--out", default=os.path.join(HERE, "cache_us8k"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N clips only")
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    meta_csv = os.path.join(args.root, "metadata", "UrbanSound8K.csv")
    audio_dir = os.path.join(args.root, "audio")
    if not os.path.isfile(meta_csv):
        sys.exit(f"metadata not found: {meta_csv}")

    rows = []
    with open(meta_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["slice_file_name"], int(r["fold"]),
                         int(r["classID"]), r["class"]))
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    print(f"[us8k] {n} clips listed in metadata")

    os.makedirs(args.out, exist_ok=True)
    wav_path = os.path.join(args.out, "waveforms.npy")

    # write straight into a memmap: the full array is ~6.2 GB in float32
    W = np.lib.format.open_memmap(wav_path, mode="w+",
                                  dtype=np.float32, shape=(n, N_SAMPLES))
    y = np.zeros(n, dtype=np.int64)
    fold = np.zeros(n, dtype=np.int64)
    dur = np.zeros(n, dtype=np.float32)
    classes = {}

    t0 = time.time()
    n_short, n_fail = 0, 0
    for i, (fname, fd, cid, cname) in enumerate(rows):
        classes[cid] = cname
        path = os.path.join(audio_dir, f"fold{fd}", fname)
        try:
            # librosa handles the mixed sample rates / stereo / codecs in US8K
            wav, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print(f"[us8k] FAILED {fname}: {e}")
            wav = np.zeros(0, dtype=np.float32)
            n_fail += 1
        dur[i] = len(wav) / SR
        if len(wav) < N_SAMPLES:
            n_short += 1
            out = np.zeros(N_SAMPLES, dtype=np.float32)
            out[:len(wav)] = wav
        else:
            out = wav[:N_SAMPLES].astype(np.float32)
        W[i] = out
        y[i] = cid
        fold[i] = fd
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"[us8k] {i+1}/{n}  {el:.0f}s  eta {el/(i+1)*(n-i-1):.0f}s", flush=True)

    W.flush()
    np.savez(os.path.join(args.out, "meta.npz"), y=y, fold=fold, duration=dur)

    # duration statistics matter for the paper: US8K clips are <=4 s and many
    # are much shorter, unlike ESC-50's uniform 5 s.
    stats = {
        "n_clips": int(n),
        "n_classes": int(len(classes)),
        "n_folds": int(len(set(fold.tolist()))),
        "sr": SR,
        "clip_sec": CLIP_SEC,
        "n_failed_decode": int(n_fail),
        "n_shorter_than_4s": int(n_short),
        "frac_shorter_than_4s": float(n_short / n),
        "n_shorter_than_3s_crop": int((dur < 3.0).sum()),
        "frac_shorter_than_3s_crop": float((dur < 3.0).mean()),
        "duration_mean_s": float(dur.mean()),
        "duration_median_s": float(np.median(dur)),
        "per_fold_counts": {str(k): int((fold == k).sum())
                            for k in sorted(set(fold.tolist()))},
        "class_names": {str(k): v for k, v in sorted(classes.items())},
        "cache_bytes": int(os.path.getsize(wav_path)),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\n[us8k] wrote {wav_path} ({stats['cache_bytes']/1024**3:.2f} GB)")
    print(f"[us8k] {n_short} clips ({100*stats['frac_shorter_than_4s']:.1f}%) shorter than 4 s; "
          f"{stats['n_shorter_than_3s_crop']} ({100*stats['frac_shorter_than_3s_crop']:.1f}%) "
          f"shorter than the 3 s crop")
    print(f"[us8k] mean duration {stats['duration_mean_s']:.2f} s, "
          f"median {stats['duration_median_s']:.2f} s")
    print(f"[us8k] per-fold counts: {stats['per_fold_counts']}")
    print(f"[us8k] failed decodes: {n_fail}")


if __name__ == "__main__":
    main()
