#!/usr/bin/env bash
# Extract UrbanSound8K, cache its waveforms, and run the official 10-fold CV
# with the ESC-50 recipe unchanged apart from the declared budget reduction.
#
#   epochs 100  (vs 200 on ESC-50) -- reduced compute budget, fixed in advance
#   swa    15   (vs 30)            -- the SAME 15% tail of the cosine schedule,
#                                     so the averaged window spans the same LR
#                                     range and the protocol stays comparable
# Everything else (lr, wd, warmup, batch, crop, BC mixing, SpecAugment,
# 3-crop TTA, seed) is identical to the ESC-50 runs.
set -euo pipefail

ROOT="G:/ESC-50-master/ESC-50-master"
cd "$ROOT"

TARBALL="data/UrbanSound8K.tar.gz"
EXPECTED=6023741708

echo "[us8k] verifying download..."
SIZE=$(stat -c %s "$TARBALL")
if [ "$SIZE" -ne "$EXPECTED" ]; then
  echo "[us8k] ABORT: $TARBALL is $SIZE bytes, expected $EXPECTED"
  exit 1
fi
echo "[us8k] size OK ($SIZE bytes)"

if [ ! -f "data/UrbanSound8K/metadata/UrbanSound8K.csv" ]; then
  echo "[us8k] extracting (this takes several minutes)..."
  tar -xzf "$TARBALL" -C data/
fi
echo "[us8k] extracted: $(ls data/UrbanSound8K/audio | wc -l) fold dirs"

if [ ! -f "benchmark/cache_us8k/stats.json" ]; then
  echo "[us8k] caching waveforms..."
  python benchmark/prepare_us8k.py --root data/UrbanSound8K
fi

echo "[us8k] starting 10-fold training..."
python benchmark/train.py \
  --dataset us8k \
  --model fresca \
  --aug full \
  --epochs 100 \
  --swa-epochs 15 \
  --warmup 5 \
  --seed 42 \
  --tta-crops 3 \
  --log-every 25

echo "[us8k] DONE"
