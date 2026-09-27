"""
Sequential benchmark driver. Runs the full experiment matrix, each run gated on
GPU idle (train.py waits for the GPU to be free before/while training).

Matrix:
  1. bcressa   + full augmentation   -> the proposed model (headline)
  2. bcressa   + no augmentation      -> ablation: architecture alone
  3. piczak    + no augmentation      -> faithful baseline (~published 64.5%)
  4. piczak    + full augmentation    -> baseline under the same aug recipe

Run:  python benchmark/run_all.py
Env:  ESC_EPOCHS (default 100), ESC_GPU_* to tune the idle gate.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
EPOCHS = os.environ.get("ESC_EPOCHS", "100")

# Ordered by priority: the two headline numbers (proposed + baseline) run first,
# so partial results are still meaningful if training is interrupted.
RUNS = [
    ["--model", "fresca", "--aug", "full", "--epochs", EPOCHS],   # 1. proposed (headline)
    ["--model", "piczak", "--aug", "none", "--epochs", EPOCHS],   # 2. faithful baseline (~64.5%)
    ["--model", "fresca", "--aug", "none", "--epochs", EPOCHS],   # 3. ablation: architecture w/o aug
    ["--model", "piczak", "--aug", "full", "--epochs", EPOCHS],   # 4. baseline under same aug recipe
]

for i, extra in enumerate(RUNS, 1):
    cmd = [PY, os.path.join(HERE, "train.py")] + extra
    print(f"\n########## RUN {i}/{len(RUNS)}: {' '.join(extra)} ##########", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[run_all] run {i} failed (rc={rc}); continuing")
print("\n[run_all] all runs done.")
