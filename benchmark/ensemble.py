"""
Multi-seed ensemble of FRESCA-Net to push the from-scratch 5-fold accuracy above
the from-scratch SOTA (EnvNet-v2 + aug + BC, 84.9%).

For each held-out fold k we train N models (different seeds) on the OTHER 4 folds
and average their SWA + multi-crop-TTA softmax on fold k. Every ensemble member
is trained only on folds != k, so this stays a leakage-free official-5-fold CV.

Resumable: each member's per-sample test probabilities are cached to
cache/ens_probs/<tag>_fold{k}_seed{s}.npz, so an interrupted run resumes without
recomputing finished members. GPU-idle gated exactly like train.py.

Run:  python benchmark/ensemble.py --seeds 0 1 2 --epochs 300 --tta-crops 7
"""
import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS, count_params
import train as T
import gpu_gate

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PROB_DIR = os.path.join(CACHE, "ens_probs")


def build_args(a):
    """Namespace with every field run_fold expects (FRESCA-Net, full aug)."""
    ns = argparse.Namespace(
        model="fresca", epochs=a.epochs, batch=a.batch, lr=1e-3, eta_min=1e-5,
        warmup=5, wd=5e-4, swa_epochs=a.swa_epochs, crop_sec=3.0,
        label_smooth=0.05, specaug_ramp=20, aug="full",
        crop=True, wave_aug=True, bc=True, spec_aug=True,
        no_gate=a.no_gate, amp=True, seed=0, tta_crops=a.tta_crops,
        return_probs=True, log_every=a.log_every,
    )
    return ns


def train_member(W, y, fold_ids, fold, seed, args, device):
    """Train one ensemble member with OOM-retry; return (probs[N,50], yte[N], acc)."""
    for attempt in range(1, 4):
        try:
            res = T.run_fold(W, y, fold_ids, fold, args, device, seed)
            return res["swa_probs"], res["yte"], res["swa_acc"]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"[ens] fold{fold} seed{seed} OOM (attempt {attempt}); freeing + waiting", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if not args.no_gate:
                gpu_gate.wait_for_idle(label=f"[fold{fold} seed{seed} OOM]")
    raise RuntimeError(f"fold{fold} seed{seed} failed after 3 OOM retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--swa-epochs", type=int, default=40)
    ap.add_argument("--tta-crops", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--tag", default="ens")
    ap.add_argument("--limit", type=int, default=0, help="subset clips (smoke test)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    if device == "cpu":
        args.no_gate = True
    ra = build_args(args)
    os.makedirs(PROB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    W = np.load(os.path.join(CACHE, "waveforms.npy"), mmap_mode="r")
    meta = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
    y, fold_ids = meta["y"], meta["fold"]
    if args.limit:
        sel = np.concatenate([np.where(fold_ids == f)[0][:args.limit // 5]
                              for f in [1, 2, 3, 4, 5]])
        W, y, fold_ids = W[sel], y[sel], fold_ids[sel]
    n_params = count_params(MODELS["fresca"](in_ch=2)) / 1e6
    print(f"[ens] FRESCA-Net ensemble  seeds={args.seeds} epochs={args.epochs} "
          f"tta_crops={args.tta_crops} params={n_params:.2f}M device={device}", flush=True)

    if not args.no_gate:
        gpu_gate.wait_for_idle(label="(startup)")

    out_path = os.path.join(RESULTS_DIR, "fresca_ensemble.json")
    per_fold, single_all, t0 = [], [], time.time()
    for fold in args.folds:
        prob_stack, yte_ref, seed_accs = [], None, []
        for seed in args.seeds:
            cf = os.path.join(PROB_DIR, f"{args.tag}_fold{fold}_seed{seed}.npz")
            if os.path.exists(cf):
                d = np.load(cf)
                probs, yte, acc = d["probs"], d["yte"], float(d["acc"])
                print(f"[ens] fold{fold} seed{seed}: cached acc={acc:.2f}", flush=True)
            else:
                print(f"[ens] === training fold{fold} seed{seed} ({gpu_gate.snapshot()}) ===", flush=True)
                probs, yte, acc = train_member(W, y, fold_ids, fold, seed, ra, device)
                np.savez(cf, probs=probs, yte=yte, acc=acc)
                # release this member's model/optimizer cache so it does not
                # accumulate across the 15 members and eventually OOM
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            prob_stack.append(probs)
            seed_accs.append(acc)
            yte_ref = yte
            single_all.append(acc)

        ens_probs = np.mean(prob_stack, axis=0)
        ens_acc = float((ens_probs.argmax(1) == yte_ref).mean() * 100)
        per_fold.append({"test_fold": int(fold), "ensemble_acc": ens_acc,
                         "seed_accs": [float(a) for a in seed_accs]})
        print(f"[ens] fold{fold}: seeds={[f'{a:.2f}' for a in seed_accs]} "
              f"-> ENSEMBLE={ens_acc:.2f}", flush=True)

        # save incrementally
        ea = [p["ensemble_acc"] for p in per_fold]
        summary = {
            "model": "fresca", "aug": "ensemble", "seeds": args.seeds,
            "epochs": args.epochs, "tta_crops": args.tta_crops,
            "folds": args.folds, "completed_folds": [p["test_fold"] for p in per_fold],
            "per_fold": per_fold,
            "mean_swa_acc": float(np.mean(ea)), "std_swa_acc": float(np.std(ea)),
            "mean_single_seed": float(np.mean(single_all)),
            "params_millions": n_params, "minutes": (time.time() - t0) / 60,
            "partial": len(per_fold) != len(args.folds),
        }
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    print("\n[ens] ===== ENSEMBLE SUMMARY =====", flush=True)
    for p in per_fold:
        print(f"  fold{p['test_fold']}: ensemble={p['ensemble_acc']:.2f} "
              f"(seeds {[f'{a:.1f}' for a in p['seed_accs']]})", flush=True)
    ea = [p["ensemble_acc"] for p in per_fold]
    print(f"  MEAN single-seed = {np.mean(single_all):.2f}")
    print(f"  MEAN ENSEMBLE    = {np.mean(ea):.2f} +/- {np.std(ea):.2f}")
    print(f"  wall-clock = {(time.time()-t0)/60:.1f} min")
    print(f"[ens] saved {out_path}")


if __name__ == "__main__":
    main()
