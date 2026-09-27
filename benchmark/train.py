"""
FRESCA-Net 5-fold CV trainer (waveform pipeline).

Pipeline per batch (all on GPU, front-end in fp32, backbone under AMP):
  raw waveform -> random 3s crop -> time-shift/gain/noise -> energy-domain BC mix
              -> log-mel front-end (+CoordConv) -> SpecAugment -> SE-ResNet -> loss

Protocol (leakage-free, per the design synthesis):
  * official 5-fold CV; for fold k, train on the other 4 folds, test on k.
  * FIXED epoch budget; no early-stopping / checkpoint selection on the test fold.
  * report the SWA-averaged final model (BN recomputed) -> headline number.
  * last-epoch and best-epoch accuracies are logged for reference only.

GPU-idle gated: waits for the GPU to be idle before each fold and re-checks
between epochs (honors the "train only when the GPU is free" request).

Examples:
  python benchmark/train.py --model fresca --epochs 200
  python benchmark/train.py --model piczak --aug none --epochs 200      # baseline
  python benchmark/train.py --model fresca --folds 1 --epochs 120       # single fold
  python benchmark/train.py --model fresca --cpu --limit 200 --epochs 3 # CPU smoke
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS, count_params
from frontend import (LogMelFrontend, random_crop, fixed_crops,
                      wave_shift, wave_gain, wave_noise, bc_mix, spec_augment, SR)
import gpu_gate

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_CLASSES = 50

# Dataset registry. The recipe, the 3 s crop and the front-end are identical
# across datasets on purpose -- only the fold count and the label space differ,
# so a second-dataset result measures transfer of the architecture and not of a
# re-tuned pipeline.
DATASETS = {
    "esc50": {"cache": "cache",      "n_classes": 50, "folds": list(range(1, 6))},
    "us8k":  {"cache": "cache_us8k", "n_classes": 10, "folds": list(range(1, 11))},
}


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()


def save_summary(results, args, n_params, t0, out_path, partial):
    swa = [r["swa_acc"] for r in results] or [0.0]
    fin = [r["final_acc"] for r in results] or [0.0]
    # never serialize the per-sample prob arrays (ensemble uses them in-memory)
    per_fold = [{k: v for k, v in r.items() if k not in ("swa_probs", "yte")}
                for r in results]
    summary = {
        "model": args.model, "dataset": getattr(args, "dataset", "esc50"),
        "aug": args.aug, "tag": args.tag,
        "n_classes": N_CLASSES, "swa_epochs": args.swa_epochs,
        "warmup": args.warmup, "crop_sec": args.crop_sec, "seed": args.seed,
        "epochs": args.epochs, "folds": args.folds,
        "completed_folds": [r["test_fold"] for r in results],
        "partial": partial, "per_fold": per_fold,
        "mean_swa_acc": float(np.mean(swa)), "std_swa_acc": float(np.std(swa)),
        "mean_final_acc": float(np.mean(fin)), "std_final_acc": float(np.std(fin)),
        "params_millions": n_params, "minutes": (time.time() - t0) / 60,
    }
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)   # BN/bias -> no weight decay
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999))


def lr_lambda(epoch, warmup, total, min_mult):
    if epoch < warmup:
        return (epoch + 1) / warmup
    prog = (epoch - warmup) / max(1, total - warmup)
    return min_mult + (1 - min_mult) * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(net, fe, W, y_idx, device, crop_len, tta, bs=100, n_crops=3, return_probs=False):
    net.eval()
    correct, n = 0, len(y_idx)
    yv = torch.from_numpy(y_idx)
    all_probs = []
    for i in range(0, n, bs):
        wav = torch.from_numpy(np.ascontiguousarray(W[i:i + bs])).to(device).float()
        if tta:
            probs = 0
            crops = fixed_crops(wav, crop_len, n_crops)
            for c in crops:
                probs = probs + torch.softmax(net(fe(c)).float(), dim=1)
            probs = probs / len(crops)
        else:
            probs = torch.softmax(net(fe(wav)).float(), dim=1)
        correct += (probs.argmax(1).cpu() == yv[i:i + bs]).sum().item()
        if return_probs:
            all_probs.append(probs.cpu())
    acc = correct / n * 100
    if return_probs:
        return acc, torch.cat(all_probs).numpy()
    return acc


@torch.no_grad()
def update_bn_swa(swa, fe, W, device, crop_len, g, bs=64):
    for m in swa.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    swa.train()
    n = len(W)
    perm = torch.randperm(n).numpy()
    for i in range(0, n, bs):
        idx = np.sort(perm[i:i + bs])
        wav = torch.from_numpy(np.ascontiguousarray(W[idx])).to(device).float()
        if crop_len:
            wav = random_crop(wav, crop_len, g)
        swa(fe(wav))


def run_fold(W, y, fold_ids, test_fold, args, device, seed):
    # seed everything for this (fold, seed) run so ensemble members differ
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    tr = np.where(fold_ids != test_fold)[0]
    te = np.where(fold_ids == test_fold)[0]
    Wtr, ytr = W[tr], y[tr]
    Wte, yte = W[te], y[te]
    n_tr = len(tr)

    use_coord = (args.model == "fresca")
    fe = LogMelFrontend(coord=use_coord).to(device).eval()
    in_ch = 2 if use_coord else 1
    net = MODELS[args.model](n_classes=N_CLASSES, **({"in_ch": in_ch} if args.model == "fresca" else {})).to(device)
    swa = AveragedModel(net)

    opt = make_optimizer(net, args.lr, args.wd)
    min_mult = args.eta_min / args.lr
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_lambda(e, args.warmup, args.epochs, min_mult))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    crop_len = int(args.crop_sec * SR) if args.crop else 0
    swa_start = max(1, args.epochs - args.swa_epochs)
    bs = args.batch
    best_acc, last_acc = 0.0, 0.0

    for ep in range(1, args.epochs + 1):
        # Between epochs, pause only if ANOTHER app is grabbing the GPU's VRAM.
        # We subtract our own reserved memory so our multi-GB training footprint
        # and 100% util never count as "busy". Only a genuine external app pauses
        # us; then we release our cache and wait for the GPU to clear.
        if not args.no_gate:
            reserved = torch.cuda.memory_reserved() / (1024 * 1024) if device == "cuda" else 0.0
            if gpu_gate.external_busy(reserved):
                if device == "cuda":
                    torch.cuda.empty_cache()
                gpu_gate.wait_for_idle(label=f"[fold{test_fold} ep{ep}]")

        net.train()
        perm = np.random.permutation(n_tr)
        ramp = min(1.0, ep / max(1, args.specaug_ramp))
        run_loss = 0.0
        for i in range(0, n_tr, bs):
            idx = np.sort(perm[i:i + bs])
            wav = torch.from_numpy(np.ascontiguousarray(Wtr[idx])).to(device).float()
            yb = torch.from_numpy(ytr[idx]).long().to(device)

            with torch.autocast("cuda", enabled=False):    # front-end/aug in fp32
                if crop_len:
                    wav = random_crop(wav, crop_len, g)
                if args.wave_aug:
                    wav = wave_shift(wav, 0.2, g)
                    wav = wave_gain(wav, 6.0, g)
                    wav = wave_noise(wav, 20.0, 40.0, 0.3, g)
                # base one-hot (+ label smoothing only when BC is off this batch)
                ls = 0.0 if args.bc else args.label_smooth
                y1h = torch.full((wav.size(0), N_CLASSES), ls / N_CLASSES, device=device)
                y1h.scatter_(1, yb.view(-1, 1), 1 - ls + ls / N_CLASSES)
                if args.bc:
                    wav, y1h = bc_mix(wav, y1h, g)
                spec = fe(wav)
                if args.spec_aug:
                    spec = spec_augment(spec, g, ramp=ramp)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.amp):
                loss = soft_ce(net(spec), y1h)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * wav.size(0)
        sched.step()
        if ep >= swa_start:
            swa.update_parameters(net)

        last_acc = evaluate(net, fe, Wte, yte, device, crop_len, tta=args.crop, n_crops=args.tta_crops)
        best_acc = max(best_acc, last_acc)
        if ep % args.log_every == 0 or ep == args.epochs:
            print(f"  fold{test_fold} ep{ep:3d}/{args.epochs} lr={sched.get_last_lr()[0]:.2e} "
                  f"loss={run_loss/n_tr:.3f} test={last_acc:5.2f} best={best_acc:5.2f}")

    # SWA final model (leakage-free headline)
    update_bn_swa(swa, fe, Wtr, device, crop_len, g)
    if getattr(args, "return_probs", False):
        swa_acc, swa_probs = evaluate(swa.module, fe, Wte, yte, device, crop_len,
                                      tta=args.crop, n_crops=args.tta_crops, return_probs=True)
    else:
        swa_acc = evaluate(swa.module, fe, Wte, yte, device, crop_len,
                           tta=args.crop, n_crops=args.tta_crops)
        swa_probs = None
    print(f"  fold{test_fold} SWA test = {swa_acc:5.2f}  (last={last_acc:.2f} best={best_acc:.2f})")
    res = {"test_fold": int(test_fold), "swa_acc": swa_acc,
           "final_acc": last_acc, "best_acc": best_acc}
    if swa_probs is not None:
        res["swa_probs"] = swa_probs      # numpy [N,50]; used by ensemble, not saved to JSON
        res["yte"] = yte                  # numpy [N]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="fresca")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eta-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--swa-epochs", type=int, default=30)
    ap.add_argument("--crop-sec", type=float, default=3.0)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--specaug-ramp", type=int, default=20)
    ap.add_argument("--aug", choices=["full", "none"], default="full")
    ap.add_argument("--dataset", choices=list(DATASETS), default="esc50")
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="default: every official fold of --dataset")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate-every", type=int, default=10)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta-crops", type=int, default=3)
    ap.add_argument("--return-probs", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # dataset selection: cache dir, label space and official fold list
    global N_CLASSES
    spec = DATASETS[args.dataset]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), spec["cache"])
    N_CLASSES = spec["n_classes"]
    if args.folds is None:
        args.folds = list(spec["folds"])

    # aug preset -> individual switches
    full = (args.aug == "full")
    args.crop = full
    args.wave_aug = full
    args.bc = full
    args.spec_aug = full

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    if device == "cpu":
        args.amp = False
        args.no_gate = True
    g = torch.Generator(device=device).manual_seed(args.seed)

    W = np.load(os.path.join(cache_dir, "waveforms.npy"), mmap_mode="r")
    meta = np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=True)
    y, fold_ids = meta["y"], meta["fold"]
    if args.limit:
        per = max(1, args.limit // len(args.folds))
        sel = np.concatenate([np.where(fold_ids == f)[0][:per] for f in args.folds])
        W, y, fold_ids = W[sel], y[sel], fold_ids[sel]

    n_params = count_params(MODELS[args.model](
        n_classes=N_CLASSES, **({"in_ch": 2} if args.model == "fresca" else {}))) / 1e6
    print(f"[train] dataset={args.dataset} classes={N_CLASSES} folds={args.folds} "
          f"clips={len(y)}")
    print(f"[train] model={args.model} params={n_params:.2f}M aug={args.aug} "
          f"epochs={args.epochs} batch={args.batch} device={device} amp={args.amp}")
    print(f"[train] crop={args.crop} bc={args.bc} wave_aug={args.wave_aug} "
          f"spec_aug={args.spec_aug} swa_last={args.swa_epochs}")

    if not args.no_gate:
        gpu_gate.wait_for_idle(label="(startup)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else f"_{args.aug}"
    prefix = "" if args.dataset == "esc50" else f"{args.dataset}_"
    out_path = os.path.join(RESULTS_DIR, f"{prefix}{args.model}{tag}.json")
    results, t0 = [], time.time()
    for f in args.folds:
        print(f"[train] === fold {f} === ({gpu_gate.snapshot()})")
        for attempt in range(1, 4):
            try:
                results.append(run_fold(W, y, fold_ids, f, args, device, args.seed))
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"[train] fold {f} OOM (attempt {attempt}); freeing + waiting for idle", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                if not args.no_gate:
                    gpu_gate.wait_for_idle(label=f"[fold{f} OOM-retry]")
        else:
            # never broke -> all OOM retries exhausted. Do NOT silently drop a
            # fold: that would make the mean a <5-fold average stamped complete.
            raise RuntimeError(f"fold {f} failed after 3 OOM retries")
        # save incrementally so completed folds survive an interruption
        save_summary(results, args, n_params, t0, out_path, partial=(len(results) < len(args.folds)))

    # partial is true iff we did not complete every requested fold
    save_summary(results, args, n_params, t0, out_path, partial=(len(results) != len(args.folds)))
    print("\n[train] ===== SUMMARY =====")
    for r in results:
        print(f"  fold{r['test_fold']}: SWA={r['swa_acc']:.2f} "
              f"final={r['final_acc']:.2f} best={r['best_acc']:.2f}")
    swa = [r["swa_acc"] for r in results]
    fin = [r["final_acc"] for r in results]
    print(f"  MEAN SWA (headline) = {np.mean(swa):.2f} +/- {np.std(swa):.2f}")
    print(f"  MEAN final          = {np.mean(fin):.2f} +/- {np.std(fin):.2f}")
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
