"""
Build the benchmark comparison table from results/*.json and place the new
model in the context of published ESC-50 results.

Published reference numbers are taken from the ESC-50 README leaderboard. We
compare against the *from-scratch / trained-on-ESC-50-only* family (no AudioSet
or other external pretraining) -- the fair comparison class for this work.
Large pretrained transformers (CLAP/BEATs/AST, 95-98%) use external data and
are listed only for orientation.
"""
import os
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# (label, accuracy%, category)  -- from README leaderboard
PUBLISHED = [
    ("k-NN (MFCC+ZCR), piczak2015a",                 32.2, "from-scratch baseline"),
    ("SVM (MFCC+ZCR), piczak2015a",                  39.6, "from-scratch baseline"),
    ("Random Forest (MFCC+ZCR), piczak2015a",        44.3, "from-scratch baseline"),
    ("5-layer CNN on raw audio (SoundNet, ESC-50)",  51.1, "from-scratch CNN"),
    ("3-layer CNN mel-STFT (huzaifah2017)",          56.4, "from-scratch CNN"),
    ("Piczak CNN (piczak2015b)",                     64.5, "from-scratch CNN"),
    ("EnvNet-v2 raw waveform (tokozume2017a)",       74.4, "from-scratch CNN"),
    ("Baseline CNN+BN+BC learning (tokozume2017b)",  76.9, "from-scratch + aug"),
    ("Multi-scale CNN (zhu2018)",                    79.1, "from-scratch + aug"),
    ("EnvNet-v2 + BC learning (tokozume2017b)",      81.8, "from-scratch + aug"),
    ("Human accuracy (piczak2015a)",                 81.3, "reference"),
    ("EnvNet-v2 + aug + BC learning (tokozume2017b)",84.9, "from-scratch + aug (SOTA)"),
    ("CLAP (pretrained, external data)",             96.7, "pretrained (out of scope)"),
    ("BEATs (pretrained, external data)",            98.1, "pretrained (out of scope)"),
]


def load_ours():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        aug = d.get("aug", "aug" if d.get("aug") else "no-aug")
        name = f"{d['model'].upper()} ({aug} aug)"
        # headline = SWA accuracy when present (leakage-free), else final-epoch
        acc = d.get("mean_swa_acc", d.get("mean_final_acc"))
        std = d.get("std_swa_acc", d.get("std_final_acc"))
        rows.append((name, acc, std,
                     d.get("mean_final_acc"), d.get("std_final_acc"),
                     d.get("params_millions"), d.get("epochs"), len(d.get("folds", []))))
    return rows


def main():
    ours = load_ours()
    print("=" * 78)
    print("ESC-50 BENCHMARK  --  proposed model vs published from-scratch methods")
    print("=" * 78)
    if not ours:
        print("(no results yet -- run benchmark/run_all.py first)")
    else:
        print("\nOUR RUNS (official 5-fold CV; headline = SWA-averaged model, leakage-free):")
        print(f"  {'model':26s} {'SWA acc':>13s} {'final acc':>13s} {'params':>8s}")
        for n, mf, sf, mb, sb, pm, ep, nf in ours:
            bb = f"{mb:.2f}+/-{sb:.2f}" if mb is not None else "-"
            pp = f"{pm:.2f}M" if pm else "-"
            print(f"  {n:26s} {mf:6.2f}+/-{sf:4.2f} {bb:>13s} {pp:>8s}  ({nf} folds)")

    print("\nPUBLISHED REFERENCE (ESC-50 leaderboard):")
    for label, acc, cat in PUBLISHED:
        print(f"  {acc:5.1f}%   {label:48s} [{cat}]")

    # ---- load raw runs keyed by (model, aug) for the ablation matrix ----
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        runs[(d["model"], d.get("aug"))] = d

    def acc(model, aug):
        d = runs.get((model, aug))
        if not d:
            return None
        return d.get("mean_swa_acc", d.get("mean_final_acc")), d.get("std_swa_acc", d.get("std_final_acc"))

    def cell(model, aug):
        a = acc(model, aug)
        return f"{a[0]:.2f}% ± {a[1]:.2f}" if a else "—"

    def delta(m1, a1, m2, a2):
        x, y = acc(m1, a1), acc(m2, a2)
        return f"{x[0] - y[0]:+.2f} pt" if (x and y) else "—"

    fresca_full = acc("fresca", "full")
    fresca_ens = acc("fresca", "ensemble")

    out = os.path.join(HERE, "BENCHMARK.md")
    md = []
    md.append("# ESC-50 Benchmark — FRESCA-Net\n")
    md.append("Official **5-fold cross-validation**, trained **from scratch on ESC-50 only** "
              "(no external data / no pretraining). Each of the 2000 clips is tested exactly "
              "once as part of its held-out fold. Accuracy is the **SWA-averaged model with "
              "multi-crop test-time augmentation** (label-free), under the leakage-free "
              "protocol described below.\n")

    # merge our entries with the published leaderboard and sort by accuracy
    rows = [(lab, a, cat, False) for lab, a, cat in PUBLISHED]
    if fresca_ens:
        rows.append(("FRESCA-Net, 3-seed ensemble (ours)", fresca_ens[0],
                     f"proposed (±{fresca_ens[1]:.2f})", True))
    if fresca_full:
        rows.append(("FRESCA-Net, single model (ours)", fresca_full[0],
                     f"proposed (±{fresca_full[1]:.2f})", True))
    rows.sort(key=lambda r: r[1], reverse=True)

    md.append("## Result in context (from-scratch methods)\n")
    md.append("| Method | Accuracy | Class |")
    md.append("|---|---|---|")
    for lab, a, cat, ours in rows:
        star = "**" if ours else ""
        md.append(f"| {star}{lab}{star} | {star}{a:.2f}%{star} | {cat} |")

    # 2x2 ablation matrix
    md.append("\n## Ablation (same pipeline & protocol)\n")
    md.append("| Architecture | no augmentation | full aug + SWA + 3-crop TTA |")
    md.append("|---|---|---|")
    md.append(f"| **FRESCA-Net** (2.88M) | {cell('fresca','none')} | {cell('fresca','full')} |")
    md.append(f"| Piczak CNN (0.71M) | {cell('piczak','none')} | {cell('piczak','full')} |")
    md.append("\n**Contribution decomposition (matched recipe):**\n")
    md.append(f"- Architecture, no-aug (FRESCA-Net vs Piczak): **{delta('fresca','none','piczak','none')}**")
    md.append(f"- Augmentation on FRESCA-Net (full vs none): **{delta('fresca','full','fresca','none')}**")
    md.append(f"- Augmentation on Piczak (full vs none): **{delta('piczak','full','piczak','none')}**")

    md.append("\n## Protocol (leakage-free)\n")
    md.append("- Official 5 folds; for fold *k*, train on the other 4, test on *k*.")
    md.append("- **No** early-stopping / checkpoint / EMA selection on the test fold; fixed 200-epoch budget.")
    md.append("- Headline = SWA-averaged weights (last 30 epochs), BN recomputed on training folds only.")
    md.append("- Inference uses **3-crop TTA** (label-free averaging of 3 evenly-spaced 3 s crops) — standard for ESC-50.")
    md.append("- Per-instance log-mel standardization (no train/test statistic sharing).")
    md.append("- Pretrained transformers (CLAP/BEATs/AST, 95–98%) use external data and are out of scope; shown only for orientation.\n")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\n[report] wrote {out}")


if __name__ == "__main__":
    main()
