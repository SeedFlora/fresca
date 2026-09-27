"""
Re-derive every number the manuscript asserts, straight from the result JSONs.

Run this before touching the .tex so that any claim that has drifted from the
data is caught rather than copied forward.
"""
import os
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

R = {}
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    k = os.path.splitext(os.path.basename(f))[0]
    if "edge" in k or "smoke" in k:
        continue
    R[k] = json.load(open(f))

CFGS = ["fresca_full", "fresca_none", "piczak_full", "piczak_none"]

print("--- per-fold SWA accuracy by configuration ---")
rank = {}
for k in CFGS:
    acc = [p["swa_acc"] for p in R[k]["per_fold"]]
    best, worst = int(np.argmax(acc)) + 1, int(np.argmin(acc)) + 1
    order = [int(i) + 1 for i in np.argsort(acc)]
    print(f"{k:13s} {['%.2f' % a for a in acc]} easiest=f{best} hardest=f{worst} asc={order}")
    rank[k] = (best, worst)

e = R["fresca_ensemble"]["per_fold"]
ea = [p["ensemble_acc"] for p in e]
rank["ensemble"] = (int(np.argmax(ea)) + 1, int(np.argmin(ea)) + 1)
print(f"{'ensemble':13s} {['%.2f' % a for a in ea]} "
      f"easiest=f{rank['ensemble'][0]} hardest=f{rank['ensemble'][1]}")

print()
print("fold4 easiest in", sum(1 for v in rank.values() if v[0] == 4), "of", len(rank))
print("fold5 hardest in", sum(1 for v in rank.values() if v[1] == 5), "of", len(rank))

print("\n--- SWA table (last / best / swa) ---")
for k in CFGS:
    b = np.mean([p["best_acc"] for p in R[k]["per_fold"]])
    print(f"{k:13s} last={R[k]['mean_final_acc']:.2f} best={b:.2f} swa={R[k]['mean_swa_acc']:.2f}")

print("\n--- ablation deltas ---")
ff, fn = R["fresca_full"]["mean_swa_acc"], R["fresca_none"]["mean_swa_acc"]
pf, pn = R["piczak_full"]["mean_swa_acc"], R["piczak_none"]["mean_swa_acc"]
print(f"arch @ no-aug (fresca-piczak) = {fn - pn:+.2f}")
print(f"aug on fresca                 = {ff - fn:+.2f}")
print(f"aug on piczak                 = {pf - pn:+.2f}")
print(f"std: fresca_full={R['fresca_full']['std_swa_acc']:.2f} "
      f"fresca_none={R['fresca_none']['std_swa_acc']:.2f} "
      f"piczak_full={R['piczak_full']['std_swa_acc']:.2f} "
      f"piczak_none={R['piczak_none']['std_swa_acc']:.2f}")

print("\n--- ensemble ---")
seeds = [p["seed_accs"] for p in e]
em = R["fresca_ensemble"]["mean_swa_acc"]
print(f"mean single seed={np.mean(seeds):.4f} ensemble={em:.2f} "
      f"gain={em - np.mean(seeds):+.3f} std={R['fresca_ensemble']['std_swa_acc']:.2f}")
for p in e:
    sm = np.mean(p["seed_accs"])
    print(f"  fold{p['test_fold']} ens={p['ensemble_acc']:.2f} seedmean={sm:.2f} "
          f"gain={p['ensemble_acc'] - sm:+.2f}")

print("\n--- config actually used (for the Methods section) ---")
for k in CFGS + ["fresca_ensemble"]:
    d = R[k]
    print(f"{k:16s} epochs={d.get('epochs')} swa_epochs={d.get('swa_epochs', 'default=30')} "
          f"warmup={d.get('warmup', 'default=5')} params={d.get('params_millions')}")

# UrbanSound8K, once it exists
us = {k: v for k, v in R.items() if k.startswith("us8k")}
if us:
    print("\n--- UrbanSound8K ---")
    for k, d in us.items():
        acc = [p["swa_acc"] for p in d["per_fold"]]
        print(f"{k}: n_folds={len(acc)} mean={np.mean(acc):.2f} +/- {np.std(acc):.2f}")
        print(f"   per-fold {['%.2f' % a for a in acc]}")
        print(f"   last={d['mean_final_acc']:.2f} "
              f"best={np.mean([p['best_acc'] for p in d['per_fold']]):.2f} "
              f"epochs={d.get('epochs')} partial={d.get('partial')}")
