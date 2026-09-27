"""
Replace the provisional numbers in the manuscript with the measured ones.

Reads the authoritative result files and rewrites the edge-cost table, the
derived three-crop figures, and the UrbanSound8K accuracy everywhere they
appear, so no hand-typed number can drift from the data.

    python finalize_numbers.py            # report what would change
    python finalize_numbers.py --write    # apply
"""
import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "benchmark", "results")
TEX = os.path.join(HERE, "FRESCA-Net-ESC50.tex")
WRITE = "--write" in sys.argv

edge = json.load(open(os.path.join(RES, "edge_profile.json")))
f = edge["models"]["fresca"]
p = edge["models"]["piczak"]


def lat(model, key):
    v = model["latency"].get(key)
    return v["median_ms"] if isinstance(v, dict) and "median_ms" in v else None


cpu_f = lat(f, "torch_cpu_1thread")
cpu_p = lat(p, "torch_cpu_1thread")
fe_ms = lat(f, "frontend_cpu_1thread")
gpu_f = lat(f, "cuda")

# three-crop cost of one 5 s clip on one core
clip_s = 3 * (cpu_f + fe_ms) / 1000.0
rtf = clip_s / 5.0

vals = {
    "cpu_f": "%.1f" % cpu_f,
    "cpu_p": "%.1f" % cpu_p,
    "fe_ms": "%.1f" % fe_ms,
    "gpu_f": "%.1f" % gpu_f,
    "macs_f": "%.2f" % (f["backbone_macs_crop_M"] / 1000.0),
    "macs_p": "%.2f" % (p["backbone_macs_crop_M"] / 1000.0),
    "clip_s": "%.2f" % clip_s,
    "rtf": "%.2f" % rtf,
    "e2e3": "%.2f" % (f["end_to_end_macs_3crop_M"] / 1000.0),
    "act_f": "%.1f" % f["peak_activation_mb_crop"],
    "act_p": "%.1f" % p["peak_activation_mb_crop"],
    "w_f": "%.1f" % f["size"]["fp32_mb"],
    "w_p": "%.1f" % p["size"]["fp32_mb"],
    "int8_f": "%.1f" % f["size"]["int8_mb"],
    "cpu": edge["host"].get("cpu_name", "?"),
}

us_path = os.path.join(RES, "us8k_fresca_full.json")
if os.path.isfile(us_path):
    us = json.load(open(us_path))
    accs = [r["swa_acc"] for r in us["per_fold"]]
    import statistics
    vals["us_mean"] = "%.2f" % (sum(accs) / len(accs))
    vals["us_std"] = "%.2f" % (statistics.pstdev(accs))
    vals["us_folds"] = str(len(accs))
    vals["us_partial"] = str(us.get("partial"))
else:
    vals["us_mean"] = None

print("measured values:")
for k, v in vals.items():
    print("  %-10s %s" % (k, v))

s = open(TEX, encoding="utf-8").read()
orig = s

# --- edge table rows (match on the row label, rewrite the numeric cells) ---
s = re.sub(r"(\\textbf\{FRESCA-Net\} & )[\d.]+ & [\d.]+ & [\d.]+ & [\d.]+ & [\d.]+",
           r"\g<1>2.88 & %s & %s & %s & %s" % (vals["macs_f"], vals["w_f"],
                                               vals["act_f"], vals["cpu_f"]), s)
s = re.sub(r"(Piczak CNN & )[\d.]+ & [\d.]+ & [\d.]+ & [\d.]+ & [\d.]+",
           r"\g<1>0.71 & %s & %s & %s & %s" % (vals["macs_p"], vals["w_p"],
                                               vals["act_p"], vals["cpu_p"]), s)

# --- table footnote ---
s = re.sub(r"Log-mel front-end adds 22\.9 MMAC and [\d.]+ms",
           "Log-mel front-end adds 22.9 MMAC and %sms" % vals["fe_ms"], s)
s = re.sub(r"FRESCA-Net on GPU: [\d.]+ms", "FRESCA-Net on GPU: %sms" % vals["gpu_f"], s)

# --- prose derived figures ---
s = s.replace("about 0.29 s on one core", "about %s s on one core" % vals["clip_s"])
s = re.sub(r"a real-time factor near [\d.]+",
           "a real-time factor near %s" % vals["rtf"], s)

# --- abstract latency ---
s = re.sub(r"11\.0 MB of weights and [\d.]+ ms per crop",
           "%s MB of weights and %s ms per crop" % (vals["w_f"], vals["cpu_f"]), s)

# --- UrbanSound8K accuracy, everywhere ---
if vals["us_mean"]:
    s = s.replace("USMEAN", vals["us_mean"])
    s = re.sub(r"(?<![\d.])79\.00(?= percent)", vals["us_mean"], s)

if s != orig:
    if WRITE:
        open(TEX, "w", encoding="utf-8").write(s)
        print("\nWROTE", TEX)
    else:
        print("\nchanges pending (re-run with --write)")
else:
    print("\nno changes needed")

if vals["us_mean"] is None:
    print("NOTE: UrbanSound8K result not present yet; USMEAN left in place.")
