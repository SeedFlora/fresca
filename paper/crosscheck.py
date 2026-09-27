"""
Cross-check: every headline number printed in the compiled PDF must equal the
value derived from the raw result files. Guards against a stale figure
surviving an edit.
"""
import os
import json
import subprocess
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "benchmark", "results")
PDF = os.path.join(HERE, "FRESCA-Net-ESC50.pdf")
TXT = os.path.join(HERE, "_pdftext.txt")

subprocess.run(["pdftotext", PDF, TXT], check=True)
t = open(TXT, encoding="utf-8", errors="replace").read()

R = {k: json.load(open(os.path.join(RES, k + ".json"))) for k in
     ["fresca_full", "fresca_none", "piczak_full", "piczak_none",
      "fresca_ensemble", "us8k_fresca_full"]}
e = json.load(open(os.path.join(RES, "edge_profile.json")))
f, p = e["models"]["fresca"], e["models"]["piczak"]
us = [r["swa_acc"] for r in R["us8k_fresca_full"]["per_fold"]]
us_best = [r["best_acc"] for r in R["us8k_fresca_full"]["per_fold"]]

checks = [
    ("81.55", "%.2f" % R["fresca_full"]["mean_swa_acc"]),
    ("2.83",  "%.2f" % R["fresca_full"]["std_swa_acc"]),
    ("82.45", "%.2f" % R["fresca_ensemble"]["mean_swa_acc"]),
    ("1.60",  "%.2f" % R["fresca_ensemble"]["std_swa_acc"]),
    ("78.65", "%.2f" % R["fresca_none"]["mean_swa_acc"]),
    ("70.25", "%.2f" % R["piczak_full"]["mean_swa_acc"]),
    ("62.20", "%.2f" % R["piczak_none"]["mean_swa_acc"]),
    ("74.57", "%.2f" % st.mean(us)),
    ("6.10",  "%.2f" % st.pstdev(us)),
    ("76.48", "%.2f" % st.mean(us_best)),
    ("64.32", "%.2f" % min(us)),
    ("83.65", "%.2f" % max(us)),
    ("2.23",  "%.2f" % (f["backbone_macs_crop_M"] / 1000)),
    ("0.61",  "%.2f" % (p["backbone_macs_crop_M"] / 1000)),
    ("100.1", "%.1f" % f["latency"]["torch_cpu_1thread"]["median_ms"]),
    ("96.3",  "%.1f" % p["latency"]["torch_cpu_1thread"]["median_ms"]),
    ("9.3",   "%.1f" % f["latency"]["cuda"]["median_ms"]),
    ("3.0",   "%.1f" % f["latency"]["frontend_cpu_1thread"]["median_ms"]),
    ("45.2",  "%.1f" % f["peak_activation_mb_crop"]),
    ("9.7",   "%.1f" % p["peak_activation_mb_crop"]),
    ("11.0",  "%.1f" % f["size"]["fp32_mb"]),
    ("2.7",   "%.1f" % p["size"]["fp32_mb"]),
    ("2.8",   "%.1f" % f["size"]["int8_mb"]),
    ("6.77",  "%.2f" % (f["end_to_end_macs_3crop_M"] / 1000)),
]

bad = 0
for printed, derived in checks:
    in_pdf = printed in t
    ok = (printed == derived) and in_pdf
    if not ok:
        bad += 1
    print("%-7s derived=%-7s in_pdf=%-5s %s"
          % (printed, derived, in_pdf, "OK" if ok else "<-- CHECK"))

# derived quantities stated in prose
swa_minus_best = st.mean(us) - st.mean(us_best)
print("\nderived prose claims:")
print("  US8K SWA - best = %+.2f  (paper says 1.91 below)" % swa_minus_best)
print("  ESC50 SWA - best = %+.2f (paper says 1.35 above)"
      % (R["fresca_full"]["mean_swa_acc"]
         - st.mean([r["best_acc"] for r in R["fresca_full"]["per_fold"]])))
print("  MAC ratio fresca/piczak = %.2f (paper says 3.7)"
      % (f["backbone_macs_crop_M"] / p["backbone_macs_crop_M"]))
print("  latency ratio = %.2f (paper says 1.04 / four percent)"
      % (f["latency"]["torch_cpu_1thread"]["median_ms"]
         / p["latency"]["torch_cpu_1thread"]["median_ms"]))
print("  3-crop clip time = %.2f s, RTF %.3f (paper says 0.31 s, 0.06)"
      % (3 * (f["latency"]["torch_cpu_1thread"]["median_ms"]
              + f["latency"]["frontend_cpu_1thread"]["median_ms"]) / 1000,
         3 * (f["latency"]["torch_cpu_1thread"]["median_ms"]
              + f["latency"]["frontend_cpu_1thread"]["median_ms"]) / 5000))
# exact per-stage sum (top_layers_macs_M holds only the 8 heaviest layers,
# which undercounts; benchmark/stage_macs.py does the full breakdown)
import subprocess as _sp
_out = _sp.run([__import__("sys").executable,
                os.path.join(os.path.dirname(HERE), "benchmark", "stage_macs.py")],
               capture_output=True, text=True).stdout
for _l in _out.splitlines():
    if "stage3 + stage4" in _l:
        print("  " + _l.strip() + "   (paper says 1.64 G / 73 percent)")

os.remove(TXT)
print("\nmismatches:", bad)
