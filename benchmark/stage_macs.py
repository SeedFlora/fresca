"""Exact MAC breakdown of FRESCA-Net by stage (for the claim in Section III-B)."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS
from profile_edge import count_macs, CROP_SEC, SR, HOP, N_MELS

net = MODELS["fresca"](n_classes=50, in_ch=2).eval()
T = int(CROP_SEC * SR) // HOP + 1
total, per_layer = count_macs(net, torch.randn(1, 2, N_MELS, T))

groups = {}
for name, kind, m in per_layer:
    g = name.split(".")[0] if name else "other"
    groups[g] = groups.get(g, 0.0) + m

print("input 2x%dx%d, total %.1f MMAC\n" % (N_MELS, T, total / 1e6))
for g, m in sorted(groups.items(), key=lambda kv: -kv[1]):
    print("  %-14s %8.1f MMAC  (%5.1f%%)" % (g, m / 1e6, 100 * m / total))

deep = groups.get("stage3", 0) + groups.get("stage4", 0)
print("\nstage3 + stage4 = %.1f MMAC = %.2f G  (%.1f%% of total)"
      % (deep / 1e6, deep / 1e9, 100 * deep / total))
print("total = %.2f G" % (total / 1e9))
