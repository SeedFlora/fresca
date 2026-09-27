"""Generate the per-fold accuracy figure for the FRESCA-Net ESC-50 paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stix",
})

folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
single = [79.50, 82.25, 82.75, 85.75, 77.50]      # fresca_full SWA per fold
ensemble = [82.25, 82.75, 82.75, 84.75, 79.75]     # 3-seed ensemble per fold
mean_single = 81.55
mean_ens = 82.45

x = np.arange(len(folds))
w = 0.38

fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
c1, c2 = "#4C78A8", "#F58518"
b1 = ax.bar(x - w / 2, single, w, label="Single model", color=c1, edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w / 2, ensemble, w, label="3-seed ensemble", color=c2, edgecolor="black", linewidth=0.5)

# reference lines
ax.axhline(81.3, ls=(0, (4, 3)), lw=0.9, color="#555555")
ax.text(4.55, 81.3, "Human 81.3", ha="right", va="bottom", fontsize=6.5, color="#555555")
ax.axhline(84.9, ls=(0, (1, 2)), lw=0.9, color="#888888")
ax.text(4.55, 84.9, "SOTA 84.9", ha="right", va="bottom", fontsize=6.5, color="#888888")

ax.set_ylim(74, 88)
ax.set_ylabel("Test accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(folds, fontsize=8)
ax.set_yticks(np.arange(74, 89, 2))
ax.tick_params(width=0.7, length=3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", fontsize=6.8, frameon=False, ncol=1,
          handlelength=1.1, borderaxespad=0.2)
ax.grid(axis="y", ls=":", lw=0.5, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.3)
fig.savefig("fig_folds.pdf", bbox_inches="tight")
print("wrote fig_folds.pdf   single mean=%.2f  ensemble mean=%.2f" % (mean_single, mean_ens))
