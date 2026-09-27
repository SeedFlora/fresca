"""
Simple, clean FRESCA-Net architecture diagram in the style of a staged backbone
figure: an input embedding, four dashed stage blocks (each = downsampling + SE
block, repeated twice) with tensor-shape annotations on top, a pooling/head, and
a separate compact panel for the SE basic block. Minimal colour, thin borders,
no connector clutter between panels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.4,
})

WHITE = ("#FFFFFF", "#5B6570")     # plain box
BLK = ("#E5EEF8", "#5B8DC8")       # SE block (light blue accent)
DOWN = ("#EEEFF2", "#8A8F98")      # downsampling
HEAD = ("#E3F0E1", "#5AAE6A")      # pooling / output (light green)

fig, ax = plt.subplots(figsize=(7.1, 3.15), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(cx, cy, w, h, text, col, fs=6.6, rot=0, lw=1.0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.1,rounding_size=1.0",
                 fc=col[0], ec=col[1], lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            rotation=rot, linespacing=1.1, zorder=4)


def dbox(x0, x1, y0, y1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.1,rounding_size=1.4",
                 fc="none", ec="#8A8F98", lw=1.0, ls=(0, (4, 2.5)), zorder=1))


def arr(x1, y1, x2, y2, rad=0.0, lw=1.15, color="#3B4048"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=9, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))


# ===================== (a) architecture =====================
ax.text(1.0, 94, "(a)  FRESCA-Net architecture", ha="left", va="center",
        fontsize=8.7, fontweight="bold")

ya = 60          # block centre line
hb = 15          # block height

# input embedding
box(6.0, ya, 9.0, 24, "Log-mel\n+ CoordConv", WHITE, fs=6.4)
ax.text(6.0, ya - 15, "2$\\times$128$\\times$259", ha="center", va="top",
        fontsize=5.8, style="italic", color="#444")

# stage definitions: (container x0, x1, has_downsample, block_cx, shape, C)
stages = [
    (12.5, 24.0, False, 18.2, "F/2 $\\times$ T/2 $\\times$ 32"),
    (26.5, 43.0, True, 36.5, "F/4 $\\times$ T/4 $\\times$ 64"),
    (45.5, 62.0, True, 55.5, "F/8 $\\times$ T/4 $\\times$ 128"),
    (64.5, 81.0, True, 74.5, "F/16 $\\times$ T/4 $\\times$ 256"),
]
arr(10.6, ya, 12.5, ya)
for i, (x0, x1, down, bcx, shape) in enumerate(stages, 1):
    dbox(x0, x1, ya - 11, ya + 11)
    if down:
        dcx = x0 + 3.2
        box(dcx, ya, 4.2, hb, "Down\nsample", DOWN, fs=5.4, rot=90)
        box(bcx, ya, 9.5, hb, "SE\nBlock", BLK, fs=6.6)
        arr(dcx + 2.1, ya, bcx - 4.75, ya)
    else:
        box(bcx, ya, 9.5, hb, "SE\nBlock", BLK, fs=6.6)
    ax.text((x0 + x1) / 2, ya + 15.5, shape, ha="center", va="center",
            fontsize=5.8, style="italic", color="#444")
    ax.text((x0 + x1) / 2, ya - 12.6, "Stage %d  ($\\times$2)" % i, ha="center",
            va="top", fontsize=6.4)
    if i < 4:
        arr(x1, ya, stages[i][0] if i < 4 else x1, ya)  # into next container

# arrow from stage4 into head
arr(81.0, ya, 84.5, ya)
box(92.0, ya, 15.5, 18, "Attentive pool\n$+$ FC\n$\\Rightarrow$ “Dog bark”",
    HEAD, fs=6.2)

# ===================== (b) SE basic block =====================
ax.text(1.0, 35, "(b)  SE basic block", ha="left", va="center",
        fontsize=8.7, fontweight="bold")

by = 18
chain = [(15, ["Conv3$\\times$3", "stride s"]), (28, ["BN, ReLU"]),
         (40, ["Conv3$\\times$3"]), (51, ["BN"]), (62, ["SE"])]
ax.text(6.0, by, "$x$", ha="center", va="center", fontsize=9)
arr(7.6, by, 15 - 5.0, by)
for j, (cx, lines) in enumerate(chain):
    col = BLK if lines == ["SE"] else WHITE
    box(cx, by, 9.6, 7.0, "\n".join(lines), col, fs=6.2)
    if j > 0:
        arr(chain[j - 1][0] + 4.8, by, cx - 4.8, by)
# sum + relu + out
xsum = 71.5
arr(62 + 4.8, by, xsum - 2.3, by)
ax.text((62 + 4.8 + xsum - 2.3) / 2, by - 2.7, "DropPath", ha="center", va="top",
        fontsize=5.4, style="italic", color="#666")
ax.add_patch(plt.Circle((xsum, by), 2.1, fc="white", ec="#3B4048", lw=1.0, zorder=4))
ax.text(xsum, by, "$+$", ha="center", va="center", fontsize=9.5, zorder=5)
box(xsum + 8.5, by, 8.0, 7.0, "ReLU", WHITE, fs=6.4)
arr(xsum + 2.1, by, xsum + 8.5 - 4.0, by)
arr(xsum + 8.5 + 4.0, by, xsum + 8.5 + 4.0 + 4.2, by)
ax.text(xsum + 8.5 + 4.0 + 5.0, by, "out", ha="left", va="center",
        fontsize=7.2, style="italic")
# clean identity skip arc (flat, well clear of the title)
arr(6.4, by + 2.6, xsum - 0.3, by + 2.4, rad=-0.17, lw=1.1)
ax.text((6.4 + xsum) / 2, by + 6.7, "identity, or 1$\\times$1 conv if C or s changes",
        ha="center", va="center", fontsize=5.7, style="italic", color="#444")
# SE expansion as one simple line
ax.text(50, by - 12.5,
        "SE:  GAP $\\rightarrow$ FC $\\downarrow$C/r $\\rightarrow$ ReLU "
        "$\\rightarrow$ FC $\\uparrow$C $\\rightarrow$ $\\sigma$ $\\rightarrow$ $\\otimes$ scale",
        ha="center", va="center", fontsize=6.2, color="#D9822B")

fig.savefig("fig_arch.pdf", bbox_inches="tight")
print("wrote fig_arch.pdf")
