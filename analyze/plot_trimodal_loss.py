"""Figure: tri-modal LoRA training mechanism (Methodology Sec. 4.6.2).

Pure schematic, no data: the four inputs (drone image D, GT-centred satellite
crop S, captions T_S and T_D), the two LoRA-adapted towers, and the shared
embedding space in which the four weighted symmetric-InfoNCE pairings of the
total loss pull matched embeddings together. Node/arrow colors separate the
image pathway from the text pathway; the absent edges (S-T_D, T_S-T_D) are
absent in the loss too.

Usage:
    python analyze/plot_trimodal_loss.py --out thesis/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

IMG_C, TXT_C, LOSS_C = "#1f77b4", "#ff7f0e", "#c0392b"


def box(ax, x, y, text, color, w=2.1, h=1.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.08",
                                fc="white", ec=color, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="black")


def arrow(ax, p0, p1, color, lw=1.2):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=2, shrinkB=2))


def loss_edge(ax, p0, p1, label, lab_xy):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ls="--", lw=1.6, color=LOSS_C,
            zorder=1)
    ax.text(*lab_xy, label, fontsize=11, color=LOSS_C,
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # inputs
    box(ax, 1.45, 5.15, "drone image\n$D$", IMG_C)
    box(ax, 1.45, 3.70, "satellite crop $S$\n(GT-centred)", IMG_C)
    box(ax, 1.45, 2.25, "caption of $S$\n$T_S$", TXT_C)
    box(ax, 1.45, 0.80, "caption of $D$\n$T_D$", TXT_C)

    # towers (LoRA = only trainable weights)
    box(ax, 4.75, 4.42, "vision tower\n+ LoRA", IMG_C, w=2.0, h=1.1)
    box(ax, 4.75, 1.52, "text tower\n+ LoRA", TXT_C, w=2.0, h=1.1)

    arrow(ax, (2.6, 5.15), (3.7, 4.62), IMG_C)
    arrow(ax, (2.6, 3.70), (3.7, 4.22), IMG_C)
    arrow(ax, (2.6, 2.25), (3.7, 1.72), TXT_C)
    arrow(ax, (2.6, 0.80), (3.7, 1.32), TXT_C)

    # shared embedding space
    ax.add_patch(Ellipse((9.55, 3.0), 5.3, 5.5, fc="#f7f7f7", ec="0.55",
                         lw=1.2, zorder=0))
    ax.text(9.55, 0.55, "shared embedding space\n(L2-normalized)",
            ha="center", va="center", fontsize=9, color="0.35")

    arrow(ax, (5.85, 4.42), (7.15, 4.05), IMG_C, lw=1.8)
    arrow(ax, (5.85, 1.52), (7.15, 1.95), TXT_C, lw=1.8)

    # embedded points: square D S / T_D T_S
    nodes = {"D": (8.45, 4.35), "S": (10.75, 4.35),
             "T_D": (8.45, 1.85), "T_S": (10.75, 1.85)}
    # loss edges first (under the dots)
    loss_edge(ax, nodes["D"], nodes["S"], r"$w_{ds}$", (9.6, 4.62))
    loss_edge(ax, nodes["S"], nodes["T_S"], r"$w_{st}$", (11.15, 3.1))
    loss_edge(ax, nodes["D"], nodes["T_S"], r"$w_{dt}$", (10.0, 3.35))
    loss_edge(ax, nodes["D"], nodes["T_D"], r"$w_{ddt}$", (7.95, 3.1))
    for name, (x, y) in nodes.items():
        color = IMG_C if name in ("D", "S") else TXT_C
        ax.plot(x, y, "o", ms=10, color=color, zorder=3)
        dy = 0.38 if y > 3 else -0.42
        ax.text(x, y + dy, f"${name}$", ha="center", va="center", fontsize=11)

    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"trimodal_loss.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", path)


if __name__ == "__main__":
    main()
