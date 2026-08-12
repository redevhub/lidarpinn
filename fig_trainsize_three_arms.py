"""
Training-set size sweep, three architecture-matched arms overlaid.

Reads the three trainsize_sweep_summary.json files produced by
trainsize_sweep_three_arms.py (one per --physics-type) and plots speed RMSE
and direction MAE against training-set size, contiguous-month and
distributed (days-per-month) sampling shown with different markers.

Usage:

    python3 fig_trainsize_three_arms.py \
        --ekman trainsize_ekman/trainsize_sweep_summary.json \
        --shear trainsize_shear/trainsize_sweep_summary.json \
        --none  trainsize_noek/trainsize_sweep_summary.json \
        -o fig_trainsize_three_arms
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
import matplotlib.pyplot as plt

COLORS = {"Ekman-constrained": "#1f6fb4", "Shear-constrained": "#2a9d8f", "Physics-free": "#d1495b"}


def load(path):
    d = json.load(open(path))
    runs = d["runs"]
    contig = sorted([r for r in runs if r["scheme"] == "contiguous"], key=lambda r: r["n_train"])
    distrib = sorted([r for r in runs if r["scheme"] == "distributed"], key=lambda r: r["n_train"])
    full = [r for r in runs if r["scheme"] == "full"]
    return contig, distrib, full


def main(a):
    arms = [("Ekman-constrained", a.ekman), ("Shear-constrained", a.shear), ("Physics-free", a.none)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, ylabel, title in [
            (axes[0], "rmse", "Test speed RMSE (m/s)", "(a) Speed RMSE"),
            (axes[1], "dir_mae", "Test direction MAE (deg)", "(b) Direction MAE")]:
        for name, path in arms:
            contig, distrib, full = load(path)
            color = COLORS[name]
            x_c = [r["n_train"] for r in contig]; y_c = [r[key] for r in contig]
            ax.plot(x_c, y_c, "o-", color=color, ms=4, lw=1.6, label=f"{name} (contiguous)")
            if distrib:
                x_d = [r["n_train"] for r in distrib]; y_d = [r[key] for r in distrib]
                ax.plot(x_d, y_d, "s--", color=color, ms=4, lw=1.4, alpha=0.7,
                        label=f"{name} (distributed)")
            if full:
                ax.plot(full[0]["n_train"], full[0][key], "*", color=color, ms=12,
                        markeredgecolor="k", markeredgewidth=0.5)
        ax.set_xlabel("Number of training samples", fontsize=12.5)
        ax.set_ylabel(ylabel, fontsize=12.5)
        ax.set_title(title, fontsize=12.5)
        ax.grid(alpha=.3)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("written", a.out + ".png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ekman", required=True)
    p.add_argument("--shear", required=True)
    p.add_argument("--none", required=True)
    p.add_argument("-o", "--out", default="fig_trainsize_three_arms")
    main(p.parse_args())
