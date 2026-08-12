"""
Per-altitude RMSE reduction of the shear-constrained arm relative to each
classical/tree-based baseline (positive = shear-constrained better).

The three physics-informed/physics-free arms are in-distribution
statistically indistinguishable from one another, the
shear-constrained arm (the best of the three by a negligible margin) is used
as the reference against the classical baselines, where the gap is real; a
comparison among the three architecture-matched arms belongs in the text
(Kruskal-Wallis, paired Wilcoxon) rather than in a figure, since all three
curves would sit at approximately zero.

Reads results.json (mean over seeds for the arm) and the classical baselines'
results.json (single fixed model each).

Usage:

    python3 fig_indist_rmse_reduction.py \
        --ref "Shear-constrained=out_shear_indist_s*/pinn_ekman_results.json" \
        --classical "Random forest=baselines_fixed_np18/random_forest/*_results.json" \
        --classical "Power law=baselines_fixed_np18/power_law/*_results.json" \
        --classical "Log law=baselines_fixed_np18/log_law/*_results.json" \
        --classical "Constant=baselines_fixed_np18/constant/*_results.json" \
        -o fig_indist_rmse_reduction
"""
import argparse, glob, json
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

COLORS = ["#888888", "#c9a227", "#8a5a44", "#5c8a5c", "#d1495b"]


def load_ref(patt):
    paths = sorted(glob.glob(patt))
    if not paths:
        raise SystemExit(f"no files for reference: {patt}")
    rs = [json.load(open(p)) for p in paths]
    alts = np.array(rs[0]["altitudes"], dtype=float)
    rmse = np.array([r["rmse_spd"] for r in rs], dtype=float)
    return alts, rmse.mean(0)


def load_classical(path):
    r = json.load(open(path))
    return np.array(r["altitudes"], dtype=float), np.array(r["rmse_spd"], dtype=float)


def main(a):
    ref_name, ref_patt = a.ref.split("=", 1)
    alts_ref, rmse_ref = load_ref(ref_patt)
    order = np.argsort(alts_ref)
    alts_s = alts_ref[order]
    rmse_ref_s = rmse_ref[order]

    surface_mask = alts_s > 0.0

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, spec in enumerate(a.classical):
        name, patt = spec.split("=", 1)
        paths = sorted(glob.glob(patt))
        if not paths:
            raise SystemExit(f"no files for {name}: {patt}")
        alts_c, rmse_c = load_classical(paths[0])
        corder = np.argsort(alts_c)
        rmse_c_s = rmse_c[corder]
        reduction = 100.0 * (rmse_c_s - rmse_ref_s) / rmse_c_s
        ax.plot(reduction[surface_mask], alts_s[surface_mask], "-o", ms=4, lw=1.6,
                color=COLORS[i % len(COLORS)], label=f"vs {name}")

    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"RMSE reduction of {ref_name} (%)", fontsize=12.5)
    ax.set_ylabel("Height a.g.l. (m)", fontsize=12.5)
    ax.set_title(f"Per-altitude RMSE reduction (positive = {ref_name} better)",
                fontsize=12.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("written", a.out + ".png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True,
                   help="Name=glob for the reference arm (averaged over seeds)")
    p.add_argument("--classical", action="append", required=True,
                   help="Name=glob for a classical baseline, repeatable")
    p.add_argument("-o", "--out", default="fig_indist_rmse_reduction")
    main(p.parse_args())
