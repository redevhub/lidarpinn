"""
Per-altitude RMSE, MAE and direction MAE, in-distribution, for N arms
sharing the same architecture (Ekman-constrained, Shear-constrained, Physics-free).

Reads the aggregate results.json of each seed (rmse_spd, mae_spd, mae_dir
arrays over the 13 altitudes) and averages over seeds.

Usage:

    python3 fig_indist_by_altitude.py \
        --arm "Ekman-constrained=out_pinn_indist_s*/pinn_ekman_results.json" \
        --arm "Shear-constrained=out_shear_indist_s*/pinn_ekman_results.json" \
        --arm "Physics-free=out_noek_indist_s*/pinn_ekman_results.json" \
        --classical "Random forest=baselines_fixed_np18/random_forest/*_results.json" \
        --classical "Power law=baselines_fixed_np18/power_law/*_results.json" \
        --classical "Log law=baselines_fixed_np18/log_law/*_results.json" \
        --classical "Constant=baselines_fixed_np18/constant/*_results.json" \
        -o fig_indist_by_altitude

--arm entries are averaged over N seeds with a min-max shaded envelope (solid
lines, filled band). --classical entries are single fixed models (dashed
lines, no envelope).
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

COLORS = ["#1f6fb4", "#2a9d8f", "#d1495b", "#e9a23b"]
MARKERS = ["o", "^", "s", "D"]


def load_arm(paths):
    rs = [json.load(open(p)) for p in paths]
    alts = np.array(rs[0]["altitudes"], dtype=float)
    rmse = np.array([r["rmse_spd"] for r in rs], dtype=float)
    mae  = np.array([r["mae_spd"]  for r in rs], dtype=float)
    dire = np.array([r["mae_dir"]  for r in rs], dtype=float)
    return alts, rmse, mae, dire


CLASSICAL_COLORS = ["#888888", "#c9a227", "#8a5a44", "#5c8a5c"]
CLASSICAL_LS = ["--", "-.", ":", (0, (3, 1, 1, 1))]


def load_classical(path):
    r = json.load(open(path))
    alts = np.array(r["altitudes"], dtype=float)
    rmse = np.array(r["rmse_spd"], dtype=float)
    mae  = np.array(r["mae_spd"], dtype=float)
    dire = np.array(r["mae_dir"], dtype=float)
    return alts, rmse, mae, dire


def main(a):
    arms = []
    for spec in a.arm:
        name, patt = spec.split("=", 1)
        paths = sorted(glob.glob(patt))
        if not paths:
            raise SystemExit(f"no files for {name}: {patt}")
        arms.append((name, load_arm(paths)))

    classical = []
    for spec in a.classical:
        name, patt = spec.split("=", 1)
        paths = sorted(glob.glob(patt))
        if not paths:
            raise SystemExit(f"no files for classical baseline {name}: {patt}")
        classical.append((name, load_classical(paths[0])))

    order = np.argsort(arms[0][1][0])
    alts_s = arms[0][1][0][order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.0), sharey=True)
    for ax, idx, xlabel, title in [
            (axes[0], 1, "Wind-speed RMSE (m/s)", "(a) Speed RMSE"),
            (axes[1], 2, "Wind-speed MAE (m/s)",  "(b) Speed MAE"),
            (axes[2], 3, "Direction MAE (deg)",   "(c) Direction MAE")]:
        for i, (name, data) in enumerate(arms):
            V = data[idx][:, order]
            vm, vlo, vhi = V.mean(0), V.min(0), V.max(0)
            ax.fill_betweenx(alts_s, vlo, vhi, color=COLORS[i % len(COLORS)],
                             alpha=0.18, lw=0)
            ax.plot(vm, alts_s, "-", marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], ms=4, lw=1.8, label=name)
        for j, (name, cdata) in enumerate(classical):
            calts, crmse, cmae, cdire = cdata
            corder = np.argsort(calts)
            V = [crmse, cmae, cdire][idx - 1][corder]
            ax.plot(V, calts[corder],
                    linestyle=CLASSICAL_LS[j % len(CLASSICAL_LS)],
                    color=CLASSICAL_COLORS[j % len(CLASSICAL_COLORS)],
                    lw=1.3, label=name)
        ax.set_xlabel(xlabel, fontsize=12.5)
        ax.set_title(title, fontsize=12.5)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(0, 310)
    axes[0].set_ylabel("Height a.g.l. (m)", fontsize=12.5)
    h, l = axes[0].get_legend_handles_labels()
    ncol = min(4, len(arms) + len(classical))
    leg = fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False,
                     fontsize=12.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight",
               bbox_extra_artists=(leg,))
    fig.savefig(a.out + ".pdf", bbox_inches="tight", bbox_extra_artists=(leg,))
    print("written", a.out + ".png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--classical", action="append", default=[],
                   help="Name=glob for a single fixed classical/tree baseline (no seed averaging)")
    p.add_argument("-o", "--out", default="fig_indist_by_altitude")
    main(p.parse_args())
