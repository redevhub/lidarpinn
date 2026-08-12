"""
Figure: vertical-extrapolation error profiles (speed RMSE and direction MAE).

Two panels versus height: (a) wind-speed RMSE, (b) wind-direction MAE.
Curves per panel: Ekman-informed model, same network without
physics, the log law fitted on the supervised gates, and, optionally, a
shear-constrained arm. Lines are the mean over the training seeds; shaded
envelopes span min-max across seeds. Shaded height regions mark the
supervised gates, the selection band and the held-out test.

Usage:

    python3 fig_extrapolation_profiles.py \
        --pinn-glob "out_pinnB_s*/pinn_ekman_results.json" \
        --noek-glob "out_noekB_s*/pinn_ekman_results.json" \
        --loglaw baselines_extrap_h111/log_law/log_law_results.json \
        --shear-glob "out_shear_s*/pinn_ekman_results.json" \
        -o fig_extrapolation_profiles
"""
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUPERVISED_TOP = 111.0
BAND_TOP       = 155.0

C_PINN, C_NOEK, C_LOG, C_SHEAR = "#1f6fb4", "#d1495b", "#666666", "#2a9d8f"


def load_arm(paths):
    rs = [json.load(open(p)) for p in paths]
    alts = np.array(rs[0]["altitudes"], dtype=float)
    rmse = np.array([r["rmse_spd"] for r in rs], dtype=float)
    dirm = np.array([r["mae_dir"]  for r in rs], dtype=float)
    return alts, rmse, dirm


def main(a):
    pinn_paths = sorted(glob.glob(a.pinn_glob)) + ([a.pinn_extra] if a.pinn_extra else [])
    noek_paths = sorted(glob.glob(a.noek_glob)) + ([a.noek_extra] if a.noek_extra else [])
    if not pinn_paths or not noek_paths:
        raise SystemExit("No results JSON found; check --pinn-glob / --noek-glob.")

    alts, R_p, D_p = load_arm(pinn_paths)
    _,    R_o, D_o = load_arm(noek_paths)
    ll = json.load(open(a.loglaw))
    R_l = np.array(ll["rmse_spd"], dtype=float)
    D_l = np.array(ll["mae_dir"],  dtype=float)

    shear_paths = sorted(glob.glob(a.shear_glob)) if a.shear_glob else []
    have_shear = len(shear_paths) > 0
    if have_shear:
        _, R_s, D_s = load_arm(shear_paths)

    order = np.argsort(alts)
    alts_s = alts[order]
    def stat(x):
        xo = x[:, order]
        return xo.mean(0), xo.min(0), xo.max(0)

    fig, axes = plt.subplots(
        1, 2, figsize=(a.width, a.height), sharey=True,
        gridspec_kw={"width_ratios": [1, a.right_width], "wspace": 0.12})

    series = [("Ekman-constrained", R_p, D_p, C_PINN, "o")]
    if have_shear:
        series.append((a.shear_label, R_s, D_s, C_SHEAR, "^"))
    series.append(("Physics-free", R_o, D_o, C_NOEK, "s"))

    for ax, coord, xlabel, panel in [
            (axes[0], 0, "Wind-speed RMSE (m s$^{-1}$)", "(a)"),
            (axes[1], 1, "Wind-direction MAE (deg)",     "(b)")]:
        for label, R, D, color, marker in series:
            V = R if coord == 0 else D
            vm, vlo, vhi = stat(V)
            ax.fill_betweenx(alts_s, vlo, vhi, color=color, alpha=0.20, lw=0)
            ax.plot(vm, alts_s, "-", marker=marker, color=color, ms=4, lw=1.8,
                    label=label)
        Ls = (R_l if coord == 0 else D_l)[order]
        mask = alts_s > 0
        ax.plot(Ls[mask], alts_s[mask], "--", color=C_LOG, lw=1.6,
                label="Log law (fit $\\leq$ 111 m)")
        ax.axhspan(0, SUPERVISED_TOP, color="0.92", zorder=0)
        ax.axhspan(SUPERVISED_TOP, BAND_TOP, color="#fff3d6", zorder=0)
        ax.set_xlabel(xlabel, fontsize=12.5)
        ax.text(0.02, 0.985, panel, transform=ax.transAxes, va="top",
                fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(0, 310)

    axes[0].set_ylabel("Height a.g.l. (m)", fontsize=12.5)
    ax = axes[0]
    ax.text(0.985, 55/310,  "supervised\n($z \\leq$ 111 m)", transform=ax.transAxes,
            ha="right", fontsize=12.2, color="0.35")
    ax.text(0.985, 132/310, "selection band (140 m)", transform=ax.transAxes,
            ha="right", fontsize=12.2, color="#8a6d1a")
    ax.text(0.985, 235/310, "held-out test\n(170-300 m)", transform=ax.transAxes,
            ha="right", fontsize=12.2, color="0.35")

    axes[1].axvline(90, color="k", ls=":", lw=1)
    axes[1].set_xlim(right=a.dir_xlim)
    axes[1].text(88, 250, "random\ndirection", fontsize=12.8,
                 va="top", ha="right", color="0.3")

    h, l = axes[0].get_legend_handles_labels()
    ncol = 4 if have_shear else 3
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False,
               fontsize=12.0, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out + ".png", dpi=300)
    fig.savefig(a.out + ".pdf")
    print("written", a.out + ".png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pinn-glob", default="out_pinnB_s*/pinn_ekman_results.json")
    p.add_argument("--pinn-extra", default="")
    p.add_argument("--noek-glob", default="out_noekB_s*/pinn_ekman_results.json")
    p.add_argument("--noek-extra", default="")
    p.add_argument("--loglaw", default="baselines_extrap_h111/log_law/log_law_results.json")
    p.add_argument("--shear-glob", default=None,
                   help="results.json glob for the shear-constrained arm; omit to skip it")
    p.add_argument("--shear-label", default="Shear-constrained arm")
    p.add_argument("-o", "--out", default="fig_extrapolation_profiles")
    p.add_argument("--width", type=float, default=10.2)
    p.add_argument("--height", type=float, default=5.2)
    p.add_argument("--right-width", type=float, default=1.28)
    p.add_argument("--dir-xlim", type=float, default=98)
    main(p.parse_args())
