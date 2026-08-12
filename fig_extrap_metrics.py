"""
Figures for the held-out-gate metrics beyond pointwise error.

Three panels: (a) Jensen-Shannon divergence, (b) profile shape statistics
normalised by the measured value, (c) direction error by K-Means regime.
Series: Ekman-informed arm, physics-free arm, and, optionally, a
shear-constrained arm.

Usage:

    python3 fig_extrap_metrics.py \
        --pinn-glob "out_pinnB_s*/pinn_ekman_test_predictions.npz" \
        --noek-glob "out_noekB_s*/pinn_ekman_test_predictions.npz" \
        --shear-glob "out_shear_s*/pinn_ekman_test_predictions.npz" \
        --regime-json extrap_by_regime_B/extrap_by_regime.json \
        -o fig_extrap_metrics
"""
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

C_PINN, C_NOEK, C_SHEAR, C_MEAS = "#1f6fb4", "#d1495b", "#2a9d8f", "0.35"
BINS = 60


def unwrap_deg(a, axis=-1):
    return np.degrees(np.unwrap(np.radians(a), axis=axis))


def _unseen_mask(d):
    seen = d["seen_mask"].astype(bool)
    band = (d["band_mask"].astype(bool) if "band_mask" in d.files
            else np.zeros_like(seen))
    return ~seen & ~band


def js_divergence(true_v, pred_v, bins=BINS):
    lo = float(min(np.nanmin(true_v), np.nanmin(pred_v)))
    hi = float(max(np.nanmax(true_v), np.nanmax(pred_v)))
    edges = np.linspace(lo, hi, bins + 1)
    pt, _ = np.histogram(true_v, bins=edges, density=True)
    pp, _ = np.histogram(pred_v, bins=edges, density=True)
    pt = pt / (pt.sum() + 1e-12); pp = pp / (pp.sum() + 1e-12)
    return float(jensenshannon(pt, pp, base=2) ** 2)


def arm_js(paths):
    out = []
    for p in paths:
        d = np.load(p)
        m = d["valid_mask"].astype(bool) & _unseen_mask(d)[None, :]
        out.append(js_divergence(d["spd_true"][m], d["spd_pred"][m]))
    return np.array(out)


def shape_stats(spd, dirp, z):
    dU = np.diff(spd, axis=1) / np.diff(z)[None, :]
    du = unwrap_deg(dirp, axis=1)
    turn = du[:, -1] - du[:, 0]
    return {"roughness_speed": float(np.mean(np.abs(np.diff(spd, n=2, axis=1))) / np.mean(spd)),
            "roughness_dir": float(np.mean(np.abs(np.diff(du, n=2, axis=1)))),
            "turning_sd": float(np.std(turn))}


def arm_shapes(paths):
    stats, meas = [], None
    for p in paths:
        d = np.load(p)
        order = np.argsort(d["altitudes"]); z = d["altitudes"][order]
        sel = _unseen_mask(d)[order]
        vm = d["valid_mask"][:, order].astype(bool)
        ok = vm[:, sel].all(axis=1)
        sp = d["spd_pred"][:, order][ok][:, sel]
        dp = d["dir_pred"][:, order][ok][:, sel]
        stats.append(shape_stats(sp, dp, z[sel]))
        if meas is None:
            st = d["spd_true"][:, order][ok][:, sel]
            dt = d["dir_true"][:, order][ok][:, sel]
            meas = shape_stats(st, dt, z[sel])
    return stats, meas


def main(a):
    pinn_paths = sorted(glob.glob(a.pinn_glob))
    noek_paths = sorted(glob.glob(a.noek_glob))
    shear_paths = sorted(glob.glob(a.shear_glob)) if a.shear_glob else []
    have_shear = len(shear_paths) > 0
    if not pinn_paths or not noek_paths:
        raise SystemExit("no prediction files found; check the globs")

    arms = [("Ekman", pinn_paths, C_PINN)]
    if have_shear:
        arms.append(("Shear", shear_paths, C_SHEAR))
    arms.append(("Physics-free", noek_paths, C_NOEK))
    n = len(arms)

    fig = plt.figure(figsize=(13 + 2.2 * (n - 2), 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8 + 0.15*(n-2), 1.5, 1.5], wspace=0.32)

    # (a) JS divergence
    ax = fig.add_subplot(gs[0, 0])
    js_vals = [arm_js(paths) for _, paths, _ in arms]
    x = np.arange(n)
    ax.bar(x, [j.mean() for j in js_vals],
           yerr=[j.std(ddof=1) if len(j) > 1 else 0 for j in js_vals],
           color=[c for _, _, c in arms], capsize=4, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels([nm for nm, _, _ in arms], fontsize=12.5)
    ax.set_ylabel("Jensen-Shannon divergence", fontsize=12.5)
    ax.set_title("(a) Speed distribution", fontsize=12.5)
    ax.grid(axis="y", alpha=.3)
    ax.set_ylim(top=max(j.mean() + (j.std(ddof=1) if len(j)>1 else 0) for j in js_vals) * 1.3)

    # (b) profile shape
    ax = fig.add_subplot(gs[0, 1])
    keys = ["roughness_speed", "roughness_dir", "turning_sd"]
    labels = ["Speed\ncurvature", "Direction\njaggedness", "Turning\nspread"]
    x = np.arange(len(keys))
    w = 0.8 / n
    meas = None
    for i, (name, paths, color) in enumerate(arms):
        stats, m = arm_shapes(paths)
        if meas is None:
            meas = m
        vm = [np.mean([s[k] for s in stats]) / meas[k] for k in keys]
        vs = [np.std([s[k] for s in stats], ddof=1) / meas[k] for k in keys]
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, vm, w, yerr=vs, color=color, capsize=3, label=name)
    ax.axhline(1.0, color=C_MEAS, ls="--", lw=1.4, label="Measured")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("value / measured value", fontsize=12.5)
    ax.set_title("(b) Profile shape (1.0 = measured)", fontsize=12.5)
    ax.legend(fontsize=12.8); ax.grid(axis="y", alpha=.3)

    # (c) error by regime
    ax = fig.add_subplot(gs[0, 2])
    if a.regime_json:
        reg = json.load(open(a.regime_json))
        names = sorted(reg.keys())
        x = np.arange(len(names))
        # discover which arm prefixes are actually present in the json
        # (Ekman/Physics-free always; shear_label only if that arm was included)
        prefixes = [("pinn", "Ekman", C_PINN)]
        if any(f"{a.shear_label}_dir_mean" in reg[nm] for nm in names):
            prefixes.append((a.shear_label, "Shear", C_SHEAR))
        prefixes.append(("noek", "Physics-free", C_NOEK))
        m = len(prefixes)
        w2 = 0.8 / m
        for i, (key, label, color) in enumerate(prefixes):
            means = [reg[nm].get(f"{key}_dir_mean", np.nan) for nm in names]
            sds   = [reg[nm].get(f"{key}_dir_sd", 0.0) for nm in names]
            off = (i - (m - 1) / 2) * w2
            ax.bar(x + off, means, w2, yerr=sds, color=color, capsize=3, label=label)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12.5)
        ax.set_ylabel("Direction MAE (deg)", fontsize=12.5)
        ax.set_title("(c) Direction error by regime", fontsize=12.5)
        ax.legend(fontsize=12.8); ax.grid(axis="y", alpha=.3)
    else:
        ax.text(0.5, 0.5, "pass --regime-json", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("written", a.out + ".png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pinn-glob", default="out_pinnB_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--noek-glob", default="out_noekB_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--shear-glob", default=None)
    p.add_argument("--shear-label", default="shear-constrained")
    p.add_argument("--regime-json", default=None)
    p.add_argument("-o", "--out", default="fig_extrap_metrics")
    main(p.parse_args())
