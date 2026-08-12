"""
Overall (aggregate) test RMSE bar chart and measured-vs-predicted speed
histograms with Jensen-Shannon divergence, in-distribution, for the three
architecture-matched arms.

Reads results.json (for overall_rmse, mean over seeds) and the prediction
.npz files (for the measured/predicted histograms and JS).

Usage:

    python3 fig_indist_overall.py \
        --arm "Ekman-constrained=out_pinn_indist_s*" \
        --arm "Shear-constrained=out_shear_indist_s*" \
        --arm "Physics-free=out_noek_indist_s*" \
        -o fig_indist_overall
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

COLORS = ["#1f6fb4", "#2a9d8f", "#d1495b", "#e9a23b"]
BINS = 60


def js_divergence(true_v, pred_v, bins=BINS):
    lo = float(min(np.nanmin(true_v), np.nanmin(pred_v)))
    hi = float(max(np.nanmax(true_v), np.nanmax(pred_v)))
    edges = np.linspace(lo, hi, bins + 1)
    pt, _ = np.histogram(true_v, bins=edges, density=True)
    pp, _ = np.histogram(pred_v, bins=edges, density=True)
    pt = pt / (pt.sum() + 1e-12); pp = pp / (pp.sum() + 1e-12)
    return float(jensenshannon(pt, pp, base=2) ** 2), pt, pp, edges


def main(a):
    arms = []
    for spec in a.arm:
        name, folder_patt = spec.split("=", 1)
        folders = sorted(glob.glob(folder_patt))
        if not folders:
            raise SystemExit(f"no folders for {name}: {folder_patt}")
        rmse = [json.load(open(os.path.join(f, "pinn_ekman_results.json")))["overall_rmse"]
                for f in folders]
        npz_paths = [os.path.join(f, "pinn_ekman_test_predictions.npz") for f in folders]
        arms.append((name, np.array(rmse), npz_paths))

    fig, axes = plt.subplots(1, 1 + len(arms), figsize=(4 + 3.3 * len(arms), 4.2))

    ax0 = axes[0]
    means = [r.mean() for _, r, _ in arms]
    sds = [r.std(ddof=1) if len(r) > 1 else 0 for _, r, _ in arms]
    x = np.arange(len(arms))
    ax0.bar(x, means, yerr=sds, color=COLORS[:len(arms)], capsize=4, width=0.6)
    for xi, m in zip(x, means):
        ax0.text(xi, m + 0.02, f"{m:.3f}", ha="center", fontsize=12.5)
    ax0.set_xticks(x); ax0.set_xticklabels([nm for nm, _, _ in arms], fontsize=12.5, rotation=15)
    ax0.set_ylabel("Overall speed RMSE (m/s)", fontsize=12.5)
    ax0.set_title("(a) Overall RMSE", fontsize=12.5)
    ax0.grid(axis="y", alpha=.3)
    ax0.set_ylim(top=max(m + s for m, s in zip(means, sds)) * 1.2)

    for i, (name, _, npz_paths) in enumerate(arms):
        ax = axes[i + 1]
        true_all, pred_all = [], []
        for p in npz_paths:
            d = np.load(p)
            m = d["valid_mask"].astype(bool)
            true_all.append(d["spd_true"][m]); pred_all.append(d["spd_pred"][m])
        true_all = np.concatenate(true_all); pred_all = np.concatenate(pred_all)
        js, pt, pp, edges = js_divergence(true_all, pred_all)
        centers = (edges[:-1] + edges[1:]) / 2
        w = edges[1] - edges[0]
        ax.bar(centers, pt, width=w, color="0.6", alpha=.6, label="Measured")
        ax.bar(centers, pp, width=w, color=COLORS[i % len(COLORS)], alpha=.6, label="Predicted")
        ax.set_title(f"({chr(98 + i)}) {name}\nJS = {js:.4f}", fontsize=12.5)
        ax.set_xlabel("Wind speed (m/s)", fontsize=12)
        if i == 0:
            ax.legend(fontsize=12.5)

    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=300)
    fig.savefig(a.out + ".pdf")
    print("written", a.out + ".png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True,
                   help="Name=folder glob (each folder has results.json and the predictions npz)")
    p.add_argument("-o", "--out", default="fig_indist_overall")
    main(p.parse_args())
