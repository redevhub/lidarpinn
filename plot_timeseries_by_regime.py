"""
Time-series prediction plots by regime

This script selects continuous time
windows that are representative of each K-Means regime (e.g. a nocturnal stable
stretch in C0, a convective stretch in C1/C2) and, for each window, plots the
measured and predicted wind speed (and optionally direction) of every model at a
reference height, as a function of time.

Because the stratified test split is not contiguous in calendar time, "windows"
are built as the longest available runs of consecutive test samples (in time
order) that share a regime label; the longest few per regime are plotted.

Outputs:

  fig_timeseries_C0.png, fig_timeseries_C1.png, fig_timeseries_C2.png
  timeseries_windows.json    (the chosen windows and their metrics)

Usage:

    python3 plot_timeseries_by_regime.py lidar_dataset.npz \
        --model "Log-law=baselines_fixed_np18/log_law" \
        --model "Power-law=baselines_fixed_np18/power_law" \
        --model "Constant=baselines_fixed_np18/constant" \
        --model "Random forest=baselines_fixed_np18/random_forest" \
        --model "Physics-free=tuning_output/best_n_past_18" \
        --model "Ekman=pinn_ekman_tuning_output/best_n_past_18" \
        --ref-height 100 --n-past 18 -o timeseries
"""

import os
import re
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def _unwrap_deg(a):
    """Unwrap a degree-valued angle series so 0/360 crossings do not draw
    spurious vertical jumps. NaNs are preserved (unwrapped over valid points)."""
    a = np.asarray(a, dtype=float)
    out = a.copy()
    mask = np.isfinite(a)
    if mask.sum() >= 2:
        out[mask] = np.degrees(np.unwrap(np.radians(a[mask])))
    return out

import lidar_pinn_ekman_v2_extrap as ek

RANDOM_STATE = 42
MIN_VALID_ALTS = 6
K_CLUSTERS = 3
# how many windows per regime to plot, and the minimum length (in samples) to qualify
N_WINDOWS_PER_REGIME = 1
MIN_WINDOW_LEN = 12


def reconstruct_test(npz_file, n_past, min_valid_alts, k_clusters):
    """Return, per test sample, the regime label, timestamp and a permutation
    that sorts the test set chronologically. Predictions are saved in the
    original test order (the order produced by the split), so the
    chronological permutation is returned to reorder them and all metadata together."""
    data = ek.load_npz(npz_file)
    labels = ek.build_kmeans_labels(data, k_clusters, min_valid_alts)
    ws_prof = data["wind_speed"].astype(float)
    ground = ek.build_ground_features_with_time(data)
    N = ground.shape[0]
    valid = np.sum(~np.isnan(ws_prof), axis=1) >= min_valid_alts
    ts = data["timestamps"]

    C_list, t_list = [], []
    for t in range(n_past - 1, N):
        if labels[t] < 0 or not valid[t]:
            continue
        seq = ground[t - n_past + 1: t + 1]
        if not np.all(np.isfinite(seq)):
            continue
        C_list.append(labels[t]); t_list.append(t)
    C = np.asarray(C_list, int); t_idx = np.asarray(t_list, int)
    M = C.shape[0]

    from sklearn.model_selection import train_test_split
    idx = np.arange(M)
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, stratify=C)
    va, te = train_test_split(tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=C[tmp])

    # te is the order in which predictions were saved (row i of preds == te[i]).
    te_time = ts[t_idx[te]]                 # timestamp of each saved-prediction row
    te_cluster = C[te]                      # regime of each saved-prediction row
    chrono = np.argsort(te_time)            # permutation -> chronological order
    return {"cluster_saved": te_cluster, "time_saved": te_time,
            "chrono": chrono, "alts": data["altitudes"].astype(float)}


def resolve_dir(root, n_past):
    if any(f.endswith("_results.json") for f in os.listdir(root)):
        return root
    for name in os.listdir(root):
        if re.fullmatch(rf"best_n_past_{n_past}", name):
            return os.path.join(root, name)
    cand = {int(re.fullmatch(r"best_n_past_(\d+)", n).group(1)): os.path.join(root, n)
            for n in os.listdir(root) if re.fullmatch(r"best_n_past_(\d+)", n)}
    if n_past in cand:
        return cand[n_past]
    raise FileNotFoundError(f"cannot resolve model dir under {root}")


def load_preds(path):
    preds = None
    for fn in os.listdir(path):
        if fn.endswith("_test_predictions.npz"):
            preds = dict(np.load(os.path.join(path, fn), allow_pickle=False))
    if preds is None:
        raise FileNotFoundError(f"no predictions in {path}")
    return preds


def find_windows(cluster, time, regime, n_windows, min_len):
    """Longest runs of consecutive test samples (in chronological order) whose
    timestamps are ~evenly spaced and that share the given regime label."""
    is_r = cluster == regime
    windows, start = [], None
    for i in range(len(is_r)):
        if is_r[i] and start is None:
            start = i
        if (not is_r[i] or i == len(is_r) - 1) and start is not None:
            end = i if not is_r[i] else i + 1
            if end - start >= min_len:
                windows.append((start, end))
            start = None
    windows.sort(key=lambda w: w[1] - w[0], reverse=True)
    return windows[:n_windows]


def main(npz_file, models, ref_height, n_past, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = reconstruct_test(npz_file, n_past, MIN_VALID_ALTS, K_CLUSTERS)
    alts = meta["alts"]
    hcol = int(np.argmin(np.abs(alts - ref_height)))
    chrono = meta["chrono"]
    cluster = meta["cluster_saved"][chrono]       # chronological
    time = meta["time_saved"][chrono]
    print(f"  reference height: {alts[hcol]} m (column {hcol})")
    print(f"  test samples: {cluster.size:,}")

    # load every model's predictions, reorder them chronologically
    loaded = {}
    n_expected = cluster.size
    for label, root in models:
        p = load_preds(resolve_dir(root, n_past))
        if p["spd_pred"].shape[0] != n_expected:
            raise SystemExit(
                f"{label}: predictions have {p['spd_pred'].shape[0]} rows but the "
                f"reconstructed test set has {n_expected}. n_past must match.")
        entry = {"spd": p["spd_pred"][chrono]}
        if "dir_pred" in p:
            entry["dir"] = p["dir_pred"][chrono]
        loaded[label] = entry
    # ground truth from the first model (shared)
    first_root = dict(models)[next(iter(models))[0]]
    ptrue = load_preds(resolve_dir(first_root, n_past))
    spd_true = ptrue["spd_true"][chrono]
    dir_true = ptrue["dir_true"][chrono] if "dir_true" in ptrue else None

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loaded), 3)))
    regime_name = {0: "C0 (nocturnal stable)", 1: "C1 (convective)", 2: "C2 (convective)"}
    chosen = {}

    for regime in sorted(set(cluster.tolist())):
        wins = find_windows(cluster, time, regime, N_WINDOWS_PER_REGIME, MIN_WINDOW_LEN)
        if not wins:
            print(f"  regime C{regime}: no run >= {MIN_WINDOW_LEN} samples"); continue
        s, e = wins[0]
        t_axis = time[s:e]
        chosen[f"C{regime}"] = {"n": int(e - s), "start": str(t_axis[0]),
                                "end": str(t_axis[-1])}

        has_dir = all("dir" in mc for mc in loaded.values()) and dir_true is not None
        nrows = 2 if has_dir else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(9, 3 * nrows + 0.5),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]
        xr = range(e - s)
        axes[0].plot(xr, spd_true[s:e, hcol], "k-", lw=2.4, label="Measured", zorder=10)
        for (label, mc), col in zip(loaded.items(), colors):
            axes[0].plot(xr, mc["spd"][s:e, hcol], "--", color=col, lw=1.2, label=label)
        axes[0].set(ylabel=f"Speed at {int(alts[hcol])} m (m/s)",
                    title=f"{regime_name.get(regime, f'C{regime}')} — "
                          f"{str(t_axis[0])[:16]} to {str(t_axis[-1])[:16]}")
        axes[0].grid(alpha=.3); axes[0].legend(fontsize=12, ncol=3)
        if has_dir:
            axes[1].plot(xr, _unwrap_deg(dir_true[s:e, hcol]), "k-", lw=2.4, label="Measured", zorder=10)
            for (label, mc), col in zip(loaded.items(), colors):
                axes[1].plot(xr, _unwrap_deg(mc["dir"][s:e, hcol]), "--", color=col, lw=1.2, label=label)
            axes[1].set(ylabel=f"Direction at {int(alts[hcol])} m (deg)")
            axes[1].grid(alpha=.3)
        axes[-1].set_xlabel("Consecutive test samples in the window")
        fig.tight_layout()
        fn = os.path.join(out_dir, f"fig_timeseries_C{regime}.png")
        fig.savefig(fn, dpi=160); plt.close(fig)
        print(f"  regime C{regime}: window of {e-s} samples -> {fn}")

    json.dump({"ref_height": float(alts[hcol]), "windows": chosen},
              open(os.path.join(out_dir, "timeseries_windows.json"), "w"), indent=2)
    print(f"\n  Outputs in {os.path.abspath(out_dir)}")


def parse_model(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--model must be 'Label=path', got: {s}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Time-series prediction plots by regime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz_file", nargs="?", default="lidar_dataset.npz")
    ap.add_argument("--model", action="append", type=parse_model, required=True,
                    metavar="LABEL=PATH")
    ap.add_argument("--ref-height", type=float, default=100,
                    help="reference height (m) for the time series")
    ap.add_argument("--n-past", type=int, default=18)
    ap.add_argument("-o", "--output-dir", default="timeseries")
    args = ap.parse_args()
    main(args.npz_file, args.model, args.ref_height, args.n_past, args.output_dir)
