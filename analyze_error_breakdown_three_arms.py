"""
Temporal/regime breakdown of the test error, three architecture-matched arms
(Ekman-constrained, shear-constrained, physics-free) shown together.

The stratified test partition is reconstructed (same K-Means labels, same random_state, same n_past
filtering) to recover each test sample's timestamp and cluster label, then
aligned with each arm's saved predictions. The count is verified before use.

Usage:

    python3 analyze_error_breakdown_three_arms.py lidar_dataset.npz \
        --arm "Ekman-constrained=out_pinn_indist_s42" \
        --arm "Shear-constrained=out_shear_indist_s42" \
        --arm "Physics-free=out_noek_indist_s42" \
        --n-past 18 -o error_breakdown_three_arms
"""
import os, json, argparse
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
import lidar_pinn_ekman_v2_extrap as ek

RANDOM_STATE = 42
MIN_VALID_ALTS = 6
K_CLUSTERS = 3

SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_NAME = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}
DAY_START, DAY_END = 7, 19
COLORS = {"ekman": "#1f6fb4", "shear": "#2a9d8f", "none": "#d1495b"}


def reconstruct_test_time_cluster(npz_file, n_past, min_valid_alts, k_clusters):
    data = ek.load_npz(npz_file)
    labels = ek.build_kmeans_labels(data, k_clusters, min_valid_alts)
    ws_prof = data["wind_speed"].astype(float)
    ground = ek.build_ground_features_with_time(data)
    N, F = ground.shape
    valid = np.sum(~np.isnan(ws_prof), axis=1) >= min_valid_alts
    ts = data["timestamps"]
    hours = ts.astype("datetime64[h]").astype(int) % 24
    months = (ts.astype("datetime64[M]").astype(int) % 12) + 1

    C_list, t_list = [], []
    for t in range(n_past - 1, N):
        if labels[t] < 0 or not valid[t]:
            continue
        seq = ground[t - n_past + 1: t + 1]
        if not np.all(np.isfinite(seq)):
            continue
        C_list.append(labels[t]); t_list.append(t)
    C = np.asarray(C_list, dtype=int)
    t_idx = np.asarray(t_list, dtype=int)
    M = C.shape[0]

    from sklearn.model_selection import train_test_split
    idx = np.arange(M)
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, stratify=C)
    va, te = train_test_split(tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=C[tmp])
    te_t = t_idx[te]
    return {"hour": hours[te_t], "month": months[te_t], "cluster": C[te], "n_test": M}


def load_predictions(pred_dir):
    p = os.path.join(pred_dir, "pinn_ekman_test_predictions.npz")
    d = np.load(p, allow_pickle=False)
    spd_pred = d["spd_pred"]; spd_true = d["spd_true"]
    err = spd_pred - spd_true
    per_sample_rmse = np.sqrt(np.nanmean(err ** 2, axis=1))
    per_sample_mae = np.nanmean(np.abs(err), axis=1)
    has_dir = "dir_pred" in d.files and "dir_true" in d.files
    per_sample_dirmae = None
    if has_dir:
        de = np.abs(((d["dir_pred"] - d["dir_true"] + 180) % 360) - 180)
        per_sample_dirmae = np.nanmean(de, axis=1)
    return per_sample_rmse, per_sample_mae, per_sample_dirmae


def grouped_stats(values, keys, order):
    out = {}
    for k in order:
        m = keys == k
        if m.sum() > 0:
            out[k] = {"mean": float(np.nanmean(values[m])), "n": int(m.sum())}
    return out


def main(npz_file, arm_specs, n_past, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = reconstruct_test_time_cluster(npz_file, n_past, MIN_VALID_ALTS, K_CLUSTERS)
    hour, month, cluster = meta["hour"], meta["month"], meta["cluster"]
    season = np.array([SEASON[int(m)] for m in month])
    is_day = (hour >= DAY_START) & (hour < DAY_END)

    arms = {}
    for spec in arm_specs:
        name, pred_dir = spec.split("=", 1)
        rmse, mae, dirmae = load_predictions(pred_dir)
        if rmse.shape[0] != cluster.shape[0]:
            raise SystemExit(f"Alignment check FAILED for {name}: reconstructed "
                             f"test has {cluster.shape[0]} samples but predictions "
                             f"have {rmse.shape[0]}.")
        arms[name] = dict(rmse=rmse, mae=mae, dirmae=dirmae)
    print(f"  aligned {cluster.shape[0]:,} test samples for {len(arms)} arms")

    color_cycle = list(COLORS.values())
    summary = {"n_test": int(cluster.shape[0]), "n_past": n_past}

    # error by hour of day + season (combined)
    hrs = sorted(set(hour.tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ax = axes[0]
    for i, (name, d) in enumerate(arms.items()):
        rmse_by_hour = [float(np.nanmean(d["rmse"][hour == h])) for h in hrs]
        ax.plot(hrs, rmse_by_hour, "o-", ms=3.5, color=color_cycle[i % len(color_cycle)],
                label=name)
    ax.set(xlabel="Local hour", ylabel="Speed RMSE (m/s)", title="(a) Error by hour of day")
    ax.grid(alpha=.3); ax.legend(fontsize=12)

    ax = axes[1]
    labs = SEASON_ORDER
    x = np.arange(len(labs)); w = 0.8 / len(arms)
    summary["by_season"] = {}
    for i, (name, d) in enumerate(arms.items()):
        seas = grouped_stats(d["rmse"], season, SEASON_ORDER)
        vals = [seas[s]["mean"] if s in seas else np.nan for s in labs]
        off = (i - (len(arms) - 1) / 2) * w
        ax.bar(x + off, vals, w, color=color_cycle[i % len(color_cycle)], label=name)
        summary["by_season"][name] = seas
    ax.set_xticks(x); ax.set_xticklabels([SEASON_NAME[s] for s in labs])
    ax.set(ylabel="Speed RMSE (m/s)", title="(b) Error by season")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.95, 1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_err_by_hour_season.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # error by cluster/regime
    cl_ids = sorted(set(cluster.tolist()))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(cl_ids)); w = 0.8 / len(arms)
    summary["by_cluster"] = {}
    for i, (name, d) in enumerate(arms.items()):
        vals, ns = [], []
        for c in cl_ids:
            m = cluster == c
            vals.append(float(np.nanmean(d["rmse"][m])))
            ns.append(int(m.sum()))
        off = (i - (len(arms) - 1) / 2) * w
        ax.bar(x + off, vals, w, color=color_cycle[i % len(color_cycle)], label=name)
        summary["by_cluster"][name] = {f"C{c}": {"rmse": v, "n": n}
                                       for c, v, n in zip(cl_ids, vals, ns)}
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in cl_ids])
    ax.set(xlabel="K-Means regime", ylabel="Speed RMSE (m/s)",
           title="Test error by regime (cluster)")
    all_vals = [v for arm in summary["by_cluster"].values() for c in arm.values() for v in [c["rmse"]]]
    ax.set_ylim(top=max(all_vals) * 1.25)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_err_by_cluster.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    json.dump(summary, open(os.path.join(out_dir, "error_breakdown_summary.json"), "w"),
              indent=2)
    print(f"\n  Outputs in {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz_file", nargs="?", default="lidar_dataset.npz")
    ap.add_argument("--arm", action="append", required=True,
                    help="Name=pred_dir (folder containing pinn_ekman_test_predictions.npz), repeatable")
    ap.add_argument("--n-past", type=int, default=18)
    ap.add_argument("-o", "--output-dir", default="error_breakdown_three_arms")
    args = ap.parse_args()
    main(args.npz_file, args.arm, args.n_past, args.output_dir)
