"""

LiDAR Wind Profile Clustering

Loads the .npz produced by lidar_parser.py and applies unsupervised
clustering to the vertical wind-speed (and optionally wind-direction)
profiles using three methods:

  1. K-Means          - fast, Euclidean distance in profile space
  2. Agglomerative    - hierarchical, Ward linkage
  3. DBSCAN           - density-based, detects outliers (label = -1)

Features used (one row per valid measurement, one column per altitude):
  - Normalised horizontal wind speed  [m/s]  at each height
  - Normalised wind direction (sin/cos encoded) at each height  [optional]

Outputs:

  cluster_results.npz   - cluster labels + PCA coordinates
  cluster_profiles.png  - mean wind-speed profile per cluster
  cluster_scatter.png   - 2-D PCA scatter coloured by cluster
  cluster_heatmap.png   - cluster frequency heatmap (hour vs. month)
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer

# CONFIGURATION  - edit before running

NPZ_FILE          = "lidar_dataset.npz"   # output from lidar_parser.py
OUTPUT_NPZ        = "cluster_results.npz"
FIGURE_PROFILES   = "cluster_profiles.png"
FIGURE_SCATTER    = "cluster_scatter.png"
FIGURE_HEATMAP    = "cluster_heatmap.png"

N_CLUSTERS        = 4     # K-Means / Agglomerative number of clusters
DBSCAN_EPS        = 0.8   # neighbourhood radius (in standardised space)
DBSCAN_MIN_SAMPLES= 10    # min points to form a core cluster

USE_WIND_DIR      = True  # include sin/cos-encoded wind direction as features
MIN_VALID_ALTS    = 6     # drop profiles with fewer than this many valid levels
N_PCA_COMPONENTS  = 2     # PCA components used for visualisation

CMAP              = "tab10"

# Hours considered 'day' for day/night split (local or UTC as stored)
DAY_HOURS        = list(range(6, 18))  # 06:00-17:59 day, rest night

# I/O

def load_npz(path):
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


# FEATURE ENGINEERING

def encode_direction(wd_deg):
    """Wind direction [deg] -> (sin, cos) to handle 0/360 wrap-around."""
    rad = np.deg2rad(wd_deg)
    return np.sin(rad), np.cos(rad)


def build_feature_matrix(data, use_wind_dir=USE_WIND_DIR,
                          min_valid=MIN_VALID_ALTS):
    """
    Assemble, filter, impute and standardise the feature matrix.

    Returns:
    
    X      : (N_valid, n_features)  standardised feature matrix
    mask   : (N,) bool  True for rows kept
    scaler : fitted StandardScaler
    """
    ws = data["wind_speed"]   # (N, N_alt)
    wd = data["wind_dir"]     # (N, N_alt)

    # Row filter: need enough valid altitudes
    valid_per_row = np.sum(~np.isnan(ws), axis=1)
    mask = valid_per_row >= min_valid
    ws_f = ws[mask]
    wd_f = wd[mask]

    # Build feature columns
    parts = [ws_f]
    if use_wind_dir:
        sin_wd, cos_wd = encode_direction(wd_f)
        parts += [sin_wd, cos_wd]

    X_raw = np.hstack(parts)

    # Impute residual NaN with column medians
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    # Standardise
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_imp)

    return X, mask, scaler


# CLUSTERING

def _scores(X, labels):
    """Return silhouette / Davies-Bouldin string."""
    valid = labels >= 0
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return "  sil=N/A  DB=N/A"
    n = min(5000, valid.sum())
    sil = silhouette_score(X[valid], labels[valid], sample_size=n)
    db  = davies_bouldin_score(X[valid], labels[valid])
    return f"  sil={sil:+.3f}  DB={db:.3f}"


def run_kmeans(X, k):
    print(f"  K-Means (k={k}) ...", end=" ", flush=True)
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    print("done" + _scores(X, labels))
    return labels


def run_agglomerative(X, k, max_samples=20_000, random_state=42):
    """
    Ward agglomerative on a random subsample, then assigns the remaining
    points to the nearest centroid (like K-Means predict).
    Full Ward on >~50k samples requires O(N^2) RAM — not feasible.
    """
    print(f"  Agglomerative (k={k}, Ward) ...", end=" ", flush=True)
    rng = np.random.default_rng(random_state)

    if len(X) > max_samples:
        idx_sub = rng.choice(len(X), size=max_samples, replace=False)
        idx_sub.sort()
        X_sub   = X[idx_sub]
    else:
        idx_sub = np.arange(len(X))
        X_sub   = X

    # Fit Ward on the subsample
    agg    = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_sub = agg.fit_predict(X_sub)

    if len(X) > max_samples:
        # Compute centroids from subsample labels
        centroids = np.vstack([
            X_sub[labels_sub == c].mean(axis=0) for c in range(k)
        ])
        # Assign all points to nearest centroid
        from sklearn.metrics import pairwise_distances_argmin
        labels = pairwise_distances_argmin(X, centroids)
        # Copy subsample labels back (exact)
        labels[idx_sub] = labels_sub
    else:
        labels = labels_sub

    print(f"done (subsample={len(X_sub):,})" + _scores(X, labels))
    return labels



# PLOTTING

def plot_profiles(ws_all, wd_all, mask, labels_dict, altitudes, out_path):
    """
    Mean +/- std wind-speed and mean wind-direction profiles per cluster.

    Layout: 2 x N_methods subplots
      - Top row    : wind speed (m/s) vs height
      - Bottom row : wind direction (deg) vs height
    """
    ws   = ws_all[mask]
    wd   = wd_all[mask] if wd_all is not None else None
    alts = altitudes.astype(float)
    n_m  = len(labels_dict)

    fig, axes = plt.subplots(2, n_m, figsize=(5 * n_m, 8), sharey=True)
    if n_m == 1:
        axes = axes.reshape(2, 1)

    for col, (name, labels) in enumerate(labels_dict.items()):
        unique = sorted(set(labels))
        cmap = matplotlib.colormaps[CMAP]
        colors = cmap(np.linspace(0, 1, max(len(unique), 2)))

        # ---- wind speed (top row) ----
        ax_s = axes[0, col]
        for k in unique:
            idx   = labels == k
            mean  = np.nanmean(ws[idx], axis=0)
            std   = np.nanstd(ws[idx],  axis=0)
            color = colors[unique.index(k)]
            ls    = "--" if k < 0 else "-"
            lbl   = (f"Cluster {k}  (n={idx.sum():,})" if k >= 0
                     else f"Outliers   (n={idx.sum():,})")
            ax_s.plot(mean, alts, color=color, lw=2, ls=ls, label=lbl)
            ax_s.fill_betweenx(alts, mean - std, mean + std,
                               color=color, alpha=0.15)
        ax_s.set_xlabel("Wind speed (m/s)", fontsize=12)
        ax_s.set_title(name + " — speed", fontsize=12, fontweight="bold")
        ax_s.legend(fontsize=12, loc="lower right")
        ax_s.grid(True, alpha=0.35)
        ax_s.set_xlim(left=0)

        # ---- wind direction (bottom row) ----
        ax_d = axes[1, col]
        if wd is not None:
            for k in unique:
                idx   = labels == k
                mean_dir = np.nanmean(wd[idx], axis=0)
                color = colors[unique.index(k)]
                ls    = "--" if k < 0 else "-"
                lbl   = (f"Cluster {k}  (n={idx.sum():,})" if k >= 0
                         else f"Outliers   (n={idx.sum():,})")
                ax_d.plot(mean_dir, alts, color=color, lw=1.5, ls=ls, label=lbl)
            ax_d.set_xlim(0, 360)
            ax_d.set_xlabel("Wind direction (deg)", fontsize=12)
            ax_d.set_title(name + " — direction", fontsize=12, fontweight="bold")
            ax_d.grid(True, alpha=0.35)
            ax_d.legend(fontsize=12, loc="lower right")
        else:
            ax_d.text(0.5, 0.5, "No wind direction data",
                      transform=ax_d.transAxes, ha="center", va="center")
            ax_d.axis("off")

    axes[0, 0].set_ylabel("Height (m)", fontsize=11)
    axes[1, 0].set_ylabel("Height (m)", fontsize=11)

    fig.suptitle("Mean cluster profiles: wind speed and direction",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_scatter(X_pca, labels_dict, pca_var, out_path):
    """PCA 2-D scatter, one subplot per method."""
    n_m = len(labels_dict)
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 4.5),
                             sharex=True, sharey=True)
    if n_m == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        unique = sorted(set(labels))
        colors = matplotlib.colormaps[CMAP](np.linspace(0, 1, max(len(unique), 2)))
        for k in unique:
            idx   = labels == k
            color = colors[unique.index(k)]
            mk    = "x" if k < 0 else "o"
            lbl   = f"C{k}" if k >= 0 else "noise"
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                       c=[color], marker=mk, s=12, alpha=0.5,
                       label=lbl, linewidths=0.4)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca_var[0]:.1%})", fontsize=12)
        ax.legend(fontsize=12, markerscale=1.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"PC2 ({pca_var[1]:.1%})", fontsize=12)
    fig.suptitle("PCA projection of wind profiles",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(timestamps_full, mask, labels_dict, out_path):
    """
    Cluster frequency heatmap: hour-of-day (y) x month (x).
    Reveals diurnal / seasonal patterns per cluster.
    """
    ts_valid = timestamps_full[mask].astype("datetime64[s]")
    ts_obj   = ts_valid.astype("M8[ms]").astype("O")
    hours    = np.array([t.hour  for t in ts_obj], dtype=int)
    months   = np.array([t.month for t in ts_obj], dtype=int)

    n_m     = len(labels_dict)
    n_cl_max = max(
        len([k for k in set(lb) if k >= 0]) for lb in labels_dict.values()
    )

    fig, axes = plt.subplots(n_cl_max, n_m,
                             figsize=(4.5 * n_m, 2.2 * n_cl_max),
                             squeeze=False)

    month_labels = ["J", "F", "M", "A", "M", "J",
                    "J", "A", "S", "O", "N", "D"]

    for col, (name, labels) in enumerate(labels_dict.items()):
        unique = sorted(k for k in set(labels) if k >= 0)
        for row, k in enumerate(unique):
            ax   = axes[row][col]
            idx  = labels == k
            heat = np.zeros((24, 12), dtype=float)
            for h in range(24):
                for mo in range(1, 13):
                    heat[h, mo - 1] = np.sum(
                        idx & (hours == h) & (months == mo)
                    )
            col_sum = heat.sum(axis=0, keepdims=True)
            col_sum[col_sum == 0] = 1
            ax.imshow(heat / col_sum, aspect="auto", origin="lower",
                      cmap="YlOrRd", vmin=0)
            ax.set_title(f"{name}  |  C{k}", fontsize=12)
            ax.set_xticks(range(12))
            ax.set_xticklabels(month_labels, fontsize=12)
            ax.set_yticks([0, 6, 12, 18, 23])
            ax.set_yticklabels(["00", "06", "12", "18", "23"], fontsize=12)
            if col == 0:
                ax.set_ylabel("Hour", fontsize=12)
            if row == n_cl_max - 1:
                ax.set_xlabel("Month", fontsize=12)
        for row in range(len(unique), n_cl_max):
            axes[row][col].axis("off")

    fig.suptitle("Cluster frequency  (normalised per month)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap_daynight(timestamps_full, mask, labels_dict, out_path):
    """
    Cluster frequency heatmap: day vs night (y) x month (x).
    Day is defined by DAY_HOURS; night is the complement.
    Each column (month) is normalised so day+night = 1.
    """
    ts_valid = timestamps_full[mask].astype("datetime64[s]")
    ts_obj   = ts_valid.astype("M8[ms]").astype("O")
    hours    = np.array([t.hour  for t in ts_obj], dtype=int)
    months   = np.array([t.month for t in ts_obj], dtype=int)

    n_m     = len(labels_dict)
    n_cl_max = max(
        len([k for k in set(lb) if k >= 0]) for lb in labels_dict.values()
    )

    fig, axes = plt.subplots(n_cl_max, n_m,
                             figsize=(4.5 * n_m, 2.2 * n_cl_max),
                             squeeze=False)

    month_labels = ["J", "F", "M", "A", "M", "J",
                    "J", "A", "S", "O", "N", "D"]

    day_mask_full = np.isin(hours, DAY_HOURS)

    for col, (name, labels) in enumerate(labels_dict.items()):
        unique = sorted(k for k in set(labels) if k >= 0)
        for row, k in enumerate(unique):
            ax   = axes[row][col]
            idx  = labels == k

            heat = np.zeros((2, 12), dtype=float)  # [day, night] x month
            for mo in range(1, 13):
                sel_month = (months == mo) & idx
                if not sel_month.any():
                    continue
                day_count   = np.sum(sel_month & day_mask_full)
                night_count = np.sum(sel_month & ~day_mask_full)
                col_sum = day_count + night_count
                if col_sum == 0:
                    continue
                heat[0, mo - 1] = day_count / col_sum
                heat[1, mo - 1] = night_count / col_sum

            im = ax.imshow(heat, aspect="auto", origin="lower",
                           cmap="YlGnBu", vmin=0, vmax=1)
            ax.set_title(f"{name}  |  C{k}", fontsize=8)
            ax.set_xticks(range(12))
            ax.set_xticklabels(month_labels, fontsize=7)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Day", "Night"], fontsize=7)
            if col == 0:
                ax.set_ylabel("Period", fontsize=8)
            if row == n_cl_max - 1:
                ax.set_xlabel("Month", fontsize=8)
        for row in range(len(unique), n_cl_max):
            axes[row][col].axis("off")

    fig.suptitle("Cluster frequency  (day vs night per month)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



# MAIN

def main(npz_file=NPZ_FILE, output_npz=OUTPUT_NPZ,
         n_clusters=N_CLUSTERS, dbscan_eps=DBSCAN_EPS,
         dbscan_min=DBSCAN_MIN_SAMPLES, use_wind_dir=USE_WIND_DIR,
         fig_profiles=FIGURE_PROFILES, fig_scatter=FIGURE_SCATTER,
         fig_heatmap=FIGURE_HEATMAP):

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LiDAR Wind Profile Clustering")
    print(f"  Input  : {os.path.abspath(npz_file)}")
    print(sep)

    # ---- load ----
    print("  Loading dataset ...")
    data  = load_npz(npz_file)
    alts  = data["altitudes"]
    N_tot = data["wind_speed"].shape[0]

    # Scrub any residual sentinel values (9999 / 9998) that survived the parser
    SENTINELS = [9999.0, 9998.0, -9999.0]
    TOL = 0.5
    for key in ("wind_speed", "wind_speed_min", "wind_speed_max",
                "wind_speed_std", "wind_dir", "wind_speed_vert", "TI"):
        if key in data:
            arr = data[key].astype(np.float64)
            for s in SENTINELS:
                arr[np.abs(arr - s) < TOL] = np.nan
            # Physical sanity bounds
            if "speed" in key and "vert" not in key:
                arr[(arr < 0) | (arr > 80)] = np.nan   # wind speed 0-80 m/s
            if "vert" in key:
                arr[np.abs(arr) > 20] = np.nan          # vertical ±20 m/s
            if "dir" in key:
                arr[(arr < 0) | (arr > 360)] = np.nan  # direction 0-360 deg
            data[key] = arr

    print(f"  Total records  : {N_tot:,}")
    print(f"  Altitudes ({len(alts)}) : {alts.tolist()} m")
    print(f"  Use wind dir   : {use_wind_dir}")

    # ---- features ----
    print("\n  Building feature matrix ...")
    X, mask, scaler = build_feature_matrix(data, use_wind_dir=use_wind_dir)
    print(f"  Valid profiles : {mask.sum():,} / {N_tot:,}")
    print(f"  Feature shape  : {X.shape}")

    # ---- PCA ----
    print("\n  PCA ...")
    pca   = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X)
    var   = pca.explained_variance_ratio_
    print(f"  PC1: {var[0]:.1%}   PC2: {var[1]:.1%}")

    # ---- clustering ----
    print("\n  Clustering ...")
    lkm = run_kmeans(X, n_clusters)
    lag = run_agglomerative(X, n_clusters)

    labels_dict = {
        f"K-Means (k={n_clusters})":       lkm,
        f"Agglomerative (k={n_clusters})": lag,
    }

    # ---- plots ----
    print("\n  Generating figures ...")
    plot_profiles(data["wind_speed"], data["wind_dir"], mask, labels_dict, alts, fig_profiles)
    plot_scatter(X_pca, labels_dict, var, fig_scatter)
    plot_heatmap(data["timestamps"], mask, labels_dict, fig_heatmap)
    plot_heatmap_daynight(data["timestamps"], mask, labels_dict,
                          fig_heatmap.replace(".png", "_daynight.png"))

    # ---- k-scan: internal indices vs k (K-Means only) ----
    print("\n  k-scan (K-Means only, internal indices vs k) ...")
    k_values = list(range(2, 9))
    metrics = []
    for kk in k_values:
        km = KMeans(n_clusters=kk, n_init=10, random_state=42)
        labels_k = km.fit_predict(X)
        valid = labels_k >= 0
        if valid.sum() < 2 or len(np.unique(labels_k[valid])) < 2:
            sil_k, db_k = np.nan, np.nan
        else:
            n_samp = min(5000, valid.sum())
            sil_k = silhouette_score(X[valid], labels_k[valid], sample_size=n_samp)
            db_k  = davies_bouldin_score(X[valid], labels_k[valid])
        print(f"    k={kk}: sil={sil_k:+.3f}  DB={db_k:.3f}")
        metrics.append((kk, sil_k, db_k))

    metrics = np.array(metrics, dtype=float)
    k_scan_csv = "k_scan_metrics.csv"
    metrics = np.array(metrics, dtype=float)
    k_scan_csv = "k_scan_metrics.csv"
    header = "k,silhouette,davies_bouldin"
    np.savetxt(k_scan_csv, metrics, delimiter=",", header=header, comments="")
    print(f"  Saved: {k_scan_csv}")

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.plot(k_values, metrics[:, 1], "o-", color="tab:blue", label="Silhouette")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Silhouette", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(k_values, metrics[:, 2], "s--", color="tab:red", label="Davies-Bouldin")
    ax2.set_ylabel("Davies-Bouldin", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Internal indices vs k  (K-Means)")
    fig.tight_layout()
    k_scan_png = "k_scan_metrics.png"
    fig.savefig(k_scan_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {k_scan_png}")

    # ---- save results ----
    print(f"  Saving {output_npz} ...")
    np.savez_compressed(
        output_npz,
        valid_mask        = mask,
        labels_kmeans     = lkm,
        labels_agglom     = lag,
        pca_coords        = X_pca,
        pca_explained_var = var,
        altitudes         = alts,
        timestamps_valid  = data["timestamps"][mask],
    )

    print(f"\n{sep}")
    print("  DONE")
    print(f"  Results : {os.path.abspath(output_npz)}")
    print(sep)


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster LiDAR vertical wind profiles from .npz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npz_file", nargs="?", default=NPZ_FILE,
                        help="Path to lidar_dataset.npz")
    parser.add_argument("-k", "--n-clusters", type=int, default=N_CLUSTERS,
                        help="Number of clusters (K-Means / Agglomerative)")
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS,
                        help="DBSCAN epsilon (standardised space)")
    parser.add_argument("--min-samples", type=int, default=DBSCAN_MIN_SAMPLES,
                        help="DBSCAN minimum cluster size")
    parser.add_argument("--no-wind-dir", action="store_true",
                        help="Use wind speed only (exclude direction)")
    parser.add_argument("-o", "--output", default=OUTPUT_NPZ,
                        help="Output .npz filename")
    args = parser.parse_args()

    main(
        npz_file    = args.npz_file,
        output_npz  = args.output,
        n_clusters  = args.n_clusters,
        dbscan_eps  = args.eps,
        dbscan_min  = args.min_samples,
        use_wind_dir= not args.no_wind_dir,
    )
