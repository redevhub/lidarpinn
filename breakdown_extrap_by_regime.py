"""
Regime breakdown of the vertical-extrapolation error.

Splits the held-out-gate error by K-Means regime (C0 nocturnal stable,
C1/C2 convective), for two or three arms at once.

Usage:

    python3 breakdown_extrap_by_regime.py lidar_dataset.npz \
        --pinn-glob "out_pinn_v2_s*/pinn_ekman_test_predictions.npz" \
        --noek-glob "out_noek_v2_s*/pinn_ekman_test_predictions.npz" \
        --shear-glob "test_b4_s*/pinn_ekman_test_predictions.npz" \
        --shear-label shear \
        --n-past 18 -o extrap_by_regime
"""
import os, json, glob, argparse
import numpy as np
import lidar_pinn_ekman_v2_extrap as ek

RANDOM_STATE = 42
MIN_VALID_ALTS = 6
K_CLUSTERS = 3
REGIME_NAME = {0: "C0 (nocturnal stable)", 1: "C1 (convective)", 2: "C2 (convective)"}


def reconstruct_clusters(npz_file, n_past):
    data = ek.load_npz(npz_file)
    labels = ek.build_kmeans_labels(data, K_CLUSTERS, MIN_VALID_ALTS)
    ws = data["wind_speed"].astype(float)
    ground = ek.build_ground_features_with_time(data)
    N = ground.shape[0]
    valid = np.sum(~np.isnan(ws), axis=1) >= MIN_VALID_ALTS
    C_list = []
    for t in range(n_past - 1, N):
        if labels[t] < 0 or not valid[t]:
            continue
        if not np.all(np.isfinite(ground[t - n_past + 1: t + 1])):
            continue
        C_list.append(labels[t])
    C = np.asarray(C_list, dtype=int)
    from sklearn.model_selection import train_test_split
    idx = np.arange(C.shape[0])
    _, tmp = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, stratify=C)
    _, te = train_test_split(tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=C[tmp])
    return C[te]


def arm_errors_by_regime(paths, cluster):
    out = {}
    for p in paths:
        d = np.load(p)
        seen = d["seen_mask"].astype(bool)
        band = d["band_mask"].astype(bool) if "band_mask" in d.files else np.zeros_like(seen)
        unseen = ~seen & ~band
        vm = d["valid_mask"].astype(bool) & unseen[None, :]
        if d["spd_pred"].shape[0] != cluster.shape[0]:
            raise SystemExit(f"{p}: {d['spd_pred'].shape[0]} rows vs {cluster.shape[0]} "
                             f"reconstructed samples - n_past must match.")
        es = d["spd_pred"] - d["spd_true"]
        ed = np.abs(((d["dir_pred"] - d["dir_true"] + 180) % 360) - 180)
        for r in sorted(set(cluster.tolist())):
            sel = vm & (cluster == r)[:, None]
            if sel.sum() == 0:
                continue
            rmse = float(np.sqrt(np.mean(es[sel] ** 2)))
            dmae = float(np.mean(ed[sel]))
            out.setdefault(r, {"rmse": [], "dir": [], "n": []})
            out[r]["rmse"].append(rmse)
            out[r]["dir"].append(dmae)
            out[r]["n"].append(int(sel.sum()))
    return out


def summarise(errs, prefix):
    """{regime: {f'{prefix}_rmse_mean': ..., ...}}"""
    out = {}
    for r, d in errs.items():
        rmse = np.array(d["rmse"]); dire = np.array(d["dir"])
        out[r] = {
            f"{prefix}_rmse_mean": float(rmse.mean()),
            f"{prefix}_rmse_sd": float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0,
            f"{prefix}_dir_mean": float(dire.mean()),
            f"{prefix}_dir_sd": float(dire.std(ddof=1)) if len(dire) > 1 else 0.0,
            "n_entries": int(np.mean(d["n"])),
        }
    return out


def main(a):
    cluster = reconstruct_clusters(a.npz_file, a.n_past)
    print(f"  reconstructed test samples: {cluster.size:,}")

    arms = [("pinn", sorted(glob.glob(a.pinn_glob))),
            ("noek", sorted(glob.glob(a.noek_glob)))]
    if a.shear_glob:
        arms.append((a.shear_label, sorted(glob.glob(a.shear_glob))))

    per_arm = {name: arm_errors_by_regime(paths, cluster) for name, paths in arms}

    all_regimes = sorted(set().union(*[set(e.keys()) for e in per_arm.values()]))
    header = f"  {'regime':<24}" + "".join(f"{name+' RMSE':>16}{name+' dir':>14}" for name, _ in arms)
    print(f"\n  HELD-OUT GATES, BY REGIME (mean +- sd over seeds)\n")
    print(header)
    print("  " + "-" * (24 + 30 * len(arms)))

    summary = {}
    for r in all_regimes:
        row = f"  {REGIME_NAME.get(r, f'C{r}'):<24}"
        for name, _ in arms:
            d = per_arm[name].get(r)
            if d is None:
                row += f"{'--':>16}{'--':>14}"
                continue
            rmse = np.array(d["rmse"]); dire = np.array(d["dir"])
            row += (f"{rmse.mean():7.3f}+-{rmse.std(ddof=1) if len(rmse)>1 else 0:<6.3f}"
                    f"{dire.mean():6.1f}+-{dire.std(ddof=1) if len(dire)>1 else 0:<6.1f}")
        print(row)
        entry = {}
        for name, _ in arms:
            d = per_arm[name].get(r)
            if d is not None:
                entry.update(summarise({r: d}, name)[r])
        summary[f"C{r}"] = entry

    os.makedirs(a.output_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(a.output_dir, "extrap_by_regime.json"), "w"), indent=2)
    print(f"\n  Written to {os.path.abspath(a.output_dir)}/extrap_by_regime.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz_file", nargs="?", default="lidar_dataset.npz")
    p.add_argument("--pinn-glob", default="out_pinn_v2_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--noek-glob", default="out_noek_v2_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--shear-glob", default=None,
                   help="optional third arm, e.g. the shear-constrained arm")
    p.add_argument("--shear-label", default="shear")
    p.add_argument("--n-past", type=int, default=18)
    p.add_argument("-o", "--output-dir", default="extrap_by_regime")
    main(p.parse_args())
