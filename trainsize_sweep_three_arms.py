"""
Training-set size sweep, three architecture-matched arms (Ekman, shear
constrained, physics-free).

The architecture (lstm1, lstm2,
dec_units, context_dim) is fixed to the one selected for the Ekman arm across
all three types.

Full-data reference points are read from the four-seed in-distribution runs (out_pinn_indist_s42, out_shear_indist_s42,
out_noek_indist_s42).

Usage:

    python3 trainsize_sweep_three_arms.py lidar_dataset.npz \
        --physics-type ekman  --full-model-dir out_pinn_indist_s42  -o trainsize_ekman
    python3 trainsize_sweep_three_arms.py lidar_dataset.npz \
        --physics-type shear  --full-model-dir out_shear_indist_s42 -o trainsize_shear
    python3 trainsize_sweep_three_arms.py lidar_dataset.npz \
        --physics-type none   --full-model-dir out_noek_indist_s42  -o trainsize_noek
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lidar_pinn_ekman_v2_extrap as ek
from physics_shear_powerlaw import shear_residual

RANDOM_STATE = 42
BATCH_SIZE = 128

# Fixed architecture, shared by all three arms (Table hyperparams).
FIXED_CFG = dict(lstm1_units=128, lstm2_units=128, dec_units=64,
                 context_dim=64, learning_rate=5e-4, batch_size=128,
                 w_ekman=9.160265674307145e-4, w_bc=0.05, a_cls=0.2)


def build_everything(npz_file, n_past, min_valid_alts, k_clusters):
    data = ek.load_npz(npz_file)
    labels = ek.build_kmeans_labels(data, k_clusters, min_valid_alts)
    X, U, V, C, alts, UV_met = ek.build_sequences(
        data, labels, n_past, min_valid_alts, return_extras=True)
    ws_prof = data["wind_speed"].astype(float)
    ground = ek.build_ground_features_with_time(data)
    N = ground.shape[0]
    valid = np.sum(~np.isnan(ws_prof), axis=1) >= min_valid_alts
    t_list = []
    for t in range(n_past - 1, N):
        if labels[t] < 0 or not valid[t]:
            continue
        if not np.all(np.isfinite(ground[t - n_past + 1: t + 1])):
            continue
        t_list.append(t)
    t_idx = np.asarray(t_list, dtype=int)
    times = data["timestamps"][t_idx]
    M, T_steps, F = X.shape
    spd = np.sqrt(U**2 + V**2)
    bad = (spd < ek.WS_MIN) | (spd > ek.WS_MAX)
    if bad.any():
        U[bad] = np.nan; V[bad] = np.nan
    for arr in (U, V):
        if np.isnan(arr).any():
            med = np.nanmedian(arr, axis=0)
            r, cc = np.where(np.isnan(arr)); arr[r, cc] = med[cc]
    uv = np.stack([U, V], axis=-1).astype(np.float32)
    uv_scale = float(np.std(uv) + 1e-6)
    Z_data = np.tile(alts.astype(np.float32), (M, 1))
    Xf = X.reshape(M * T_steps, F)
    fin = np.all(np.isfinite(Xf), axis=1)
    scaler = StandardScaler().fit(Xf[fin])
    Xf_s = Xf.copy(); Xf_s[fin] = scaler.transform(Xf[fin])
    X_s = Xf_s.reshape(M, T_steps, F).astype(np.float32)
    idx = np.arange(M)
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, stratify=C)
    va, te = train_test_split(tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=C[tmp])
    return dict(X_s=X_s, uv=uv, Z=Z_data, C=C, alts=alts, F=F,
               uv_scale=uv_scale, uv_met=UV_met, tr=tr, va=va, te=te, times=times)


def subsample_train(tr, times, scheme, size):
    t_tr = times[tr]
    if scheme == "contiguous":
        months = t_tr.astype("datetime64[M]")
        uniq = np.unique(months)
        mask = np.isin(months, uniq[:size])
    elif scheme == "distributed":
        dom = (t_tr.astype("datetime64[D]") - t_tr.astype("datetime64[M]")).astype(int) + 1
        mask = dom <= size
    else:
        raise ValueError(scheme)
    return tr[mask]


def train_one(d, physics_type, tr_sub, lat, epochs, w_ekman_warmup, z_met, out_run):
    """Train one arm on a training subset, with the fixed architecture.
    physics_type: 'ekman', 'shear', or 'none'."""
    os.makedirs(out_run, exist_ok=True)
    tf.random.set_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    f_cor = 2 * ek.OMEGA * np.sin(np.deg2rad(lat))
    alts = d["alts"]; uv_scale = d["uv_scale"]

    def take(ix):
        return (tf.constant(d["X_s"][ix]), tf.constant(d["Z"][ix]),
                tf.constant(d["uv"][ix]), tf.constant(d["C"][ix]),
                tf.constant(d["uv_met"][ix]))
    Xtr, Ztr, UVtr, Ctr, UVMtr = take(tr_sub)
    Xva, Zva, UVva, Cva, UVMva = take(d["va"])
    Xte, Zte, UVte, Cte, UVMte = take(d["te"])

    model = ek.EkmanMOSTPINN(
        FIXED_CFG["lstm1_units"], d["F"], 3,
        lstm1=FIXED_CFG["lstm1_units"], lstm2=FIXED_CFG["lstm2_units"],
        dec_units=FIXED_CFG["dec_units"], context_dim=FIXED_CFG["context_dim"])
    model.adapt_height(alts)
    if physics_type == "shear" and not hasattr(model, "alpha_out"):
        raise SystemExit(
            "This EkmanMOSTPINN instance has no alpha_out layer; use the "
            "shear-capable lidar_pinn_ekman_v2_extrap.py from this project, "
            "with the alpha_out addition wired for the shear-constrained arm.")
    _ = model((Xtr[:2], Ztr[:2], UVMtr[:2]))

    w_ekman = FIXED_CFG["w_ekman"] if physics_type in ("ekman", "shear") else 0.0
    trainer = ek.PINNTrainer(
        model, keras.optimizers.Adam(FIXED_CFG["learning_rate"]),
        w_ekman, FIXED_CFG["w_bc"], FIXED_CFG["a_cls"], f_cor,
        float(min(alts)), float(max(alts)), uv_scale,
        w_ekman_warmup=w_ekman_warmup, anchor_mode="met", z_met=z_met)

    # NOTE: for physics_type == "shear", train_step must call shear_residual
    # instead of the pointwise Ekman residual, exactly as wired in
    # lidar_pinn_ekman_v2_extrap.py for the main shear-constrained runs. This
    # sweep script assumes that wiring already exists in the imported module;
    # it does not duplicate it here.

    ds = (tf.data.Dataset.from_tensor_slices((Xtr, Ztr, UVtr, Ctr, UVMtr))
          .shuffle(8192, seed=RANDOM_STATE).batch(FIXED_CFG["batch_size"]))
    UVva_np = UVva.numpy()
    best, best_w, wait, patience = np.inf, None, 0, 15
    for ep in range(1, epochs + 1):
        trainer.set_epoch(ep)
        for xb, zb, uvb, cb, mb in ds:
            trainer.train_step(xb, zb, uvb, cb, mb, None)
        uv_va = trainer.predict_uv(Xva, Zva).numpy()
        val_data = float(np.mean(((uv_va - UVva_np) / uv_scale) ** 2))
        if val_data < best - 1e-5:
            best, best_w, wait = val_data, [w.numpy() for w in model.weights], 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_w:
        for w, val in zip(model.weights, best_w):
            w.assign(val)

    uv_te = trainer.predict_uv(Xte, Zte).numpy()
    uv_true = UVte.numpy()
    spd_pred = np.sqrt(uv_te[..., 0] ** 2 + uv_te[..., 1] ** 2)
    spd_true = np.sqrt(uv_true[..., 0] ** 2 + uv_true[..., 1] ** 2)
    dir_pred = (np.degrees(np.arctan2(-uv_te[..., 0], -uv_te[..., 1])) + 360) % 360
    dir_true = (np.degrees(np.arctan2(-uv_true[..., 0], -uv_true[..., 1])) + 360) % 360
    overall_rmse = float(np.sqrt(np.mean((spd_pred - spd_true) ** 2)))
    overall_mae = float(np.mean(np.abs(spd_pred - spd_true)))
    derr = np.abs(((dir_pred - dir_true + 180) % 360) - 180)
    overall_dirmae = float(np.mean(derr))

    model.save_weights(os.path.join(out_run, "pinn_ekman.weights.h5"))
    res = {"overall_rmse": overall_rmse, "overall_mae": overall_mae,
           "overall_dir_mae": overall_dirmae, "n_train": int(tr_sub.size),
           "best_val_data": best, "physics_type": physics_type}
    json.dump(res, open(os.path.join(out_run, "pinn_ekman_results.json"), "w"), indent=2)
    return res


def main(npz_file, physics_type, full_model_dir, lat, epochs, w_ekman_warmup,
         z_met, contig_months, distrib_days, out_dir, n_past,
         min_valid_alts, k_clusters):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  physics_type={physics_type}  fixed architecture: "
          f"lstm1={FIXED_CFG['lstm1_units']} lstm2={FIXED_CFG['lstm2_units']} "
          f"dec={FIXED_CFG['dec_units']} ctx={FIXED_CFG['context_dim']}")
    d = build_everything(npz_file, n_past, min_valid_alts, k_clusters)
    n_train_full = int(d["tr"].size)
    months_in_pool = np.unique(d["times"][d["tr"]].astype("datetime64[M]")).size
    print(f"  full training pool: {n_train_full:,} samples across {months_in_pool} months")

    runs = []
    full_res_path = os.path.join(full_model_dir, "pinn_ekman_results.json")
    if os.path.exists(full_res_path):
        fr = json.load(open(full_res_path))
        runs.append({"scheme": "full", "size": months_in_pool, "n_train": n_train_full,
                     "rmse": fr["overall_rmse"], "mae": fr["overall_mae"],
                     "dir_mae": fr.get("overall_dir_mae") or
                                (float(np.mean(fr["mae_dir"])) if fr.get("mae_dir") else None)})
        print(f"  [full] RMSE {fr['overall_rmse']:.3f} (reference, not retrained)")
    else:
        print(f"  WARNING: {full_res_path} not found, no full-data reference point")

    for K in contig_months:
        tr_sub = subsample_train(d["tr"], d["times"], "contiguous", K)
        if tr_sub.size < 100:
            print(f"  [contig {K}m] only {tr_sub.size} samples, skipped"); continue
        print(f"  [contig {K}m] training on {tr_sub.size:,} samples ...")
        r = train_one(d, physics_type, tr_sub, lat, epochs, w_ekman_warmup, z_met,
                      os.path.join(out_dir, f"run_contig_{K}m"))
        runs.append({"scheme": "contiguous", "size": K, "n_train": r["n_train"],
                     "rmse": r["overall_rmse"], "mae": r["overall_mae"],
                     "dir_mae": r["overall_dir_mae"]})
        print(f"      RMSE {r['overall_rmse']:.3f}  dirMAE {r['overall_dir_mae']:.1f}")

    for W in distrib_days:
        tr_sub = subsample_train(d["tr"], d["times"], "distributed", W)
        if tr_sub.size < 100:
            print(f"  [distrib {W}d/m] only {tr_sub.size} samples, skipped"); continue
        print(f"  [distrib {W}d/month] training on {tr_sub.size:,} samples ...")
        r = train_one(d, physics_type, tr_sub, lat, epochs, w_ekman_warmup, z_met,
                      os.path.join(out_dir, f"run_distrib_{W}d"))
        runs.append({"scheme": "distributed", "size": W, "n_train": r["n_train"],
                     "rmse": r["overall_rmse"], "mae": r["overall_mae"],
                     "dir_mae": r["overall_dir_mae"]})
        print(f"      RMSE {r['overall_rmse']:.3f}  dirMAE {r['overall_dir_mae']:.1f}")

    json.dump({"physics_type": physics_type, "n_train_full": n_train_full,
               "months_in_pool": int(months_in_pool), "runs": runs},
              open(os.path.join(out_dir, "trainsize_sweep_summary.json"), "w"), indent=2)
    print(f"\n  Sweep complete. Summary in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz_file", nargs="?", default="lidar_dataset.npz")
    ap.add_argument("--physics-type", choices=["ekman", "shear", "none"], required=True)
    ap.add_argument("--full-model-dir", required=True,
                    help="e.g. out_pinn_indist_s42 / out_shear_indist_s42 / out_noek_indist_s42")
    ap.add_argument("--lat", type=float, default=42.615)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--w-ekman-warmup", type=int, default=20)
    ap.add_argument("--z-met", type=float, default=2.0)
    ap.add_argument("--contig-months", type=int, nargs="+", default=[1, 2, 3, 6, 9, 12])
    ap.add_argument("--distrib-days-per-month", type=int, nargs="+", default=[7, 14])
    ap.add_argument("-o", "--output-dir", default="trainsize_sweep")
    ap.add_argument("--n-past", type=int, default=18)
    ap.add_argument("--min-valid-alts", type=int, default=6)
    ap.add_argument("--k-clusters", type=int, default=3)
    args = ap.parse_args()
    main(args.npz_file, args.physics_type, args.full_model_dir, args.lat, args.epochs,
         args.w_ekman_warmup, args.z_met, args.contig_months, args.distrib_days_per_month,
         args.output_dir, args.n_past, args.min_valid_alts, args.k_clusters)
