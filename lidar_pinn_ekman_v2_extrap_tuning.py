"""
Hyperparameter tuning for the Ekman model (Bayesian optimisation).

Imports the model, trainer and data pipeline from lidar_pinn_ekman_v2_extrap.py
so that a tuned configuration trains exactly like the final model.

The model uses a custom training loop (nested GradientTape for the
d/dz(Km dU/dz) residual, Ekman-weight curriculum, early stopping on the
validation data loss), which is not compatible with the default tuner.search.
The search therefore subclasses keras_tuner.BayesianOptimization and overrides
run_trial, training each proposed configuration with PINNTrainer.

Objective: the best validation data loss reached during a trial, restricted to
the height-band validation gates when --val-max-height is set. The total loss
is not used as objective, so configurations are selected on profile
reconstruction rather than on how strongly they satisfy the residual.

Search space: lstm1_units, lstm2_units, dec_units, context_dim, learning_rate,
batch_size, and (physics arm only) w_ekman, w_ekman_warmup, w_ustar, w_bc.
n_past is an outer loop, since changing it rebuilds the dataset.

Requires keras-tuner.

Usage:

    python3 lidar_pinn_ekman_v2_extrap_tuning.py lidar_dataset.npz \
        -o tune_out --max-trials 25 --n-past-choices 18 \
        --train-max-height 111 --val-max-height 140 \
        --anchor-mode met --z-met 2.0 --lat 42.615
"""

import os
import gc
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Model, trainer and data pipeline are imported from the production script.
import lidar_pinn_ekman_v2_extrap as ek


# FIXED CONFIGURATION

NPZ_FILE_DEFAULT = "lidar_dataset.npz"
OUTPUT_DIR       = "./pinn_ekman_tuning_out"
MIN_VALID_ALTS   = 6
RANDOM_STATE     = 42
EPOCHS_SEARCH    = 40       # epochs per trial during the search
EPOCHS_FINAL     = 120      # epochs for the final retrain of the best config
SEARCH_PATIENCE  = 8        # early-stopping patience during trials
FINAL_PATIENCE   = 15       # early-stopping patience for the final retrain

# Search space (architecture / optimisation)
N_PAST_CHOICES   = [6, 12, 18]
LSTM1_UNITS      = [64, 128, 256]
LSTM2_UNITS      = [32, 64, 128]
DEC_UNITS        = [32, 64, 128]
CONTEXT_DIM      = [16, 32, 64]
LR_CHOICES       = [1e-4, 5e-4, 1e-3, 5e-3]
BATCH_CHOICES    = [128, 256]

# Search space (physics)
W_EKMAN_MIN, W_EKMAN_MAX = 1e-4, 5e-2      # sampled on a log scale
W_EKMAN_WARMUP_CHOICES   = [10, 20, 40]
W_BC_CHOICES             = [0.0, 0.05, 0.1]
W_USTAR_CHOICES = [0.0, 0.02, 0.05, 0.1, 0.2]   # u_star anchor weight (0 = off)


# DATA PREP (uses the production pipeline verbatim, then splits to tensors)

def prepare_tensors(npz_file, n_past, min_valid_alts, k_clusters,
                    train_max_height=None, val_max_height=None):
    """Replicates the data preparation block of lidar_pinn_ekman_v2_extrap.main(),
    returning ready-to-use tensors plus the constants the trainer needs.

    With train_max_height set, also returns the supervised-gates mask `seen`
    (gates <= train_max_height) used to restrict L_data and the model-selection
    objective, plus the pre-imputation validity mask of the test partition for
    the seen/unseen evaluation."""
    data = ek.load_npz(npz_file)
    labels = ek.build_kmeans_labels(data, k_clusters, min_valid_alts)
    X, U, V, C, alts, UV_met = ek.build_sequences(
        data, labels, n_past, min_valid_alts, return_extras=True)
    M, T_steps, F = X.shape

    # clean physically implausible speeds
    spd = np.sqrt(U**2 + V**2)
    bad = (spd < ek.WS_MIN) | (spd > ek.WS_MAX)
    if bad.any():
        U[bad] = np.nan; V[bad] = np.nan
    # validity mask before imputation (genuine measurements only)
    VMASK = np.isfinite(U) & np.isfinite(V)
    for arr in (U, V):
        if np.isnan(arr).any():
            med = np.nanmedian(arr, axis=0)
            r, cc = np.where(np.isnan(arr)); arr[r, cc] = med[cc]

    uv = np.stack([U, V], axis=-1).astype(np.float32)
    uv_scale = float(np.std(uv) + 1e-6)
    Z_data = np.tile(alts.astype(np.float32), (M, 1))

    # gates supervised by L_data
    if train_max_height is not None:
        seen = alts <= float(train_max_height)
        if not seen.any():
            raise ValueError("train_max_height below the lowest gate")
    else:
        seen = np.ones(U.shape[1], dtype=bool)

    # height-band validation (definitive protocol): selection objective on
    # the band, untouched test above it
    if val_max_height is not None:
        if train_max_height is None:
            raise ValueError("val_max_height requires train_max_height")
        band = (alts > float(train_max_height)) & (alts <= float(val_max_height))
        if not band.any():
            raise ValueError("no gates fall in the validation band")
    else:
        band = None

    # neutral log-law bulk u_star at the lowest positive seen gate (for the
    # optional w_ustar anchor; observable data only)
    pos_seen = seen & (alts > 0)
    z_anchor = float(alts[pos_seen].min())
    j_anchor = int(np.where(alts == z_anchor)[0][0])
    spd_anchor = np.sqrt(U[:, j_anchor] ** 2 + V[:, j_anchor] ** 2)
    U_BULK = (ek.KAPPA * spd_anchor / np.log(z_anchor / ek.Z0_DEF)).astype(np.float32)

    # scale inputs (scaler fitted on finite rows)
    Xf = X.reshape(M * T_steps, F)
    fin = np.all(np.isfinite(Xf), axis=1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xf[fin])
    Xf_s = Xf.copy(); Xf_s[fin] = scaler.transform(Xf[fin])
    X_s = Xf_s.reshape(M, T_steps, F).astype(np.float32)

    # stratified split on regime (same as production)
    from sklearn.model_selection import train_test_split
    idx = np.arange(M)
    tr, tmp = train_test_split(idx, test_size=0.3,
                               random_state=RANDOM_STATE, stratify=C)
    va, te = train_test_split(tmp, test_size=0.5,
                              random_state=RANDOM_STATE, stratify=C[tmp])

    def take(ix):
        return (tf.constant(X_s[ix]), tf.constant(Z_data[ix]),
                tf.constant(uv[ix]), tf.constant(C[ix]),
                tf.constant(UV_met[ix]), tf.constant(U_BULK[ix]))

    return {
        "train": take(tr), "val": take(va), "test": take(te),
        "alts": alts, "F": F, "uv_scale": uv_scale,
        "scaler": scaler, "n_alt": U.shape[1],
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "seen": seen, "band": band,
        "train_max_height": train_max_height,
        "val_max_height": val_max_height,
        "vmask_test": VMASK[te],
    }


# One trial: build model from hp, run the production PINNTrainer loop,
# return the best validation data loss reached.

def run_one_trial(hp, dat, f_cor, z0, d, k_clusters, epochs, patience,
                  capture_history=False, anchor_mode="zero", z_met=2.0,
                  physics_off=False, n_colloc=None):
    n_past_steps = dat["train"][0].shape[1]
    F = dat["F"]

    lstm1   = hp.Choice("lstm1_units",   LSTM1_UNITS)
    lstm2   = hp.Choice("lstm2_units",   LSTM2_UNITS)
    dec     = hp.Choice("dec_units",     DEC_UNITS)
    ctxd    = hp.Choice("context_dim",   CONTEXT_DIM)
    lr      = hp.Choice("learning_rate", LR_CHOICES)
    batch   = hp.Choice("batch_size",    BATCH_CHOICES)
    if physics_off:
        # physics-free arm: same architecture and search space, residual
        # disabled; the physics weights are not searched in this mode
        w_ekman, warmup, w_ustar = 0.0, 1, 0.0
    else:
        w_ekman = hp.Float("w_ekman", W_EKMAN_MIN, W_EKMAN_MAX, sampling="log")
        warmup  = hp.Choice("w_ekman_warmup", W_EKMAN_WARMUP_CHOICES)
        w_ustar = hp.Choice("w_ustar", W_USTAR_CHOICES)
    w_bc    = hp.Choice("w_bc", W_BC_CHOICES)
    a_cls   = ek.A_CLS_DEF      # kept fixed (auxiliary task; not the focus)

    model = ek.EkmanMOSTPINN(n_past_steps, F, k_clusters, z0=z0, d=d,
                             lstm1=lstm1, lstm2=lstm2,
                             dec_units=dec, context_dim=ctxd)
    model.adapt_height(dat["alts"])
    Xtr, Ztr, UVtr, Ctr, UVMtr, UBtr = dat["train"]
    _ = model((Xtr[:2], Ztr[:2]))     # build

    seen = dat["seen"]
    band = dat["band"]
    trainer = ek.PINNTrainer(model, keras.optimizers.Adam(lr),
                             w_ekman, w_bc, a_cls, f_cor,
                             float(min(dat["alts"])), float(max(dat["alts"])),
                             dat["uv_scale"], w_ekman_warmup=warmup,
                             data_mask=(seen if dat["train_max_height"]
                                        is not None else None),
                             anchor_mode=anchor_mode, z_met=z_met,
                             n_colloc=n_colloc, w_ustar=w_ustar)

    ds = (tf.data.Dataset.from_tensor_slices(
              (Xtr, Ztr, UVtr, Ctr, UVMtr, UBtr))
          .shuffle(8192, seed=RANDOM_STATE).batch(batch))

    Xva, Zva, UVva, _, _, _ = dat["val"]
    UVva_np = UVva.numpy()
    uv_scale = dat["uv_scale"]

    hist = {"val_data": [], "w_ekman": []} if capture_history else None
    best, best_w, wait = np.inf, None, 0
    for ep in range(1, epochs + 1):
        w_eff = trainer.set_epoch(ep)
        for xb, zb, uvb, cb, mb, ub in ds:
            trainer.train_step(xb, zb, uvb, cb, mb, ub)
        uv_va = trainer.predict_uv(Xva, Zva).numpy()
        # selection objective: validation band if defined, else the
        # supervised gates; the test gates never enter selection
        _vsel = band if band is not None else seen
        val_data = float(np.mean(
            ((uv_va[:, _vsel, :] - UVva_np[:, _vsel, :]) / uv_scale) ** 2))
        if capture_history:
            hist["val_data"].append(val_data); hist["w_ekman"].append(w_eff)
        if val_data < best - 1e-5:
            best, best_w, wait = val_data, [w.numpy() for w in model.weights], 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_w:
        for w, val in zip(model.weights, best_w):
            w.assign(val)
    return best, model, trainer, hist


# SEARCH over one n_past value using a Bayesian oracle

def search_one_npast(npz_file, out_dir, n_past, max_trials, min_valid_alts,
                     k_clusters, f_cor, z0, d, epochs_search,
                     train_max_height=None, val_max_height=None,
                     anchor_mode="zero", z_met=2.0,
                     physics_off=False, n_colloc=None):
    print(f"\n{'='*74}\n  Tuning Ekman arm with n_past = {n_past}"
          f"{'  [PHYSICS-OFF arm]' if physics_off else ''}\n{'='*74}")
    dat = prepare_tensors(npz_file, n_past, min_valid_alts, k_clusters,
                          train_max_height=train_max_height,
                          val_max_height=val_max_height)
    print(f"  Train {dat['n_train']:,} | Val {dat['n_val']:,} | "
          f"Test {dat['n_test']:,} | N_alt {dat['n_alt']}")
    if train_max_height is not None:
        seen = dat["seen"]
        band = dat["band"]
        if band is not None:
            print(f"  L_data on gates <= {train_max_height} m | selection on "
                  f"band {[int(h) for h in dat['alts'][band]]} m | TEST: "
                  f"{[int(h) for h in dat['alts'][~seen & ~band]]}")
        else:
            print(f"  L_data & selection objective on {int(seen.sum())}/"
                  f"{dat['n_alt']} gates (<= {train_max_height} m); held out: "
                  f"{[int(h) for h in dat['alts'][~seen]]}")

    # subclass the tuner and override run_trial to drive the custom loop
    counter = {"i": 0}

    class EkmanTuner(kt.BayesianOptimization):
        def run_trial(self, trial, dat, f_cor, z0, d, k_clusters,
                      epochs_search):
            hp = trial.hyperparameters
            try:
                score, _, _, _ = run_one_trial(
                    hp, dat, f_cor, z0, d, k_clusters,
                    epochs_search, SEARCH_PATIENCE,
                    anchor_mode=anchor_mode, z_met=z_met,
                    physics_off=physics_off, n_colloc=n_colloc)
            except tf.errors.ResourceExhaustedError:
                score = float("inf")
            counter["i"] += 1
            wek_txt = ("off" if physics_off
                       else f"{hp.get('w_ekman'):.1e}/warm{hp.get('w_ekman_warmup')}"
                            f"/us{hp.get('w_ustar')}")
            print(f"  trial {counter['i']:2d}/{max_trials} | val_data={score:.5f} | "
                  f"lstm1={hp.get('lstm1_units')} lstm2={hp.get('lstm2_units')} "
                  f"dec={hp.get('dec_units')} ctx={hp.get('context_dim')} "
                  f"lr={hp.get('learning_rate')} bs={hp.get('batch_size')} "
                  f"w_ek={wek_txt} "
                  f"w_bc={hp.get('w_bc')}")
            keras.backend.clear_session(); gc.collect()
            # report the objective back to the tuner
            return {"val_data": score}

    # declare the search space
    hp_space = kt.HyperParameters()
    hp_space.Choice("lstm1_units", LSTM1_UNITS)
    hp_space.Choice("lstm2_units", LSTM2_UNITS)
    hp_space.Choice("dec_units", DEC_UNITS)
    hp_space.Choice("context_dim", CONTEXT_DIM)
    hp_space.Choice("learning_rate", LR_CHOICES)
    hp_space.Choice("batch_size", BATCH_CHOICES)
    if not physics_off:
        hp_space.Float("w_ekman", W_EKMAN_MIN, W_EKMAN_MAX, sampling="log")
        hp_space.Choice("w_ekman_warmup", W_EKMAN_WARMUP_CHOICES)
        hp_space.Choice("w_ustar", W_USTAR_CHOICES)
    hp_space.Choice("w_bc", W_BC_CHOICES)

    tuner = EkmanTuner(
        hyperparameters=hp_space,
        tune_new_entries=False,
        objective=kt.Objective("val_data", direction="min"),
        max_trials=max_trials,
        seed=RANDOM_STATE,
        directory=out_dir,
        project_name=f"ekman_tuner_npast_{n_past}",
        overwrite=True,
    )
    tuner.search(dat=dat, f_cor=f_cor, z0=z0, d=d, k_clusters=k_clusters,
                 epochs_search=epochs_search)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_score = float(best_trial.score)
    best = {"n_past": n_past, "best_val_data": best_score}
    best.update(dict(best_hp.values))
    return best, dat


# Main


# Retrain a configuration to full capacity, evaluate on the test set and write
# its best_n_past_<n>/ folder (weights, meta, predictions, results, figures).
def retrain_and_save(best, npz_file, out_dir, f_cor, z0, d, k_clusters,
                     min_valid_alts, epochs_final, tag="",
                     train_max_height=None, val_max_height=None,
                     anchor_mode="zero", z_met=2.0,
                     physics_off=False, n_colloc=None):
    n_past = best["n_past"]
    print(f"\n  Retraining {tag}config (n_past={n_past}) "
          f"for up to {epochs_final} epochs ...")
    dat = prepare_tensors(npz_file, n_past, min_valid_alts, k_clusters,
                          train_max_height=train_max_height,
                          val_max_height=val_max_height)

    hp = kt.HyperParameters()
    hp.Fixed("lstm1_units",    best["lstm1_units"])
    hp.Fixed("lstm2_units",    best["lstm2_units"])
    hp.Fixed("dec_units",      best["dec_units"])
    hp.Fixed("context_dim",    best["context_dim"])
    hp.Fixed("learning_rate",  best["learning_rate"])
    hp.Fixed("batch_size",     best["batch_size"])
    if not physics_off:
        hp.Fixed("w_ekman",        best["w_ekman"])
        hp.Fixed("w_ekman_warmup", best["w_ekman_warmup"])
        hp.Fixed("w_ustar",        best.get("w_ustar", 0.0))
    hp.Fixed("w_bc",           best["w_bc"])

    best_val, model, trainer, hist = run_one_trial(
        hp, dat, f_cor, z0, d, k_clusters,
        epochs_final, FINAL_PATIENCE, capture_history=True,
        anchor_mode=anchor_mode, z_met=z_met, physics_off=physics_off,
        n_colloc=n_colloc)

    # test evaluation (physical units)
    Xte, Zte, UVte, _, _, _ = dat["test"]
    uv_te = trainer.predict_uv(Xte, Zte).numpy()
    uv_true = UVte.numpy()
    spd_pred = np.sqrt(uv_te[..., 0]**2 + uv_te[..., 1]**2)
    spd_true = np.sqrt(uv_true[..., 0]**2 + uv_true[..., 1]**2)
    dir_pred = (np.degrees(np.arctan2(-uv_te[..., 0], -uv_te[..., 1])) + 360) % 360
    dir_true = (np.degrees(np.arctan2(-uv_true[..., 0], -uv_true[..., 1])) + 360) % 360
    rmse_spd = np.sqrt(np.mean((spd_pred - spd_true)**2, axis=0))
    mae_spd = np.mean(np.abs(spd_pred - spd_true), axis=0)
    derr = np.abs(((dir_pred - dir_true + 180) % 360) - 180)
    mae_dir = np.mean(derr, axis=0)
    overall_rmse = float(np.sqrt(np.mean((spd_pred - spd_true)**2)))
    overall_mae = float(np.mean(np.abs(spd_pred - spd_true)))
    alts = dat["alts"]
    seen = dat["seen"]
    band = dat["band"]

    print(f"  n_past={n_past} test speed RMSE {overall_rmse:.3f} m/s  "
          f"MAE {overall_mae:.3f} m/s")
    print("  height  RMSE_spd  MAE_spd  MAE_dir(deg)")
    for h, rs, ms, md, sn in zip(alts, rmse_spd, mae_spd, mae_dir, seen):
        held = "" if sn else "   [HELD-OUT]"
        print(f"    {int(h):4d} m   {rs:6.3f}   {ms:6.3f}   {md:6.1f}{held}")

    # seen / unseen summary on genuine (non-imputed) measurements
    vmask_te = dat["vmask_test"]
    err_spd = spd_pred - spd_true

    def _group_metrics(gate_sel):
        m = vmask_te & gate_sel[None, :]
        if not m.any():
            return None
        es = err_spd[m]; ed = derr[m]
        return {"rmse_spd": float(np.sqrt(np.mean(es ** 2))),
                "mae_spd": float(np.mean(np.abs(es))),
                "mae_dir": float(np.mean(ed)),
                "n_entries": int(m.sum()),
                "heights": [int(h) for h in alts[gate_sel]]}

    res_seen = _group_metrics(seen)
    res_band = _group_metrics(band) if band is not None else None
    res_unseen = _group_metrics(~seen if band is None else (~seen & ~band))
    if res_band:
        print(f"  VBAND  gates: RMSE {res_band['rmse_spd']:.3f}  "
              f"MAE {res_band['mae_spd']:.3f}  dirMAE {res_band['mae_dir']:.1f} deg"
              f"  [selection band]")
    if res_seen:
        print(f"  SEEN   gates: RMSE {res_seen['rmse_spd']:.3f}  "
              f"MAE {res_seen['mae_spd']:.3f}  dirMAE {res_seen['mae_dir']:.1f} deg")
    if res_unseen:
        print(f"  UNSEEN gates: RMSE {res_unseen['rmse_spd']:.3f}  "
              f"MAE {res_unseen['mae_spd']:.3f}  dirMAE {res_unseen['mae_dir']:.1f} deg")

    # save
    best_dir = os.path.join(out_dir, f"best_n_past_{n_past}")
    os.makedirs(best_dir, exist_ok=True)
    model.save_weights(os.path.join(best_dir, "pinn_ekman.weights.h5"))
    scaler = dat["scaler"]
    np.savez_compressed(
        os.path.join(best_dir, "pinn_ekman_meta.npz"),
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        altitudes=alts, rmse_spd=rmse_spd, mae_spd=mae_spd, mae_dir=mae_dir,
        uv_scale=dat["uv_scale"], f=f_cor, z0=z0, d=d)
    np.savez_compressed(
        os.path.join(best_dir, "pinn_ekman_test_predictions.npz"),
        altitudes=alts, seen_mask=seen,
        band_mask=(band if band is not None else np.zeros_like(seen)),
        valid_mask=vmask_te,
        spd_pred=spd_pred.astype(np.float32), spd_true=spd_true.astype(np.float32),
        dir_pred=dir_pred.astype(np.float32), dir_true=dir_true.astype(np.float32))
    with open(os.path.join(best_dir, "pinn_ekman_results.json"), "w") as fh:
        json.dump({"model": ("ekman_pinn_physics_off" if physics_off
                             else "ekman_pinn"),
                   "overall_rmse": overall_rmse, "overall_mae": overall_mae,
                   "rmse_spd": rmse_spd.tolist(), "mae_spd": mae_spd.tolist(),
                   "mae_dir": mae_dir.tolist(),
                   "altitudes": np.asarray(alts).tolist(),
                   "best_val_data": best_val,
                   "seen_gates": res_seen, "band_gates": res_band,
                   "unseen_gates": res_unseen,
                   "experiment": {"train_max_height": train_max_height,
                                  "val_max_height": val_max_height,
                                  "anchor_mode": anchor_mode, "z_met": z_met,
                                  "physics_off": physics_off},
                   "config": best}, fh, indent=2)

    # plots
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist["val_data"], color="tab:red", label="val L_data (early-stop)")
        axw = ax.twinx()
        axw.plot(hist["w_ekman"], color="tab:green", ls=":", label="w_ekman (curriculum)")
        axw.set_ylabel("w_ekman")
        ax.set(xlabel="Epoch", ylabel="Validation data loss",
               title=f"Ekman PINN — n_past={n_past}  RMSE={overall_rmse:.3f} m/s")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(best_dir, "final_training.png"), dpi=150)
        plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 6))
    ax2[0].plot(rmse_spd, alts, "o-", label="RMSE"); ax2[0].plot(mae_spd, alts, "s--", label="MAE")
    ax2[0].set(xlabel="Speed error (m/s)", ylabel="Height (m)", title="Speed"); ax2[0].legend(); ax2[0].grid(alpha=.3)
    ax2[1].plot(mae_dir, alts, "o-", color="tab:purple")
    ax2[1].set(xlabel="Direction MAE (deg)", ylabel="Height (m)", title="Direction"); ax2[1].grid(alpha=.3)
    fig2.tight_layout(); fig2.savefig(os.path.join(best_dir, "error_profile.png"), dpi=150)
    plt.close(fig2)

    keras.backend.clear_session(); gc.collect()
    return overall_rmse, best_dir


def main(npz_file, out_dir, max_trials, n_past_choices, min_valid_alts,
         k_clusters, lat, z0, d, epochs_search, epochs_final,
         gpu_mem=None,
         train_max_height=None, val_max_height=None,
         anchor_mode="zero", z_met=2.0,
         physics_off=False, n_colloc=None):
    os.makedirs(out_dir, exist_ok=True)
    ek.configure_gpu_memory(gpu_mem)
    tf.random.set_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    f_cor = 2 * ek.OMEGA * np.sin(np.deg2rad(lat))

    print("=" * 74)
    print("  Ekman model — hyperparameter tuning (Bayesian optimisation)")
    print(f"  numpy {np.__version__} | TF {tf.__version__}")
    print(f"  lat={lat} deg -> f={f_cor:.3e} 1/s | z0={z0} d={d}")
    print(f"  objective: minimise validation DATA loss (profile fit, "
          f"supervised gates)")
    print(f"  anchor_mode={anchor_mode} (z_met={z_met} m) | "
          f"train_max_height={train_max_height} | "
          f"val_max_height={val_max_height} | "
          f"physics_off={physics_off}")
    print("=" * 74)

    all_results = []
    rmse_by_npast = {}
    for n_past in n_past_choices:
        best, _ = search_one_npast(
            npz_file, out_dir, n_past, max_trials, min_valid_alts,
            k_clusters, f_cor, z0, d, epochs_search,
            train_max_height=train_max_height, val_max_height=val_max_height,
            anchor_mode=anchor_mode,
            z_met=z_met, physics_off=physics_off, n_colloc=n_colloc)
        print(f"\n  Best for n_past={n_past}: val_data={best['best_val_data']:.5f}")
        for kk, vv in best.items():
            print(f"    {kk}: {vv}")
        all_results.append(best)
        keras.backend.clear_session(); gc.collect()

        # retrain this n_past's winning config and write its output folder
        rmse, _ = retrain_and_save(
            best, npz_file, out_dir, f_cor, z0, d, k_clusters,
            min_valid_alts, epochs_final, tag=f"n_past={n_past} ",
            train_max_height=train_max_height, val_max_height=val_max_height,
            anchor_mode=anchor_mode,
            z_met=z_met, physics_off=physics_off, n_colloc=n_colloc)
        rmse_by_npast[n_past] = rmse

    # rank by validation data loss
    all_sorted = sorted(all_results, key=lambda r: r["best_val_data"])
    best_overall = all_sorted[0]
    print("\n" + "=" * 74)
    print("  SEARCH SUMMARY (sorted by validation data loss)")
    print("=" * 74)
    for r in all_sorted:
        wek_txt = ("off" if r.get("w_ekman") is None
                   else f"{r['w_ekman']:.1e}/warm{r.get('w_ekman_warmup')}")
        print(f"  n_past={r['n_past']:2d}  val_data={r['best_val_data']:.5f}  "
              f"test_rmse={rmse_by_npast.get(r['n_past'], float('nan')):.3f}  "
              f"lstm1={r.get('lstm1_units')} lstm2={r.get('lstm2_units')} "
              f"dec={r.get('dec_units')} ctx={r.get('context_dim')} "
              f"lr={r.get('learning_rate')} bs={r.get('batch_size')} "
              f"w_ek={wek_txt} "
              f"w_bc={r.get('w_bc')}")
    print(f"\n  >>> Best overall (by val_data): n_past={best_overall['n_past']}  "
          f"val_data={best_overall['best_val_data']:.5f} <<<")

    with open(os.path.join(out_dir, "ekman_tuning_summary.json"), "w") as fh:
        json.dump({"results": all_sorted,
                   "test_rmse_by_npast": rmse_by_npast,
                   "best_overall_npast": best_overall["n_past"]}, fh, indent=2)

    # bar chart of test RMSE by look-back window
    if rmse_by_npast:
        items = sorted(rmse_by_npast.items(), key=lambda kv: kv[1])
        labels = [str(n) for n, _ in items]
        values = [v for _, v in items]
        figb, axb = plt.subplots(figsize=(9, 6))
        bars = axb.bar(labels, values, color="tab:blue", width=0.6)
        axb.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
        axb.set(xlabel="n_past (look-back steps)", ylabel="Test RMSE (m/s)",
                title="Ekman arm — Best RMSE by look-back window")
        axb.grid(axis="y", alpha=.3)
        axb.set_ylim(0, max(values) * 1.15)
        figb.tight_layout()
        figb.savefig(os.path.join(out_dir, "ekman_rmse_by_npast.png"), dpi=150)
        plt.close(figb)
        print(f"  Bar chart: {os.path.join(out_dir, 'ekman_rmse_by_npast.png')}")

    print(f"\n  All n_past folders written under: {os.path.abspath(out_dir)}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ekman arm (u,v) — Bayesian hyperparameter tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz_file", nargs="?", default=NPZ_FILE_DEFAULT)
    ap.add_argument("-o", "--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--max-trials", type=int, default=30,
                    help="Bayesian optimisation trials per n_past value")
    ap.add_argument("--n-past-choices", type=int, nargs="+", default=N_PAST_CHOICES,
                    help="Look-back windows to search over (outer loop)")
    ap.add_argument("--min-valid-alts", type=int, default=MIN_VALID_ALTS)
    ap.add_argument("--k-clusters", type=int, default=ek.K_CLUSTERS)
    ap.add_argument("--lat", type=float, default=42.615, help="site latitude (deg)")
    ap.add_argument("--z0", type=float, default=ek.Z0_DEF)
    ap.add_argument("--d", type=float, default=ek.D_DEF)
    ap.add_argument("--epochs-search", type=int, default=EPOCHS_SEARCH,
                    help="epochs per trial during the search")
    ap.add_argument("--epochs-final", type=int, default=EPOCHS_FINAL,
                    help="epochs for the final retrain of the best config")
    ap.add_argument("--train-max-height", type=float, default=None,
                    help="if set, L_data and the selection objective only use "
                         "gates <= this height; upper gates are held out and "
                         "reported separately (vertical extrapolation)")
    ap.add_argument("--val-max-height", type=float, default=None,
                    help="if set (requires --train-max-height), the selection "
                         "objective is the extrapolation error on the band "
                         "(train_max_height, val_max_height]; gates above are "
                         "the untouched test")
    ap.add_argument("--anchor-mode", choices=["zero", "met", "none"],
                    default="zero",
                    help="'zero' = original no-slip anchor at z0+d; "
                         "'met' = anchor to the measured met-station (u,v) at "
                         "--z-met; 'none' = no anchor")
    ap.add_argument("--z-met", type=float, default=2.0,
                    help="met-station measurement height (m) for --anchor-mode met")
    ap.add_argument("--gpu-mem", type=int, default=None,
                    help="explicit GPU memory limit in MB (bypasses TF's "
                         "free-memory autodetection; e.g. 7000)")
    ap.add_argument("--n-colloc", type=int, default=None,
                    help="collocation heights per step for the Ekman residual; "
                         "use the SAME value in both arms")
    ap.add_argument("--physics-off", action="store_true",
                    help="tune the matched physics-free arm: identical "
                         "architecture and search space but w_ekman fixed to 0 "
                         "(w_ekman and warmup removed from the search space)")
    args = ap.parse_args()
    main(args.npz_file, args.output_dir, args.max_trials, args.n_past_choices,
         args.min_valid_alts, args.k_clusters, args.lat, args.z0, args.d,
         args.epochs_search, args.epochs_final,
         gpu_mem=args.gpu_mem,
         train_max_height=args.train_max_height,
         val_max_height=args.val_max_height, anchor_mode=args.anchor_mode,
         z_met=args.z_met, physics_off=args.physics_off,
         n_colloc=args.n_colloc)
