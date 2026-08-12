"""

Classical / ML baselines for ABL wind-profile reconstruction

Reproduces the reference models so a new model can be compared against them:

  - logarithmic wind profile   U(z) = U_ref * ln(z/z0) / ln(z_ref/z0)
  - power-law wind profile      U(z) = U_ref * (z / z_ref)^alpha
  - constant profile            U(z) = U_ref            
  - random forest regressor     ground features -> full profile (data-driven ML)

All four are evaluated on the same stratified test partition used by
lidar_pinn_ekman_v2_tuning.py (same K-Means
regime labels, same test_size/random_state/two-step split order), so their
per-altitude errors and per-sample predictions are directly comparable and the
paired Wilcoxon test is valid.

Each baseline writes, under <out>/<name>/:
    <name>_results.json            same scheme as the NN models (rmse_spd,
                                   mae_spd, mae_dir=None, altitudes, overall_*)
    <name>_test_predictions.npz    spd_pred, spd_true, altitudes

Notes:

* The analytical laws need a near-surface reference. The surface met
  wind speed U_g is used as U_ref at reference height z_ref (--z-ref, default 2 m: the
  AIRMAR ultrasonic station of Garcia-Gutierrez et al. 2023 sits at 2 m AGL).
  Following both prior papers, the log-law roughness z0 and displacement d, and
  the power-law exponent alpha, are fitted by least squares on the training set
  rather than fixed (toggle with --no-fit-z0 / --no-fit-alpha). Anchoring at
  2 m over terrain with z0 ~ 1.9 m is an extreme extrapolation, exactly the
  regime where the fitted analytical laws are expected to struggle and the
  data-driven / physics-informed models to win; the least-squares fit gives the
  analytical baselines their fairest possible chance.
* z = 0 m is dropped for the analytical laws (ln(0) / 0^alpha undefined); it is
  kept for the constant model and the random forest, and for ALL per-altitude
  error reporting the analytical models emit NaN at 0 m so the altitude grid
  stays aligned across models.
* alpha (power-law) and z0 (log-law) default to literature/site values but are
  exposed as flags; alpha can instead be fit globally on the training set
  (--fit-alpha), which is the fairer data-informed variant.

Usage:

    python3 lidar_baselines_condireccion_extrap.py lidar_dataset.npz -o baselines_out \
        --n-past 6 --z-ref 2.0
    # z0,d and alpha are fitted on train by default; use --no-fit-z0 /
    # --no-fit-alpha to fall back to fixed --z0 / --alpha.
"""

import os
import json
import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# shared constants (match the other scripts)
RANDOM_STATE   = 42
K_CLUSTERS     = 3
MIN_VALID_ALTS = 6
WS_MIN, WS_MAX = 0.0, 80.0


def load_npz(path):
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def build_kmeans_labels(data, k_clusters, min_valid_alts):
    """Identical to the NN scripts, so the stratified split matches."""
    ws = data["wind_speed"]; wd = data["wind_dir"]
    valid = np.sum(~np.isnan(ws), axis=1) >= min_valid_alts
    rad = np.deg2rad(wd[valid])
    X_raw = np.hstack([ws[valid], np.sin(rad), np.cos(rad)])
    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(X_raw))
    km = KMeans(n_clusters=k_clusters, n_init=20, random_state=RANDOM_STATE)
    lab = km.fit_predict(X)
    labels = np.full(ws.shape[0], -1, dtype=int)
    labels[valid] = lab
    return labels


def get_ground_channels(data):
    """(N, 6): [WS_met, sin(WD_met), cos(WD_met), T, P, RH] — as in the NN scripts."""
    N = data["wind_speed"].shape[0]
    def s(k): return np.asarray(data.get(k, np.full(N, np.nan)), dtype=float)
    ws = s("met_wind_speed"); rad = np.deg2rad(s("met_wind_dir"))
    return np.column_stack([ws, np.sin(rad), np.cos(rad),
                            s("met_temperature"), s("met_pressure"),
                            s("met_humidity")])


def build_samples(data, labels, n_past, min_valid_alts):
    """Returns ground features g (last step), target speed profile, regime C,
    the surface reference speed U_g per sample, the true direction profile, and
    the surface (met) direction per sample.

    The retention criterion replicates the physics-informed pipeline exactly,
    including its two temporal-derivative channels (dws, dT), which are NaN at
    t=0 and wherever a met ws/T neighbour is NaN. Without this, the two
    pipelines keep a different number of windows (off by one on this dataset),
    the train_test_split permutation diverges, and per-sample pairing against
    the physics-informed model silently breaks even though the test-set sizes coincide."""
    ws_prof = data["wind_speed"].astype(float)
    wd_prof = data["wind_dir"].astype(float)          # true direction per height
    ground = get_ground_channels(data)
    N, F = ground.shape
    # Physics-informed-style derivative channels (only their NaN pattern matters here)
    dws = np.full(N, np.nan); dT = np.full(N, np.nan)
    dws[1:] = ground[1:, 0] - ground[:-1, 0]          # met wind speed = col 0
    dT[1:] = ground[1:, 3] - ground[:-1, 3]           # met temperature = col 3
    ground_chk = np.hstack([ground, np.column_stack([dws, dT])])   # (N, 8)
    valid = np.sum(~np.isnan(ws_prof), axis=1) >= min_valid_alts

    # surface met direction per time step (channel 1,2 are sin,cos of met dir)
    met_dir = np.rad2deg(np.arctan2(ground[:, 1], ground[:, 2])) % 360.0

    G, Y, C, Uref, Ydir, Dref = [], [], [], [], [], []
    for t in range(n_past - 1, N):
        if labels[t] < 0 or not valid[t]:
            continue
        seq = ground_chk[t - n_past + 1: t + 1]
        if not np.all(np.isfinite(seq)):
            continue
        G.append(ground[t])             # current-step ground features
        Y.append(ws_prof[t])
        C.append(labels[t])
        Uref.append(ground[t, 0])       # met wind speed (channel 0)
        Ydir.append(wd_prof[t])         # true direction profile
        Dref.append(met_dir[t])         # surface met direction
    return (np.asarray(G), np.asarray(Y), np.asarray(C, dtype=int),
            np.asarray(Uref), np.asarray(Ydir), np.asarray(Dref),
            data["altitudes"].astype(float))


def clean_targets(Y):
    bad = (Y < WS_MIN) | (Y > WS_MAX)
    Y[bad] = np.nan
    if np.isnan(Y).any():
        med = np.nanmedian(Y, axis=0)
        r, c = np.where(np.isnan(Y)); Y[r, c] = med[c]
    return Y


def per_alt_errors(pred, true):
    rmse = np.sqrt(np.nanmean((pred - true) ** 2, axis=0))
    mae = np.nanmean(np.abs(pred - true), axis=0)
    overall_rmse = float(np.sqrt(np.nanmean((pred - true) ** 2)))
    overall_mae = float(np.nanmean(np.abs(pred - true)))
    return rmse, mae, overall_rmse, overall_mae


def angular_diff(a, b):
    """Smallest signed angular difference a-b in degrees, wrapped to [-180,180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def per_alt_dir_errors(pred_deg, true_deg):
    """Per-altitude direction MAE (deg), wrapping the circular difference."""
    d = np.abs(angular_diff(pred_deg, true_deg))
    mae = np.nanmean(d, axis=0)
    overall = float(np.nanmean(d))
    return mae, overall


def fit_loglaw_z0_d(Y_tr, Uref_tr, z, z_ref, z0_init=1.91, d_init=0.0):
    """Least-squares fit of (z0, d) for the log-law shape factor, as done in
    García-Gutiérrez et al. 2021/2023.

    The single-anchor log-law predicts
        U(z) = U_ref * ln((z - d)/z0) / ln((z_ref - d)/z0),
    and (z0, d) are fitted by minimising the squared error between predicted and
    measured profiles over all training (sample, height) pairs. The fit is
    nonlinear because z0 and d sit inside the logarithm, so a bounded
    least-squares solver is used. Falls back to the literature z0 (1.91 m, d=0) if SciPy
    is unavailable or the optimiser fails.

    Y_tr must already be restricted to the positive heights `z` (same number of
    columns as len(z)).

    NOTE: with a 2 m surface anchor and z0 ~ 1.9 m, (z_ref - d)/z0 is close to 1
    and ln(.) is tiny, so the shape factor is numerically delicate; the bounds
    keep z0 < z_ref and the denominator away from zero.
    """
    z = np.asarray(z, float)
    rng = np.random.default_rng(RANDOM_STATE)
    sel = tr_subsample(Y_tr.shape[0], rng, cap=20000)
    Ys = Y_tr[sel]; Us = Uref_tr[sel]
    valid_u = Us > 0.1
    Ys = Ys[valid_u]; Us = Us[valid_u]
    if Ys.shape[0] == 0:
        return z0_init, d_init

    try:
        from scipy.optimize import least_squares
    except Exception:
        print("  [log-law] SciPy unavailable; using literature z0=1.91, d=0")
        return z0_init, d_init

    def residuals(theta):
        z0, d = theta
        denom = np.log((z_ref - d) / z0)
        if not np.isfinite(denom) or abs(denom) < 1e-6:
            return np.full(Ys.size, 1e3)
        arg = (z[None, :] - d) / z0
        arg = np.where(arg > 1e-6, arg, np.nan)          # ln domain
        shape = np.log(arg) / denom                      # (n, H)
        pred = Us[:, None] * shape
        res = (pred - Ys)
        return np.where(np.isfinite(res), res, 0.0).ravel()

    # bounds: 0.05 m <= z0 < z_ref ; 0 <= d < z_ref (physical, keeps ln defined)
    hi_z0 = max(z_ref * 0.95, 0.1)
    hi_d = max(z_ref * 0.5, 1e-3)
    try:
        sol = least_squares(
            residuals, x0=[min(z0_init, hi_z0 * 0.9), min(d_init, hi_d * 0.5)],
            bounds=([0.05, 0.0], [hi_z0, hi_d]), max_nfev=200)
        z0_fit, d_fit = float(sol.x[0]), float(sol.x[1])
        print(f"  [log-law] fitted z0={z0_fit:.3f} m, d={d_fit:.3f} m "
              f"(z_ref={z_ref} m)")
        return z0_fit, d_fit
    except Exception as e:
        print(f"  [log-law] fit failed ({e}); using literature z0=1.91, d=0")
        return z0_init, d_init


def tr_subsample(n, rng, cap=20000):
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, size=cap, replace=False)


def group_metrics(pred, true, dpred, dtrue, vmask, gate_sel):
    """Speed/dir errors over genuine measurements at the selected gates,
    requiring a finite prediction (analytical laws emit NaN at z=0; the RF
    emits NaN at unsupervised gates in extrapolation mode)."""
    m = vmask & gate_sel[None, :] & np.isfinite(pred)
    if not m.any():
        return None
    es = (pred - true)[m]
    out = {"rmse_spd": float(np.sqrt(np.mean(es ** 2))),
           "mae_spd": float(np.mean(np.abs(es))),
           "n_entries": int(m.sum())}
    if dpred is not None and dtrue is not None:
        md = m & np.isfinite(dpred) & np.isfinite(dtrue)
        if md.any():
            ed = np.abs(angular_diff(dpred[md], dtrue[md]))
            out["mae_dir"] = float(np.mean(ed))
    return out


def save_model(out_dir, name, alts, pred, true, rmse, mae, ormse, omma, extra=None,
               dir_pred=None, dir_true=None, dir_mae=None, dir_overall=None,
               seen=None, vmask=None):
    d = os.path.join(out_dir, name); os.makedirs(d, exist_ok=True)
    payload = {"model": name, "overall_rmse": ormse, "overall_mae": omma,
               "rmse_spd": [None if np.isnan(x) else float(x) for x in rmse],
               "mae_spd": [None if np.isnan(x) else float(x) for x in mae],
               "mae_dir": None, "altitudes": alts.tolist()}
    if dir_mae is not None:
        payload["mae_dir"] = [None if np.isnan(x) else float(x) for x in dir_mae]
        payload["overall_dir_mae"] = dir_overall
    if extra:
        payload["config"] = extra
    # seen/unseen summary on genuine (non-imputed) measurements
    res_seen = res_unseen = None
    if seen is not None and vmask is not None:
        res_seen = group_metrics(pred, true, dir_pred, dir_true, vmask, seen)
        res_unseen = (group_metrics(pred, true, dir_pred, dir_true, vmask, ~seen)
                      if (~seen).any() else None)
        payload["seen_gates"] = res_seen
        payload["unseen_gates"] = res_unseen
    with open(os.path.join(d, f"{name}_results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    arrs = {"altitudes": alts, "spd_pred": pred.astype(np.float32),
            "spd_true": true.astype(np.float32)}
    if dir_pred is not None and dir_true is not None:
        arrs["dir_pred"] = dir_pred.astype(np.float32)
        arrs["dir_true"] = dir_true.astype(np.float32)
    if seen is not None:
        arrs["seen_mask"] = seen
    if vmask is not None:
        arrs["valid_mask"] = vmask
    np.savez_compressed(os.path.join(d, f"{name}_test_predictions.npz"), **arrs)
    tag = "" if dir_mae is None else f"  dirMAE {dir_overall:.1f} deg"
    print(f"  {name:14s} overall RMSE {ormse:.3f}  MAE {omma:.3f} m/s{tag}")
    if res_seen:
        print(f"                 SEEN   RMSE {res_seen['rmse_spd']:.3f}  "
              f"MAE {res_seen['mae_spd']:.3f}"
              + (f"  dirMAE {res_seen['mae_dir']:.1f} deg"
                 if "mae_dir" in res_seen else ""))
    if res_unseen:
        print(f"                 UNSEEN RMSE {res_unseen['rmse_spd']:.3f}  "
              f"MAE {res_unseen['mae_spd']:.3f}"
              + (f"  dirMAE {res_unseen['mae_dir']:.1f} deg"
                 if "mae_dir" in res_unseen else ""))
    elif seen is not None and (~seen).any():
        print(f"                 UNSEEN —  (no finite predictions: model "
              f"cannot extrapolate to unsupervised heights)")


def main(npz_file, out_dir, n_past, min_valid_alts, k_clusters,
         z_ref, z0, alpha, fit_alpha, fit_z0, train_max_height=None):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    print("=" * 66)
    print("  Classical / ML baselines on the shared stratified test split")
    print(f"  z_ref={z_ref} m (surface met sensor height)")
    print(f"  log-law: {'fitting z0,d by least squares' if fit_z0 else f'fixed z0={z0}, d=0'}"
          f" | power-law: {'fitting alpha' if fit_alpha else f'fixed alpha={alpha}'}")
    if train_max_height is not None:
        print(f"  EXTRAPOLATION MODE: fits/training use only gates <= "
              f"{train_max_height} m; upper gates held out")
    print("=" * 66)

    data = load_npz(npz_file)
    labels = build_kmeans_labels(data, k_clusters, min_valid_alts)
    G, Y, C, Uref, Ydir, Dref, alts = build_samples(data, labels, n_past, min_valid_alts)
    # validity mask before imputation: genuine, in-range measurements. The
    # seen/unseen metrics use only these, so imputed medians never enter them.
    VMASK = np.isfinite(Y) & (Y >= WS_MIN) & (Y <= WS_MAX)
    Y = clean_targets(Y)
    M = G.shape[0]
    print(f"  Samples: {M:,} | N_alt {len(alts)}")

    # supervised-gates mask (matches the *_extrap NN scripts)
    if train_max_height is not None:
        seen = alts <= float(train_max_height)
        if not seen.any():
            raise ValueError("train_max_height below the lowest gate")
        print(f"  Fitting on {int(seen.sum())}/{len(alts)} gates; HELD OUT: "
              f"{[int(h) for h in alts[~seen]]}")
    else:
        seen = np.ones(len(alts), dtype=bool)

    # SAME split as the NN scripts (indices, stratified, identical params/order)
    idx = np.arange(M)
    tr, tmp = train_test_split(idx, test_size=0.3,
                               random_state=RANDOM_STATE, stratify=C)
    va, te = train_test_split(tmp, test_size=0.5,
                              random_state=RANDOM_STATE, stratify=C[tmp])
    print(f"  Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")

    alts_safe = alts.copy()
    pos = alts_safe > 0                      # heights where log/power are defined
    pos_fit = pos & seen                     # gates the analytical fits may use
    Yte = Y[te]; Uref_te = Uref[te]
    VMASK_te = VMASK[te]
    Ydir_te = Ydir[te]; Dref_te = Dref[te]   # true direction profile, surface dir
    n_alt = len(alts)

    # Direction for the analytical laws (log/power/constant): these are 1-D
    # speed laws with no rotation physics, so the only direction they can offer
    # is the "no-veering" assumption — surface direction held constant with
    # height. It is reported as such; a trivial reference, not a prediction.
    dir_const = np.repeat(Dref_te[:, None], n_alt, axis=1)
    dmae_const, dovr_const = per_alt_dir_errors(dir_const, Ydir_te)

    # analytical: power-law
    a = alpha
    if fit_alpha:
        # global least-squares of ln(U/Uref) = alpha * ln(z/z_ref) on train,
        # pooling all valid (sample, height) pairs with positive z and speed.
        # In extrapolation mode only the supervised gates enter the fit; the
        # prediction below still extends the fitted law over the full column.
        zt = alts_safe[pos_fit]
        with np.errstate(divide="ignore", invalid="ignore"):
            num, den = [], []
            for s in tr:
                u = Y[s][pos_fit]; ur = Uref[s]
                ok = (u > 0.1) & (ur > 0.1)
                num.append(np.log(u[ok] / ur))
                den.append(np.log(zt[ok] / z_ref))
            num = np.concatenate(num); den = np.concatenate(den)
            a = float(np.sum(den * num) / np.sum(den * den))
        print(f"  [power-law] fitted alpha = {a:.4f}")

    pred_pow = np.full_like(Yte, np.nan)
    pred_pow[:, pos] = Uref_te[:, None] * (alts_safe[pos][None, :] / z_ref) ** a
    r, m, orr, omm = per_alt_errors(pred_pow, Yte)
    save_model(out_dir, "power_law", alts, pred_pow, Yte, r, m, orr, omm,
               extra={"alpha": a, "z_ref": z_ref,
                      "train_max_height": train_max_height,
                      "direction": "surface-constant (no veering)"},
               dir_pred=dir_const, dir_true=Ydir_te,
               dir_mae=dmae_const, dir_overall=dovr_const,
               seen=seen, vmask=VMASK_te)

    # analytical: log-law
    z0_use, d_use = z0, 0.0
    if fit_z0:
        z0_use, d_use = fit_loglaw_z0_d(Y[tr][:, pos_fit], Uref[tr],
                                        alts_safe[pos_fit], z_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.log((z_ref - d_use) / z0_use)
        arg = (alts_safe[pos] - d_use) / z0_use
        arg = np.where(arg > 1e-6, arg, np.nan)
        shape = np.log(arg) / denom
    pred_log = np.full_like(Yte, np.nan)
    pred_log[:, pos] = Uref_te[:, None] * shape[None, :]
    r, m, orr, omm = per_alt_errors(pred_log, Yte)
    save_model(out_dir, "log_law", alts, pred_log, Yte, r, m, orr, omm,
               extra={"z0": z0_use, "d": d_use, "z_ref": z_ref,
                      "z0_fitted": bool(fit_z0),
                      "train_max_height": train_max_height,
                      "direction": "surface-constant (no veering)"},
               dir_pred=dir_const, dir_true=Ydir_te,
               dir_mae=dmae_const, dir_overall=dovr_const,
               seen=seen, vmask=VMASK_te)

    # constant / persistence
    pred_con = np.repeat(Uref_te[:, None], len(alts), axis=1)
    r, m, orr, omm = per_alt_errors(pred_con, Yte)
    save_model(out_dir, "constant", alts, pred_con, Yte, r, m, orr, omm,
               extra={"z_ref": z_ref,
                      "train_max_height": train_max_height,
                      "direction": "surface-constant (no veering)"},
               dir_pred=dir_const, dir_true=Ydir_te,
               dir_mae=dmae_const, dir_overall=dovr_const,
               seen=seen, vmask=VMASK_te)

    # random forest (data-driven ML baseline)
    # To compare like-for-like with the physics-informed model, the random forest is trained to
    # predict the (u, v) wind components at every height (not the speed scalar),
    # from which both speed and direction are recovered: a genuine, learned
    # direction, exactly as the architecture-matched arms produce it.
    #
    # meteorological convention: u = -S*sin(dir), v = -S*cos(dir); recovering
    # dir = atan2(-u, -v) mod 360.  Targets are built from the true speed and
    # direction profiles.
    def speed_dir_to_uv(S, D_deg):
        rad = np.deg2rad(D_deg)
        return -S * np.sin(rad), -S * np.cos(rad)

    def uv_to_speed_dir(u, v):
        S = np.sqrt(u ** 2 + v ** 2)
        D = (np.rad2deg(np.arctan2(-u, -v))) % 360.0
        return S, D

    # build (u,v) targets per height for all samples; clean dir NaNs by imputation
    Ydir_clean = Ydir.copy()
    if np.isnan(Ydir_clean).any():
        # circular-aware fill via per-height median of sin/cos
        s_, c_ = np.sin(np.deg2rad(Ydir_clean)), np.cos(np.deg2rad(Ydir_clean))
        for col in range(n_alt):
            sm, cm = np.nanmedian(s_[:, col]), np.nanmedian(c_[:, col])
            bad = np.isnan(Ydir_clean[:, col])
            Ydir_clean[bad, col] = np.rad2deg(np.arctan2(sm, cm)) % 360.0
    U_comp, V_comp = speed_dir_to_uv(Y, Ydir_clean)        # (M, n_alt) each

    # In extrapolation mode the RF (fixed-grid) may
    # only be supervised at the seen gates; it has no mechanism to predict
    # unsupervised heights, so those columns are reported as NaN — the RF is
    # structurally incapable of vertical extrapolation.
    seen_idx = np.where(seen)[0]
    UV = np.concatenate([U_comp[:, seen_idx], V_comp[:, seen_idx]], axis=1)
    n_seen = len(seen_idx)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(G[tr], UV[tr])                     # ground features -> (u,v) profile
    pred_uv = rf.predict(G[te])
    pu_s, pv_s = pred_uv[:, :n_seen], pred_uv[:, n_seen:]
    pred_rf = np.full_like(Yte, np.nan)
    pred_rf_dir = np.full_like(Yte, np.nan)
    s_spd, s_dir = uv_to_speed_dir(pu_s, pv_s)             # genuine speed & dir
    pred_rf[:, seen_idx] = s_spd
    pred_rf_dir[:, seen_idx] = s_dir

    r, m, orr, omm = per_alt_errors(pred_rf, Yte)
    dmae_rf, dovr_rf = per_alt_dir_errors(pred_rf_dir, Ydir_te)
    save_model(out_dir, "random_forest", alts, pred_rf, Yte, r, m, orr, omm,
               extra={"n_estimators": 300, "min_samples_leaf": 5,
                      "target": "uv-components (speed+direction)",
                      "train_max_height": train_max_height,
                      "extrapolation": ("structurally incapable: NaN at "
                                        "held-out gates"
                                        if train_max_height is not None
                                        else None)},
               dir_pred=pred_rf_dir, dir_true=Ydir_te,
               dir_mae=dmae_rf, dir_overall=dovr_rf,
               seen=seen, vmask=VMASK_te)

    print("\n  Baselines written to:", os.path.abspath(out_dir))
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Classical/ML baselines on the shared test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz_file", nargs="?", default="lidar_dataset.npz")
    ap.add_argument("-o", "--output-dir", default="baselines_out")
    ap.add_argument("--n-past-choices", type=int, nargs="+", default=[6, 12, 18],
                    help="One or more look-back windows. For each value a "
                         "subfolder <output-dir>_np<n> is created. Each MUST "
                         "match the n_past of the model it is compared against, "
                         "so the test partition coincides. Pass a single value "
                         "(e.g. --n-past-choices 18) to do just one.")
    ap.add_argument("--min-valid-alts", type=int, default=MIN_VALID_ALTS)
    ap.add_argument("--k-clusters", type=int, default=K_CLUSTERS)
    ap.add_argument("--z-ref", type=float, default=2.0,
                    help="reference height of the surface met sensor [m]. The "
                         "AIRMAR station in Garcia-Gutierrez et al. 2023 sits "
                         "at 2 m; set to the LiDAR's lowest valid level if the "
                         "anchor is the profile itself (2021 setup).")
    ap.add_argument("--z0", type=float, default=1.91,
                    help="log-law roughness length [m] used only if --no-fit-z0")
    ap.add_argument("--alpha", type=float, default=0.143,
                    help="power-law exponent used only if --no-fit-alpha")
    ap.add_argument("--no-fit-alpha", dest="fit_alpha", action="store_false",
                    help="use the fixed --alpha instead of fitting it on train")
    ap.add_argument("--no-fit-z0", dest="fit_z0", action="store_false",
                    help="use the fixed --z0 (d=0) instead of fitting z0,d on train")
    ap.add_argument("--train-max-height", type=float, default=None,
                    help="if set, the analytical fits (z0/d, alpha) and the RF "
                         "training only use gates <= this height; upper gates "
                         "are held out and reported separately (matches the "
                         "vertical-extrapolation experiment of the NN scripts)")
    ap.set_defaults(fit_alpha=True, fit_z0=True)
    args = ap.parse_args()

    # Run each requested look-back window into its own subfolder, so a single
    # invocation produces baselines_out_np6 / _np12 / _np18.
    # The base output dir name has the _np<n> suffix appended; if
    # a name ending in a number is already passed it still just appends.
    multi = len(args.n_past_choices) > 1
    for n_past in args.n_past_choices:
        out_dir = (f"{args.output_dir}_np{n_past}" if multi else args.output_dir)
        print("\n" + "#" * 66)
        print(f"#  n_past = {n_past}  ->  {out_dir}")
        print("#" * 66)
        main(args.npz_file, out_dir, n_past, args.min_valid_alts,
             args.k_clusters, args.z_ref, args.z0, args.alpha,
             args.fit_alpha, args.fit_z0,
             train_max_height=args.train_max_height)
