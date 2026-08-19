"""
LiDAR ABL wind-profile PINN — Ekman balance with Monin-Obukhov closure

Reconstructs the full ABL wind profile (both horizontal components u, v, hence
speed and direction) from a single near-surface measurement time series, using
the stationary, horizontally homogeneous boundary-layer momentum balance

    d/dz ( Km(z) du/dz ) + f (v - v_g) = 0
    d/dz ( Km(z) dv/dz ) - f (u - u_g) = 0

as a soft constraint, with

    Km(z) = kappa * u_star * (z - d) / phi_m((z - d) / L)

where (u_g, v_g) is the geostrophic wind, f the Coriolis parameter, Km the eddy
viscosity and phi_m the Businger-Dyer stability function. In neutral conditions
the closure recovers the logarithmic surface layer; aloft the balance produces
the Ekman spiral, i.e. the veering of wind direction with height. Posing the
constraint on (u, v) rather than on the speed modulus is what allows the model
to represent that veering.

Architecture:

    surface sequence (n_past, F=8) -> Bi-LSTM encoder -> context c
        c      -> physics head: u_star > 0, 1/L, (u_g, v_g)
        (c, z) -> decoder: (u(z), v(z)), continuous in z for autodiff

Loss:

    L = L_data + a_cls*L_cls + w_ekman*L_ekman + w_bc*L_bc

    L_data   MSE on (u, v) at the measured heights
    L_cls    K-Means regime classification head (--a-cls 0 disables it)
    L_ekman  physics residual, selected explicitly via --physics-type:
               'ekman' - the two Ekman-MOST residuals above, with du/dz,
                         dv/dz and d/dz(Km du/dz) by automatic differentiation
               'shear' - parameterised power-law shear constraint on the
                         speed modulus alone (z dU/dz - alpha*U = 0), see
                         physics_shear_powerlaw.py
               'none'  - zero, regardless of --w-ekman
             Its weight is ramped from 0 to the target over --w-ekman-warmup
             epochs; early stopping is monitored on L_data, not the total loss
    L_bc     surface anchor, either (u, v) -> 0 at z0 + d or the measured
             met-station wind at --z-met (see --anchor-mode)

Vertical extrapolation:

With --train-max-height, L_data is restricted to the gates below that height
and the remaining gates are held out; --val-max-height additionally reserves a
band for model selection, so the test gates above it are never used for
training, early stopping or hyperparameter selection.

Input:

The .npz must contain wind_speed (N, N_alt), wind_dir (N, N_alt),
altitudes (N_alt,), met_wind_speed, met_wind_dir, met_temperature,
met_pressure and met_humidity (N,).

Usage:

    python3 lidar_pinn_ekman_v2_extrap.py lidar_dataset.npz -o out \
        --n-past 18 --epochs 120 --lat 42.615 --physics-type ekman \
        --train-max-height 111 --val-max-height 140 \
        --anchor-mode met --z-met 2.0

    # shear-constrained arm, same architecture, leakage-free temporal split
    python3 lidar_pinn_ekman_v2_extrap.py lidar_dataset.npz -o out_shear \
        --n-past 18 --epochs 120 --lat 42.615 --physics-type shear \
        --anchor-mode met --z-met 2.0 \
        --temporal-split-file temporal_block_split.npz

    # physics-free control (same architecture, w_ekman=0)
    python3 lidar_pinn_ekman_v2_extrap.py lidar_dataset.npz -o out_noek \
        --n-past 18 --epochs 120 --lat 42.615 --w-ekman 0 \
        --anchor-mode met --z-met 2.0
"""

import os
import argparse
import json
import numpy as np

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from physics_shear_powerlaw import shear_residual

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CONFIGURATION

NPZ_FILE_DEFAULT = "lidar_dataset.npz"
OUTPUT_DIR       = "./pinn_ekman_out"
N_PAST           = 6
MIN_VALID_ALTS   = 6
K_CLUSTERS       = 3
RANDOM_STATE     = 42
BATCH_SIZE       = 256
EPOCHS           = 150
WS_MIN, WS_MAX   = 0.0, 80.0      # physical wind-speed range [m/s]

# Physics constants
KAPPA   = 0.40                    # von Karman constant
BETA_S  = 4.7                     # Businger-Dyer stable coefficient
GAMMA_U = 15.0                    # Businger-Dyer unstable coefficient
OMEGA   = 7.2921159e-5            # Earth angular velocity [rad/s]
Z0_DEF  = 1.91                    # roughness length [m] (Leon site)
D_DEF   = 0.0                     # zero-plane displacement [m]

# Loss weights (defaults; tune via CLI)
W_EKMAN_DEF = 5e-3
W_EKMAN_WARMUP_DEF = 20           # epochs to ramp Ekman weight 0 -> target
W_BC_DEF    = 0.05
A_CLS_DEF   = 0.2

N_COLLOCATION = 24                # collocation heights per step for L_ekman


# DATA LOADING

def load_npz(path):
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def build_kmeans_labels(data, k_clusters, min_valid_alts):
    """K-Means regime labels on [U, sin(dir), cos(dir)] over all heights."""
    ws = data["wind_speed"]
    wd = data["wind_dir"]
    valid = np.sum(~np.isnan(ws), axis=1) >= min_valid_alts
    rad = np.deg2rad(wd[valid])
    X_raw = np.hstack([ws[valid], np.sin(rad), np.cos(rad)])
    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(X_raw))
    km = KMeans(n_clusters=k_clusters, n_init=20, random_state=RANDOM_STATE)
    lab_valid = km.fit_predict(X)
    labels = np.full(ws.shape[0], -1, dtype=int)
    labels[valid] = lab_valid
    return labels


def get_ground_base(data):
    N = data["wind_speed"].shape[0]
    def s(k): return np.asarray(data.get(k, np.full(N, np.nan)), dtype=float)
    ws = s("met_wind_speed"); wd = s("met_wind_dir")
    T = s("met_temperature"); P = s("met_pressure"); RH = s("met_humidity")
    rad = np.deg2rad(wd)
    return np.column_stack([ws, np.sin(rad), np.cos(rad), T, P, RH]), ws, T


def build_ground_features_with_time(data):
    base, ws, T = get_ground_base(data)
    N = base.shape[0]
    dws = np.full(N, np.nan); dT = np.full(N, np.nan)
    dws[1:] = ws[1:] - ws[:-1]; dT[1:] = T[1:] - T[:-1]
    return np.hstack([base, np.column_stack([dws, dT])])   # (N, 8)


def build_sequences(data, labels, n_past, min_valid_alts, return_extras=False):
    """Targets are (u, v) profiles, derived from speed & direction.

    With return_extras=True, additionally returns:
      UV_met : (M, 2) met-station (u, v) at the target time t, for use as a
               surface data anchor of the continuous-z decoder.
    (Default signature unchanged so the tuning script keeps working.)
    """
    ws_prof = data["wind_speed"].astype(float)    # (N, N_alt)
    wd_prof = data["wind_dir"].astype(float)       # (N, N_alt)
    alts = data["altitudes"].astype(float)
    ground = build_ground_features_with_time(data)
    N, F = ground.shape

    # meteorological convention: u = -|V| sin(dir), v = -|V| cos(dir)
    rad = np.deg2rad(wd_prof)
    u_prof = -ws_prof * np.sin(rad)
    v_prof = -ws_prof * np.cos(rad)

    # met-station wind at target time (columns of `ground`: ws, sin, cos, ...)
    met_ws = np.asarray(data.get("met_wind_speed"), dtype=float)
    met_wd = np.asarray(data.get("met_wind_dir"), dtype=float)
    met_rad = np.deg2rad(met_wd)
    u_met = -met_ws * np.sin(met_rad)
    v_met = -met_ws * np.cos(met_rad)

    valid = np.sum(~np.isnan(ws_prof), axis=1) >= min_valid_alts
    X_list, U_list, V_list, C_list, M_list = [], [], [], [], []
    for t in range(n_past - 1, N):
        if labels[t] < 0 or not valid[t]:
            continue
        seq = ground[t - n_past + 1: t + 1]
        if not np.all(np.isfinite(seq)):
            continue
        X_list.append(seq)
        U_list.append(u_prof[t]); V_list.append(v_prof[t])
        C_list.append(labels[t])
        M_list.append([u_met[t], v_met[t]])
    X = np.stack(X_list)
    U = np.stack(U_list); V = np.stack(V_list)
    C = np.asarray(C_list, dtype=int)
    if return_extras:
        UV_met = np.asarray(M_list, dtype=np.float32)
        return X, U, V, C, alts, UV_met
    return X, U, V, C, alts


# STABILITY FUNCTION

def phi_m(zeta):
    stable = 1.0 + BETA_S * zeta
    unstable = tf.pow(tf.maximum(1.0 - GAMMA_U * zeta, 1e-3), -0.25)
    return tf.where(zeta >= 0.0, stable, unstable)


# MODEL

def _lstm_kwargs():
    """Set PINN_LSTM_NO_CUDNN=1 to force the generic (non-cuDNN) LSTM
    implementation via unroll=True. Needed on TF 2.17 with cuDNN 9, where the
    fused RNN kernel fails; semantically identical and cheap for n_past <= 18.
    Unset by default."""
    if os.environ.get("PINN_LSTM_NO_CUDNN", "0") == "1":
        return {"unroll": True}
    return {}


class EkmanMOSTPINN(keras.Model):
    def __init__(self, n_steps, n_feat, n_classes, z0=Z0_DEF, d=D_DEF,
                 lstm1=128, lstm2=64, dec_units=64, context_dim=32, **kw):
        super().__init__(**kw)
        self.z0 = float(z0); self.d = float(d)
        self.enc1 = layers.Bidirectional(
            layers.LSTM(lstm1, return_sequences=True, **_lstm_kwargs()))
        self.ln = layers.LayerNormalization()
        self.enc2 = layers.Bidirectional(
            layers.LSTM(lstm2, **_lstm_kwargs()))
        self.enc_dense = layers.Dense(context_dim, activation="tanh")

        # physics head: u_star (>0), 1/L (free), (u_g, v_g)
        self.phys1 = layers.Dense(32, activation="relu")
        self.ustar_out = layers.Dense(1, activation="softplus")
        self.invL_out  = layers.Dense(1, activation="linear")
        self.ug_out    = layers.Dense(2, activation="linear")   # (u_g, v_g)
        self.alpha_out = layers.Dense(1, activation="sigmoid")

        # decoder: (context, z) -> (u, v)
        self.dec1 = layers.Dense(dec_units, activation="tanh")
        self.dec2 = layers.Dense(dec_units, activation="tanh")
        self.dec_out = layers.Dense(2, activation="linear")

        # optional classification head
        self.cls_out = layers.Dense(n_classes, activation="softmax")

        self.z_mean = tf.Variable(150.0, trainable=False, dtype=tf.float32)
        self.z_std  = tf.Variable(100.0, trainable=False, dtype=tf.float32)

    def adapt_height(self, alts):
        self.z_mean.assign(float(np.mean(alts)))
        self.z_std.assign(float(np.std(alts) + 1e-6))

    def encode(self, x):
        h = self.enc1(x); h = self.ln(h); h = self.enc2(h)
        return self.enc_dense(h)

    def physics_params(self, c):
        h = self.phys1(c)
        u_star = self.ustar_out(h) + 1e-3
        inv_L = self.invL_out(h)
        ug = self.ug_out(h)                  # (B,2)
        alpha = 0.6 * self.alpha_out(h)      # (B,1) in [0, 0.6]
        return u_star, inv_L, ug, alpha

    def regime(self, c):
        return self.cls_out(c)

    def profile(self, c, z):
        """(u,v) at heights z. c:(B,C), z:(B,K) -> (B,K,2)."""
        K = tf.shape(z)[1]
        z_norm = (z - self.z_mean) / self.z_std
        c_rep = tf.repeat(c[:, None, :], K, axis=1)
        zin = tf.concat([c_rep, z_norm[..., None]], axis=-1)
        h = self.dec1(zin); h = self.dec2(h)
        return self.dec_out(h)               # (B,K,2)

    def Km(self, z, u_star, inv_L):
        """Stability-dependent eddy viscosity, MOST closure. z:(B,K).

        zeta is clamped to the empirical validity range of the surface-layer
        similarity functions (approximately -2 <= zeta <= 10; Foken 2006).
        Beyond it, phi_m = 1 + 5*zeta is an unphysical extrapolation, and it
        also opens a degeneracy: the optimizer can inflate 1/L to kill Km at
        the upper collocation heights and trivialise the Ekman residual
        the unsupervised heights."""
        z_eff = tf.maximum(z - self.d, 1.0)
        zeta = tf.clip_by_value(z_eff * inv_L, -2.0, 10.0)   # (B,K)
        return KAPPA * u_star * z_eff / phi_m(zeta)   # (B,K)

    def call(self, inputs, training=False):
        x, z, uv_met = inputs
        c = self.encode(x)
        return self.profile(c, z)


# TRAINER (custom loop for Ekman residual via autodiff)

class PINNTrainer:
    def __init__(self, model, opt, w_ekman, w_bc, a_cls, f_coriolis,
                 z_min, z_max, uv_scale, w_ekman_warmup=20,
                 data_mask=None, anchor_mode="zero", z_met=2.0,
                 n_colloc=None, w_ustar=0.0, physics_type="ekman"):
        # w_ustar : weight of the surface-layer u_star anchor, regularising
        #           the inferred friction velocity toward the neutral bulk
        #           log-law estimate at the lowest supervised gate (0 = off).
        # n_colloc : collocation heights per step for the Ekman residual
        #            (default N_COLLOCATION). Both arms must use the same
        #            entirely anyway).
        # data_mask : optional bool (N_alt,), gates supervised in L_data;
        #             the remaining gates are held out. The Ekman residual
        #             still spans the full
        #             column via the collocation points.
        # anchor_mode: "zero" = (u, v) -> 0 at z0 + d,
        #              "met"  = anchor to the measured met-station (u, v)
        #                       at height z_met,
        #              "none" = no anchor term.
        self.m = model; self.opt = opt
        self.physics_type = str(physics_type)
        self.anchor_mode = str(anchor_mode)
        self.z_met = float(z_met)
        if data_mask is None:
            self.data_w = None
        else:
            self.data_w = tf.constant(
                np.asarray(data_mask, dtype=np.float32))
        # --- Ekman-weight curriculum ---
        # w_ekman is the target weight; the effective weight is ramped
        # linearly from 0 over `w_ekman_warmup` epochs, so the data fit is
        # allowed to
        # settle before the Ekman residual is given full strength.
        self.w_ekman_target = float(w_ekman)
        self.w_ekman_warmup = int(max(w_ekman_warmup, 1))
        # effective weight used inside the (graph-compiled) train_step;
        # updated once per epoch from the training loop.
        self.w_ekman = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.w_bc = w_bc; self.a_cls = a_cls
        self.f = float(f_coriolis)
        self.z_min = float(z_min); self.z_max = float(z_max)
        self.uv_scale = tf.constant(uv_scale, dtype=tf.float32)
        self.n_colloc = int(n_colloc) if n_colloc else N_COLLOCATION
        self.w_ustar = float(w_ustar)

    def set_epoch(self, epoch):
        """Update the effective Ekman weight for the current epoch (1-based).

        Linear ramp: 0 at epoch 1, reaching the target at w_ekman_warmup,
        and held at the target thereafter.
        """
        frac = min(float(epoch) / float(self.w_ekman_warmup), 1.0)
        self.w_ekman.assign(self.w_ekman_target * frac)
        return float(self.w_ekman.numpy())

    def _colloc(self, B):
        z = tf.random.uniform((B, self.n_colloc), self.z_min, self.z_max)
        return tf.sort(z, axis=1)

    @tf.function
    def train_step(self, x, z_data, uv_data, c_lab, uv_met=None,
                   u_bulk=None):
        B = tf.shape(x)[0]
        z_col = (self._colloc(B) if self.w_ekman_target != 0.0 else None)
        with tf.GradientTape() as tape:
            ctx = self.m.encode(x)
            u_star, inv_L, ug, alpha = self.m.physics_params(ctx)
            ug_u = ug[:, 0:1]; ug_v = ug[:, 1:2]

            # data loss on (u, v), restricted to the supervised gates when
            # a height mask is set
            uv_pred = self.m.profile(ctx, z_data)            # (B,Kd,2)
            data_res = (uv_pred - uv_data) / self.uv_scale
            sq = tf.square(data_res)
            if self.data_w is None:
                L_data = tf.reduce_mean(sq)
            else:
                w = self.data_w[None, :, None]               # (1,Kd,1)
                L_data = tf.reduce_sum(sq * w) / (
                    tf.reduce_sum(w) * tf.cast(B, tf.float32) * 2.0)

            # classification loss (optional)
            probs = self.m.regime(ctx)
            L_cls = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(c_lab, probs))

            # physics residual: Ekman-MOST momentum balance, or the
            # parameterised shear constraint (z dU/dz - alpha U = 0),
            # selected explicitly via self.physics_type
            if self.w_ekman_target == 0.0 or self.physics_type == "none":
                L_ekman = tf.constant(0.0, dtype=tf.float32)
            elif self.physics_type == "shear":
                L_ekman = shear_residual(
                    self.m, ctx, alpha, self.z_min, self.z_max, self.uv_scale)
            elif self.physics_type == "ekman":
                # need d/dz( Km dU/dz ): use nested gradient tapes
                with tf.GradientTape(persistent=True) as t2:
                    t2.watch(z_col)
                    with tf.GradientTape(persistent=True) as t1:
                        t1.watch(z_col)
                        uv_col = self.m.profile(ctx, z_col)      # (B,Kc,2)
                        u_c = uv_col[..., 0]; v_c = uv_col[..., 1]
                    du = t1.gradient(u_c, z_col)                 # (B,Kc)
                    dv = t1.gradient(v_c, z_col)
                    Km = self.m.Km(z_col, u_star, inv_L)         # (B,Kc)
                    flux_u = Km * du
                    flux_v = Km * dv
                    del t1
                dflux_u = t2.gradient(flux_u, z_col)             # d/dz(Km du/dz)
                dflux_v = t2.gradient(flux_v, z_col)
                del t2

                res_u = dflux_u + self.f * (v_c - ug_v)
                res_v = dflux_v - self.f * (u_c - ug_u)
                L_ekman = tf.reduce_mean(tf.square(res_u / self.uv_scale)
                                         + tf.square(res_v / self.uv_scale))
            else:
                raise ValueError(
                    f"unknown physics_type {self.physics_type!r}; "
                    f"expected 'ekman', 'shear' or 'none'")

            # surface anchor
            if self.anchor_mode == "met" and uv_met is not None:
                # anchor to the measured met-station wind at z_met
                z_bc = tf.fill((B, 1), self.z_met)
                uv_bc = self.m.profile(ctx, z_bc)[:, 0, :]   # (B,2)
                L_bc = tf.reduce_mean(
                    tf.square((uv_bc - uv_met) / self.uv_scale))
            elif self.anchor_mode == "zero":
                # original behaviour: (u,v)->0 at z0+d
                z_bc = tf.fill((B, 1), self.m.z0 + self.m.d)
                uv_bc = self.m.profile(ctx, z_bc)
                L_bc = tf.reduce_mean(tf.square(uv_bc / self.uv_scale))
            else:
                L_bc = tf.constant(0.0, dtype=tf.float32)

            # surface-layer u_star anchor (see __init__)
            if self.w_ustar > 0.0 and u_bulk is not None:
                L_us = tf.reduce_mean(tf.square(
                    (u_star[:, 0] - u_bulk) / (u_bulk + 0.05)))
            else:
                L_us = tf.constant(0.0, dtype=tf.float32)

            loss = (L_data + self.a_cls * L_cls
                    + self.w_ekman * L_ekman + self.w_bc * L_bc
                    + self.w_ustar * L_us)

        grads = tape.gradient(loss, self.m.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.opt.apply_gradients(zip(grads, self.m.trainable_variables))
        return loss, L_data, L_cls, L_ekman, L_bc

    @tf.function
    def predict_uv(self, x, z, batch=1024):
        """Batched forward pass over the full split. Chunking keeps the
        evaluation within GPU memory and changes nothing numerically."""
        n = int(x.shape[0])
        if n <= batch:
            return self.m.profile(self.m.encode(x), z)
        outs = []
        for i in range(0, n, batch):
            outs.append(self.m.profile(self.m.encode(x[i:i + batch]),
                                       z[i:i + batch]))
        return tf.concat(outs, axis=0)


# MAIN

def configure_gpu_memory(mb):
    """Set an explicit GPU memory limit, bypassing TF's free-memory
    autodetection, which under-reports inside the NGC container. Must run
    before any TF op initialises the device."""
    if not mb:
        return
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=int(mb))])
        print(f"  GPU memory limit set explicitly: {int(mb)} MB")


def load_config_json(path):
    """Extract a hyperparameter dict from a tuning output JSON.

    Accepts either:
      - ekman_tuning_summary.json  ({"results": [...], "best_overall_npast": N})
        -> returns the best-overall entry;
      - best_n_past_*/pinn_ekman_results.json ({"config": {...}})
        -> returns the "config" dict;
      - a flat dict of hyperparameters.
    Recognised keys: n_past, lstm1_units, lstm2_units, dec_units, context_dim,
    learning_rate, batch_size, w_ekman, w_ekman_warmup, w_bc.
    """
    with open(path) as fh:
        j = json.load(fh)
    if isinstance(j, dict) and isinstance(j.get("config"), dict):
        return dict(j["config"])
    if isinstance(j, dict) and "results" in j:
        res = j["results"]
        bestn = j.get("best_overall_npast")
        for r in res:
            if bestn is None or r.get("n_past") == bestn:
                return dict(r)
        return dict(res[0])
    if isinstance(j, list) and j and isinstance(j[0], dict):
        # bare list of result entries; pick the best
        # by validation loss if available, else the first (lists were sorted
        # best-first)
        for key in ("best_val_data", "best_val_loss"):
            if all(key in r for r in j):
                return dict(min(j, key=lambda r: r[key]))
        return dict(j[0])
    return dict(j)


def main(npz_file, out, n_past, min_valid_alts, epochs,
         w_ekman, w_bc, a_cls, lat, z0, d, k_clusters, w_ekman_warmup=20,
         train_max_height=None, anchor_mode="zero", z_met=2.0,
         lstm1=None, lstm2=None, dec_units=None, context_dim=None,
         lr=None, batch_size=None, config_json=None, n_colloc=None,
         w_ustar=None, val_max_height=None, seed=None, gpu_mem=None,
         temporal_split_file=None, physics_type="ekman"):
    configure_gpu_memory(gpu_mem)
    # resolve hyperparameters: explicit CLI > config JSON > defaults
    cfg = load_config_json(config_json) if config_json else {}

    def _pick(cli_val, key, default):
        if cli_val is not None:
            return cli_val
        if cfg.get(key) is not None:
            return cfg[key]
        return default

    n_past          = int(_pick(n_past, "n_past", N_PAST))
    lstm1           = int(_pick(lstm1, "lstm1_units", 128))
    lstm2           = int(_pick(lstm2, "lstm2_units", 64))
    dec_units       = int(_pick(dec_units, "dec_units", 64))
    context_dim     = int(_pick(context_dim, "context_dim", 32))
    lr              = float(_pick(lr, "learning_rate", 1e-3))
    batch_size      = int(_pick(batch_size, "batch_size", BATCH_SIZE))
    w_ekman         = float(_pick(w_ekman, "w_ekman", W_EKMAN_DEF))
    w_ekman_warmup  = int(_pick(w_ekman_warmup, "w_ekman_warmup",
                                W_EKMAN_WARMUP_DEF))
    w_bc            = float(_pick(w_bc, "w_bc", W_BC_DEF))
    w_ustar         = float(_pick(w_ustar, "w_ustar", 0.0))
    if config_json:
        print(f"  Config JSON: {config_json}")
        print(f"    -> n_past={n_past} lstm1={lstm1} lstm2={lstm2} "
              f"dec={dec_units} ctx={context_dim} lr={lr} bs={batch_size}")
        print(f"    -> w_ekman={w_ekman} warmup={w_ekman_warmup} w_bc={w_bc} "
              f"(explicit CLI flags override the JSON)")

    os.makedirs(out, exist_ok=True)
    run_seed = int(seed) if seed is not None else RANDOM_STATE
    if seed is not None:
        print(f"  Run seed: {run_seed} (data split stays at {RANDOM_STATE} "
              f"so paired comparisons across seeds remain valid)")
    tf.random.set_seed(run_seed); np.random.seed(run_seed)
    f_cor = 2 * OMEGA * np.sin(np.deg2rad(lat))

    print("=" * 74)
    print("  Ekman balance + Monin-Obukhov closure (u,v profiles)")
    print(f"  numpy {np.__version__} | TF {tf.__version__}")
    print(f"  lat={lat} deg -> f={f_cor:.3e} 1/s | z0={z0} d={d}")
    print(f"  w_ekman(target)={w_ekman} warmup={w_ekman_warmup}ep "
          f"w_bc={w_bc} a_cls={a_cls}")
    print(f"  anchor_mode={anchor_mode} (z_met={z_met} m) | "
          f"train_max_height={train_max_height}")
    print("=" * 74)

    data = load_npz(npz_file)

    print("\n  K-Means regime labels ...")
    labels = build_kmeans_labels(data, k_clusters, min_valid_alts)

    print("  Building (u,v) sequences ...")
    X, U, V, C, alts, UV_met = build_sequences(
        data, labels, n_past, min_valid_alts, return_extras=True)
    M, T_steps, F = X.shape
    N_alt = U.shape[1]
    print(f"  Samples: {M:,}  seq {X.shape[1:]}  N_alt {N_alt}")

    # clean physically implausible speeds
    spd = np.sqrt(U**2 + V**2)
    bad = (spd < WS_MIN) | (spd > WS_MAX)
    if bad.any():
        U[bad] = np.nan; V[bad] = np.nan
    # validity mask BEFORE imputation: entries that are genuine measurements.
    # The seen/unseen evaluation below uses only these, so imputed medians
    # never contaminate the extrapolation metrics.
    VMASK = np.isfinite(U) & np.isfinite(V)                  # (M, N_alt)
    for arr in (U, V):
        if np.isnan(arr).any():
            med = np.nanmedian(arr, axis=0)
            r, cc = np.where(np.isnan(arr)); arr[r, cc] = med[cc]

    uv = np.stack([U, V], axis=-1).astype(np.float32)        # (M,N_alt,2)
    uv_scale = float(np.std(uv) + 1e-6)
    Z_data = np.tile(alts.astype(np.float32), (M, 1))

    # vertical-extrapolation experiment: supervised-gates mask
    if train_max_height is not None:
        seen = alts <= float(train_max_height)
        if not seen.any():
            raise ValueError("train_max_height below the lowest gate")
        print(f"  L_data supervised at {int(seen.sum())}/{N_alt} gates "
              f"(<= {train_max_height} m); HELD OUT: "
              f"{[int(h) for h in alts[~seen]]}")
    else:
        seen = np.ones(N_alt, dtype=bool)

    # height-band validation
    # With --val-max-height, model selection (early stopping) is driven by
    # the extrapolation error on the band train_max_height < z <= val_max_height,
    # and the gates above val_max_height are the untouched test. This keeps
    # every design decision (lambda_Ekman, w_ustar, stopping point) selected
    # by out-of-band skill, never by the test gates.
    if val_max_height is not None:
        if train_max_height is None:
            raise ValueError("--val-max-height requires --train-max-height")
        band = (alts > float(train_max_height)) & (alts <= float(val_max_height))
        if not band.any():
            raise ValueError("no gates fall in the validation band")
        print(f"  Validation band: {[int(h) for h in alts[band]]} m | "
              f"TEST gates: {[int(h) for h in alts[~seen & ~band]]}")
    else:
        band = None

    # scale inputs
    Xf = X.reshape(M * T_steps, F)
    fin = np.all(np.isfinite(Xf), axis=1)
    scaler = StandardScaler().fit(Xf[fin])
    Xf_s = Xf.copy(); Xf_s[fin] = scaler.transform(Xf[fin])
    X_s = Xf_s.reshape(M, T_steps, F).astype(np.float32)

    # split (stratified on regime, OR temporal block split if requested)
    idx = np.arange(M)
    if temporal_split_file:
        _split = np.load(temporal_split_file)
        if _split["t_idx"].size != M:
            raise SystemExit(
                f"Temporal split file has {_split['t_idx'].size:,} sequences "
                f"but this run built {M:,}; the two pipelines are not aligned "
                f"(check n_past/min_valid_alts/k_clusters match), aborting "
                f"rather than silently using a misaligned split.")
        tr, va, te = _split["train_pos"], _split["val_pos"], _split["test_pos"]
        print(f"  Using TEMPORAL BLOCK split from {temporal_split_file}")
    else:
        tr, tmp = train_test_split(idx, test_size=0.3,
                                   random_state=RANDOM_STATE, stratify=C)
        va, te = train_test_split(tmp, test_size=0.5,
                                  random_state=RANDOM_STATE, stratify=C[tmp])
    print(f"  Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")

    # neutral log-law bulk estimate of u_star at the lowest positive seen
    # gate (observable data only; used by the optional --w-ustar anchor)
    pos_seen = seen & (alts > 0)
    z_anchor = float(alts[pos_seen].min())
    j_anchor = int(np.where(alts == z_anchor)[0][0])
    spd_anchor = np.sqrt(U[:, j_anchor] ** 2 + V[:, j_anchor] ** 2)
    U_BULK = (KAPPA * spd_anchor / np.log(z_anchor / z0)).astype(np.float32)
    if w_ustar > 0:
        print(f"  u_star anchor: z={z_anchor:.0f} m, "
              f"u*_bulk mean {U_BULK.mean():.3f} m/s (w_ustar={w_ustar})")

    def take(ix):
        return (tf.constant(X_s[ix]), tf.constant(Z_data[ix]),
                tf.constant(uv[ix]), tf.constant(C[ix]),
                tf.constant(UV_met[ix]), tf.constant(U_BULK[ix]))
    Xtr, Ztr, UVtr, Ctr, UVMtr, UBtr = take(tr)
    Xva, Zva, UVva, Cva, UVMva, UBva = take(va)
    Xte, Zte, UVte, Cte, UVMte, UBte = take(te)

    model = EkmanMOSTPINN(n_past, F, k_clusters, z0=z0, d=d,
                          lstm1=lstm1, lstm2=lstm2,
                          dec_units=dec_units, context_dim=context_dim)
    model.adapt_height(alts)
    _ = model((Xtr[:2], Ztr[:2], UVMtr[:2]))
    model.summary()

    trainer = PINNTrainer(model, keras.optimizers.Adam(lr),
                          w_ekman, w_bc, a_cls, f_cor,
                          float(min(alts)), float(max(alts)), uv_scale,
                          w_ekman_warmup=w_ekman_warmup,
                          data_mask=(seen if train_max_height is not None
                                     else None),
                          anchor_mode=anchor_mode, z_met=z_met,
                          n_colloc=n_colloc, w_ustar=w_ustar,
                          physics_type=physics_type)

    ds = (tf.data.Dataset.from_tensor_slices(
              (Xtr, Ztr, UVtr, Ctr, UVMtr, UBtr))
          .shuffle(8192, seed=run_seed).batch(batch_size))

    hist = {"loss": [], "data": [], "cls": [], "ekman": [], "bc": [],
            "val": [], "val_data": [], "w_ekman": []}
    best, best_w, wait, patience = np.inf, None, 0, 15
    UVva_np = UVva.numpy()
    print("\n  Training ...")
    for ep in range(1, epochs + 1):
        # advance the Ekman-weight curriculum for this epoch
        w_eff = trainer.set_epoch(ep)
        agg = np.zeros(5); nb = 0
        for xb, zb, uvb, cb, mb, ub in ds:
            l, ld, lc, le, lbc = trainer.train_step(xb, zb, uvb, cb, mb, ub)
            agg += [float(l), float(ld), float(lc), float(le), float(lbc)]; nb += 1
        agg /= max(nb, 1)
        uv_va = trainer.predict_uv(Xva, Zva).numpy()
        val_rmse = float(np.sqrt(np.mean((uv_va - UVva_np)**2)))
        # validation DATA loss: same scale-normalized MSE as training L_data,
        # this (not the total loss) is what early stopping monitors, so the
        # physics term cannot prematurely halt training before the profile
        # fit has converged. Restricted to the supervised gates so that the
        # held-out heights are never used for model selection either.
        _vsel = band if band is not None else seen
        val_data = float(np.mean(
            ((uv_va[:, _vsel, :] - UVva_np[:, _vsel, :]) / uv_scale) ** 2))
        for k, vv in zip(["loss", "data", "cls", "ekman", "bc"], agg):
            hist[k].append(vv)
        hist["val"].append(val_rmse)
        hist["val_data"].append(val_data)
        hist["w_ekman"].append(w_eff)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d} | loss {agg[0]:.4f} (data {agg[1]:.4f} "
                  f"cls {agg[2]:.4f} ekman {agg[3]:.4f} bc {agg[4]:.4f}) "
                  f"| w_ek {w_eff:.2e} | val RMSE(uv) {val_rmse:.3f} "
                  f"| val L_data {val_data:.4f}")
        # early stopping on validation L_data (not total loss)
        if val_data < best - 1e-5:
            best, best_w, wait = val_data, [w.numpy() for w in model.weights], 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop ep {ep} (best val L_data {best:.4f})"); break
    if best_w:
        for w, val in zip(model.weights, best_w):
            w.assign(val)

    # test evaluation in physical units (speed)
    uv_te = trainer.predict_uv(Xte, Zte).numpy()
    uv_true = UVte.numpy()
    spd_pred = np.sqrt(uv_te[..., 0]**2 + uv_te[..., 1]**2)
    spd_true = np.sqrt(uv_true[..., 0]**2 + uv_true[..., 1]**2)
    # direction (meteorological deg)
    dir_pred = (np.degrees(np.arctan2(-uv_te[..., 0], -uv_te[..., 1])) + 360) % 360
    dir_true = (np.degrees(np.arctan2(-uv_true[..., 0], -uv_true[..., 1])) + 360) % 360

    rmse_spd = np.sqrt(np.mean((spd_pred - spd_true)**2, axis=0))
    mae_spd = np.mean(np.abs(spd_pred - spd_true), axis=0)
    # circular direction error
    derr = np.abs(((dir_pred - dir_true + 180) % 360) - 180)
    mae_dir = np.mean(derr, axis=0)
    overall_rmse = float(np.sqrt(np.mean((spd_pred - spd_true)**2)))
    overall_mae = float(np.mean(np.abs(spd_pred - spd_true)))

    print(f"\n  Overall speed RMSE {overall_rmse:.3f} m/s  MAE {overall_mae:.3f} m/s")
    print("  Per-altitude:  height  RMSE_spd  MAE_spd  MAE_dir(deg)")
    for h, rs, ms, md, sn in zip(alts, rmse_spd, mae_spd, mae_dir, seen):
        tag = "" if sn else "   [HELD-OUT]"
        print(f"    {int(h):4d} m   {rs:6.3f}   {ms:6.3f}   {md:6.1f}{tag}")

    # seen / unseen summary on genuine measurements only
    # Uses the pre-imputation validity mask so imputed medians never enter
    # the extrapolation metrics.
    vmask_te = VMASK[te]                                     # (n_te, N_alt)
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
    print("\n  Seen/unseen gate summary (valid measurements only):")
    if res_band:
        print(f"    VBAND  ({res_band['heights']}): "
              f"RMSE {res_band['rmse_spd']:.3f}  MAE {res_band['mae_spd']:.3f}"
              f"  dirMAE {res_band['mae_dir']:.1f} deg  [selection band]")
    if res_seen:
        print(f"    SEEN   ({res_seen['heights']}): "
              f"RMSE {res_seen['rmse_spd']:.3f}  MAE {res_seen['mae_spd']:.3f}"
              f"  dirMAE {res_seen['mae_dir']:.1f} deg"
              f"  (n={res_seen['n_entries']:,})")
    if res_unseen:
        print(f"    UNSEEN ({res_unseen['heights']}): "
              f"RMSE {res_unseen['rmse_spd']:.3f}  MAE {res_unseen['mae_spd']:.3f}"
              f"  dirMAE {res_unseen['mae_dir']:.1f} deg"
              f"  (n={res_unseen['n_entries']:,})")
    # per-sample errors at held-out gates, for the paired Wilcoxon test
    # against the physics-off arm on the identical partition
    np.savez_compressed(
        os.path.join(out, "pinn_ekman_test_predictions.npz"),
        altitudes=alts, seen_mask=seen,
        band_mask=(band if band is not None else np.zeros_like(seen)),
        valid_mask=vmask_te,
        spd_pred=spd_pred.astype(np.float32),
        spd_true=spd_true.astype(np.float32),
        dir_pred=dir_pred.astype(np.float32),
        dir_true=dir_true.astype(np.float32))

    ctx = model.encode(Xte)
    u_star, inv_L, ug, alpha = model.physics_params(ctx)
    u_star = u_star.numpy().ravel(); inv_L = inv_L.numpy().ravel()
    print(f"\n  Inferred u_star mean {u_star.mean():.3f} m/s "
          f"[{u_star.min():.3f},{u_star.max():.3f}]")
    print(f"  Inferred alpha mean {alpha.numpy().mean():.3f} "
      f"[{alpha.numpy().min():.3f},{alpha.numpy().max():.3f}]")
    print(f"  Stable fraction {float(np.mean(inv_L>0))*100:.1f}%")

    # save
    model.save_weights(os.path.join(out, "pinn_ekman.weights.h5"))
    np.savez_compressed(os.path.join(out, "pinn_ekman_meta.npz"),
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        altitudes=alts, rmse_spd=rmse_spd, mae_spd=mae_spd,
                        mae_dir=mae_dir, uv_scale=uv_scale, f=f_cor, z0=z0, d=d)
    with open(os.path.join(out, "pinn_ekman_results.json"), "w") as fh:
        json.dump({"overall_rmse": overall_rmse, "overall_mae": overall_mae,
                   "rmse_spd": rmse_spd.tolist(), "mae_spd": mae_spd.tolist(),
                   "mae_dir": mae_dir.tolist(),
                   "altitudes": np.asarray(alts).tolist(),
                   "best_val_data": best, "u_star_mean": float(u_star.mean()),
                   "frac_stable": float(np.mean(inv_L > 0)),
                   "seen_gates": res_seen, "band_gates": res_band,
                   "unseen_gates": res_unseen,
                   "config": {"n_past": n_past, "epochs": epochs, "lat": lat,
                              "w_ekman": w_ekman, "w_ekman_warmup": w_ekman_warmup,
                              "w_bc": w_bc, "a_cls": a_cls,
                              "z0": z0, "d": d,
                              "train_max_height": train_max_height,
                              "val_max_height": val_max_height,
                              "w_ustar": w_ustar,
                              "anchor_mode": anchor_mode, "z_met": z_met}},
                  fh, indent=2)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ["loss", "data", "ekman"]:
        ax[0].plot(hist[k], label=k)
    ax[0].set(xlabel="Epoch", ylabel="Loss", title="PINN losses"); ax[0].legend()
    ax[1].plot(hist["val_data"], color="tab:red", label="val L_data (early-stop)")
    ax[1].plot(hist["val"], color="tab:gray", alpha=.6, label="val RMSE(uv)")
    axw = ax[1].twinx()
    axw.plot(hist["w_ekman"], color="tab:green", ls=":", label="w_ekman (curriculum)")
    axw.set_ylabel("w_ekman")
    ax[1].set(xlabel="Epoch", ylabel="Validation", title="Validation & curriculum")
    ax[1].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ekman_training.png"), dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 6))
    ax2[0].plot(rmse_spd, alts, "o-", label="RMSE"); ax2[0].plot(mae_spd, alts, "s--", label="MAE")
    ax2[0].set(xlabel="Speed error (m/s)", ylabel="Height (m)", title="Speed"); ax2[0].legend(); ax2[0].grid(alpha=.3)
    ax2[1].plot(mae_dir, alts, "o-", color="tab:purple")
    ax2[1].set(xlabel="Direction MAE (deg)", ylabel="Height (m)", title="Direction"); ax2[1].grid(alpha=.3)
    fig2.tight_layout(); fig2.savefig(os.path.join(out, "ekman_error_profile.png"), dpi=150)
    plt.close(fig2)

    print(f"\n  Outputs in: {os.path.abspath(out)}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ekman + Monin-Obukhov closure (u,v)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz_file", nargs="?", default=NPZ_FILE_DEFAULT)
    ap.add_argument("-o", "--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--n-past", type=int, default=None,
                    help=f"look-back window (default {N_PAST}, or the value "
                         f"from --config-json)")
    ap.add_argument("--min-valid-alts", type=int, default=MIN_VALID_ALTS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--w-ekman", type=float, default=None,
                    help=f"target Ekman physics-loss weight, reached after "
                         f"warmup (default {W_EKMAN_DEF}, or --config-json; "
                         f"use 0 for the physics-off arm — explicit CLI "
                         f"always overrides the JSON)")
    ap.add_argument("--temporal-split-file", type=str, default=None,
                    help="path to a .npz from build_temporal_block_split.py; "
                         "if given, uses a leakage-free temporal split instead "
                         "of the stratified random one")
    ap.add_argument("--physics-type", choices=["ekman", "shear", "none"],
                    default="ekman",
                    help="which physics residual to use when --w-ekman != 0: "
                         "'ekman' = genuine Ekman-MOST momentum balance "
                         "(Coriolis-coupled, uses u_star/inv_L/ug via Km); "
                         "'shear' = parameterised power-law shear constraint "
                         "(uses alpha only). 'none' zeroes the residual "
                         "regardless of --w-ekman (default: ekman).")
    ap.add_argument("--w-ekman-warmup", type=int, default=None,
                    help=f"epochs to linearly ramp the Ekman weight from 0 to "
                         f"target (default {W_EKMAN_WARMUP_DEF}, or --config-json)")
    ap.add_argument("--w-bc", type=float, default=None,
                    help=f"anchor-term weight (default {W_BC_DEF}, or --config-json)")
    ap.add_argument("--config-json", default=None,
                    help="path to ekman_tuning_summary.json or a "
                         "best_n_past_*/pinn_ekman_results.json; applies the "
                         "tuned n_past/architecture/lr/batch and loss weights "
                         "automatically (explicit CLI flags override it)")
    ap.add_argument("--lstm1", type=int, default=None,
                    help="encoder Bi-LSTM 1 units (default 128, or --config-json)")
    ap.add_argument("--lstm2", type=int, default=None,
                    help="encoder Bi-LSTM 2 units (default 64, or --config-json)")
    ap.add_argument("--dec-units", type=int, default=None,
                    help="decoder hidden units (default 64, or --config-json)")
    ap.add_argument("--context-dim", type=int, default=None,
                    help="context vector size (default 32, or --config-json)")
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam learning rate (default 1e-3, or --config-json)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"batch size (default {BATCH_SIZE}, or --config-json)")
    ap.add_argument("--a-cls", type=float, default=A_CLS_DEF)
    ap.add_argument("--lat", type=float, default=42.6, help="site latitude (deg)")
    ap.add_argument("--z0", type=float, default=Z0_DEF)
    ap.add_argument("--d", type=float, default=D_DEF)
    ap.add_argument("--k-clusters", type=int, default=K_CLUSTERS)
    ap.add_argument("--train-max-height", type=float, default=None,
                    help="if set, L_data (and early stopping) only use gates "
                         "<= this height; upper gates are held out and "
                         "reported separately (vertical extrapolation)")
    ap.add_argument("--anchor-mode", choices=["zero", "met", "none"],
                    default="zero",
                    help="'zero' = original no-slip anchor at z0+d; "
                         "'met' = anchor to the measured met-station (u,v) "
                         "at --z-met; 'none' = no anchor")
    ap.add_argument("--z-met", type=float, default=2.0,
                    help="met-station measurement height (m) for --anchor-mode met")
    ap.add_argument("--val-max-height", type=float, default=None,
                    help="if set (requires --train-max-height), early stopping "
                         "uses the extrapolation error on the band "
                         "(train_max_height, val_max_height]; gates above are "
                         "the untouched test and are reported separately")
    ap.add_argument("--gpu-mem", type=int, default=None,
                    help="explicit GPU memory limit in MB (bypasses TF's "
                         "free-memory autodetection; e.g. 7000)")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (weights init, shuffling). The data "
                         "split always uses 42, so test partitions are "
                         "identical across seeds")
    ap.add_argument("--w-ustar", type=float, default=None,
                    help="weight of the surface-layer u_star anchor "
                         "(u*_bulk from the lowest seen gate via neutral "
                         "log law). 0 = off (default). Prevents the "
                         "degenerate u*->0 minimum of the Ekman residual")
    ap.add_argument("--n-colloc", type=int, default=None,
                    help=f"collocation heights per step for the Ekman residual "
                         f"(default {N_COLLOCATION}); lower = cheaper physics "
                         f"loss with higher estimator variance. Use the SAME "
                         f"value in both arms")
    args = ap.parse_args()
    main(args.npz_file, args.output_dir, args.n_past, args.min_valid_alts,
         args.epochs, args.w_ekman, args.w_bc, args.a_cls, args.lat,
         args.z0, args.d, args.k_clusters, w_ekman_warmup=args.w_ekman_warmup,
         train_max_height=args.train_max_height,
         anchor_mode=args.anchor_mode, z_met=args.z_met,
         lstm1=args.lstm1, lstm2=args.lstm2, dec_units=args.dec_units,
         context_dim=args.context_dim, lr=args.lr,
         batch_size=args.batch_size, config_json=args.config_json,
         n_colloc=args.n_colloc, w_ustar=args.w_ustar,
         val_max_height=args.val_max_height, seed=args.seed,
         gpu_mem=args.gpu_mem, temporal_split_file=args.temporal_split_file,
         physics_type=args.physics_type)
