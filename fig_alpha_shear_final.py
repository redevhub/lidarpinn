"""
Distribution and diurnal cycle of the inferred shear exponent alpha, for the
shear-constrained arm.

Weights are loaded manually via h5py instead of model.load_weights(), because
the .h5 file was saved with tf_keras (inside a NGC Docker container) and
this script may run under standalone Keras 3 (e.g. a local Windows venv),
whose stricter structural loader rejects the file with errors like "Layer
expected 0 variables, but received N". Reading the .h5 directly and assigning
weights to each named sublayer by hand sidesteps that incompatibility.

Two Dense(32,1) layers (inv_L and alpha heads) are structurally identical and
cannot be told apart from the .h5 file alone; the script tries both
assignments and keeps the one whose alpha statistics match the known
reference (mean ~0.170, median ~0.103), printing which assignment was used.

Usage:

    python3 fig_alpha_shear_final.py lidar_dataset.npz \
        --model-dir out_shear_indist_s42 \
        --n-past 18 -o fig_alpha_shear
"""
import os, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lidar_pinn_ekman_v2_extrap as ek

RANDOM_STATE = 42
MIN_VALID_ALTS = 6
K_CLUSTERS = 3

LSTM1, LSTM2, DEC, CTX = 128, 128, 64, 64
W_EKMAN = 9.160265674307145e-4
W_BC = 0.05
A_CLS = 0.2
W_WARMUP = 20

# expected reference stats, used only to
# disambiguate the two structurally-identical (32,1) heads (inv_L vs alpha)
REF_ALPHA_MEAN = 0.170
REF_ALPHA_MEDIAN = 0.103
REF_TOL = 0.05


def build_test_set(npz_file, n_past, min_valid_alts, k_clusters):
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
    hours = times.astype("datetime64[h]").astype(int) % 24

    M, T_steps, F = X.shape
    spd = np.sqrt(U ** 2 + V ** 2)
    bad = (spd < ek.WS_MIN) | (spd > ek.WS_MAX)
    if bad.any():
        U[bad] = np.nan; V[bad] = np.nan
    for arr in (U, V):
        if np.isnan(arr).any():
            med = np.nanmedian(arr, axis=0)
            r, cc = np.where(np.isnan(arr)); arr[r, cc] = med[cc]
    uv = np.stack([U, V], axis=-1).astype(np.float32)
    uv_scale = float(np.std(uv) + 1e-6)

    Xf = X.reshape(M * T_steps, F)
    fin = np.all(np.isfinite(Xf), axis=1)
    scaler = StandardScaler().fit(Xf[fin])
    Xf_s = Xf.copy(); Xf_s[fin] = scaler.transform(Xf[fin])
    X_s = Xf_s.reshape(M, T_steps, F).astype(np.float32)

    idx = np.arange(M)
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, stratify=C)
    va, te = train_test_split(tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=C[tmp])

    return dict(X_te=X_s[te], uv_te=uv[te], uv_met_te=UV_met[te], C_te=C[te],
               hours_te=hours[te], F=F, alts=alts, uv_scale=uv_scale)


def build_model(d, lat):
    """Build the model and every sublayer via one real training step, without
    loading any weights (they are assigned manually afterwards)."""
    model = ek.EkmanMOSTPINN(18, d["F"], 3,
                             lstm1=LSTM1, lstm2=LSTM2,
                             dec_units=DEC, context_dim=CTX)
    model.adapt_height(d["alts"])

    f_cor = 2 * ek.OMEGA * np.sin(np.deg2rad(lat))
    trainer = ek.PINNTrainer(
        model, keras.optimizers.Adam(5e-4),
        W_EKMAN, W_BC, A_CLS, f_cor,
        float(min(d["alts"])), float(max(d["alts"])), d["uv_scale"],
        w_ekman_warmup=W_WARMUP, anchor_mode="met", z_met=2.0)

    n_dummy = 4
    Xd = tf.constant(d["X_te"][:n_dummy])
    Zd = tf.constant(np.tile(d["alts"].astype(np.float32), (n_dummy, 1)))
    UVd = tf.constant(d["uv_te"][:n_dummy])
    Cd = tf.constant(d["C_te"][:n_dummy].astype(np.int32))
    UVMd = tf.constant(d["uv_met_te"][:n_dummy])
    trainer.set_epoch(1)
    trainer.train_step(Xd, Zd, UVd, Cd, UVMd, None)
    return model


def load_weights_manual(model, h5_path, swap_invL_alpha=False):
    """Assign weights read directly from the .h5 to each named sublayer,
    bypassing Keras's own (version-sensitive) structural loader."""
    with h5py.File(h5_path, "r") as f:
        def g(path):
            """For sublayers nested under the top-level 'layers' group
            (bidirectional, bidirectional_1, dense, dense_1..dense_6)."""
            grp = f["layers/" + path + "/vars"]
            return [np.asarray(grp[str(i)]) for i in range(len(grp))]

        def g_root(path):
            """For sublayers saved at the file root (ln, phys1, ustar_out,
            ug_out), not nested under 'layers'."""
            grp = f[path + "/vars"]
            return [np.asarray(grp[str(i)]) for i in range(len(grp))]

        # encoder: two bidirectional LSTM layers + layernorm + context dense
        def lstm_weights(path):
            fw = f[f"layers/{path}/forward_layer/cell/vars"]
            bw = f[f"layers/{path}/backward_layer/cell/vars"]
            fwd = [np.asarray(fw[str(i)]) for i in range(len(fw))]
            bwd = [np.asarray(bw[str(i)]) for i in range(len(bw))]
            return fwd, bwd

        fwd1, bwd1 = lstm_weights("bidirectional")
        model.enc1.forward_layer.set_weights(fwd1)
        model.enc1.backward_layer.set_weights(bwd1)

        fwd2, bwd2 = lstm_weights("bidirectional_1")
        model.enc2.forward_layer.set_weights(fwd2)
        model.enc2.backward_layer.set_weights(bwd2)

        model.ln.set_weights(g_root("ln"))

        # context projection dense: the only Dense with input dim 256 taking
        # the encoder output; identified by weight shape, not by h5 name
        for cand in ["dense", "dense_1", "dense_2", "dense_3", "dense_4",
                    "dense_5", "dense_6"]:
            w = g(cand)
            if w[0].shape == (256, CTX):
                model.enc_dense.set_weights(w)
                context_dense_name = cand
                break
        else:
            raise RuntimeError("could not find the (256,CTX) context-projection Dense")

        # decoder: dec1 (CTX+1 -> DEC), dec2 (DEC -> DEC), dec_out (DEC -> 2)
        for cand in ["dense", "dense_1", "dense_2", "dense_3", "dense_4",
                    "dense_5", "dense_6"]:
            w = g(cand)
            if w[0].shape == (CTX + 1, DEC):
                model.dec1.set_weights(w)
            elif w[0].shape == (DEC, DEC):
                model.dec2.set_weights(w)
            elif w[0].shape == (DEC, 2):
                model.dec_out.set_weights(w)
            elif w[0].shape == (CTX, 3):
                model.cls_out.set_weights(w)

        # physics head
        model.phys1.set_weights(g_root("phys1"))
        model.ustar_out.set_weights(g_root("ustar_out"))
        model.ug_out.set_weights(g_root("ug_out"))

        # inv_L and alpha: both Dense(32,1), structurally identical in the
        # .h5. dense_1 and dense_2 are the two unnamed (32,1) heads.
        w1 = g("dense_1")
        w2 = g("dense_2")
        if w1[0].shape != (32, 1) or w2[0].shape != (32, 1):
            raise RuntimeError(f"unexpected shapes for dense_1/dense_2: "
                               f"{w1[0].shape}, {w2[0].shape}")
        if not swap_invL_alpha:
            model.invL_out.set_weights(w1)
            model.alpha_out.set_weights(w2)
            assignment = "dense_1->invL_out, dense_2->alpha_out"
        else:
            model.invL_out.set_weights(w2)
            model.alpha_out.set_weights(w1)
            assignment = "dense_2->invL_out, dense_1->alpha_out"
        return assignment


def main(a):
    d = build_test_set(a.npz_file, a.n_past, MIN_VALID_ALTS, K_CLUSTERS)
    model = build_model(d, a.lat)
    h5_path = os.path.join(a.model_dir, "pinn_ekman.weights.h5")

    # try the default assignment first, verify against known reference stats,
    # swap and retry if the check fails
    for attempt, swap in enumerate([False, True]):
        assignment = load_weights_manual(model, h5_path, swap_invL_alpha=swap)
        ctx = model.encode(tf.constant(d["X_te"][:2000]))  # small probe batch
        _, _, _, alpha_probe = model.physics_params(ctx)
        m = float(alpha_probe.numpy().mean())
        med = float(np.median(alpha_probe.numpy()))
        ok = abs(m - REF_ALPHA_MEAN) < REF_TOL and abs(med - REF_ALPHA_MEDIAN) < REF_TOL
        print(f"  attempt {attempt+1} [{assignment}]: alpha probe mean={m:.3f} "
             f"median={med:.3f}  {'OK' if ok else 'does not match reference, retrying'}")
        if ok:
            break
    else:
        print("  Warning: neither assignment matched the reference stats; "
             "results below may have inv_L and alpha swapped. Verify manually.")

    ctx = model.encode(tf.constant(d["X_te"]))
    u_star, inv_L, ug, alpha = model.physics_params(ctx)
    alpha = alpha.numpy().ravel()
    hours = d["hours_te"]

    print(f"  alpha: mean {alpha.mean():.3f}  median {np.median(alpha):.3f}  "
          f"[{alpha.min():.3f}, {alpha.max():.3f}]  n={alpha.size:,}")
    near_cap = float(np.mean(alpha > 0.599)) * 100
    near_floor = float(np.mean(alpha < 0.001)) * 100
    print(f"  fraction near upper bound (>0.599): {near_cap:.1f}%")
    print(f"  fraction near lower bound (<0.001): {near_floor:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(alpha, bins=60, color="#2a9d8f")
    ax.axvline(np.median(alpha), color="k", ls="--", lw=1.3)
    ax.set(xlabel="Inferred shear exponent $\\alpha$", ylabel="Count",
          title="Distribution of inferred shear exponent")
    ax.text(0.98, 0.95, f"median {np.median(alpha):.3f}", transform=ax.transAxes,
           ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + "_hist.png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + "_hist.pdf", bbox_inches="tight")
    plt.close(fig)

    hrs = sorted(set(hours.tolist()))
    means = [float(np.mean(alpha[hours == h])) for h in hrs]
    for h, m in zip(hrs, means):
        print(f"  hour {h:02d}: alpha_mean = {m:.4f}")
    min_i = int(np.argmin(means)); max_i = int(np.argmax(means))
    print(f"  min at hour {hrs[min_i]:02d} = {means[min_i]:.4f}")
    print(f"  max at hour {hrs[max_i]:02d} = {means[max_i]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(hrs, means, "o-", color="#2a9d8f", ms=4, lw=1.8)
    ax.set(xlabel="Local hour", ylabel="Mean inferred $\\alpha$",
          title="Diurnal cycle of inferred shear exponent")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(a.out + "_diurnal.png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + "_diurnal.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  written {a.out}_hist.png/.pdf and {a.out}_diurnal.png/.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz_file", nargs="?", default="lidar_dataset.npz")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--n-past", type=int, default=18)
    p.add_argument("--lat", type=float, default=42.615)
    p.add_argument("-o", "--out", default="fig_alpha_shear")
    main(p.parse_args())
