"""
Physical plausibility of the reconstructed profiles.

Pointwise error says how close a prediction is; it does not say whether the
predicted profile has a physically admissible shape. This script quantifies
shape, comparing each arm against the measured profiles:

  frac_dU_negative  fraction of vertical segments where wind speed decreases
                    with height (occasional in reality, pervasive if the
                    profile oscillates)
  roughness_speed   mean |d2U/dz2| normalised by mean speed: curvature of the
                    speed profile
  roughness_dir     mean |d2(theta)/dz2| after circular unwrapping: how jagged
                    the direction profile is
  total_turning     direction change from the lowest to the highest gate
                    (the veering), its mean and its spread across profiles

The measured profiles are the reference: a model is more physically plausible
the closer its statistic is to the measured one, not the smoothest.

Usage:

    python3 profile_physical_consistency.py \
        --pinn-glob "out_pinn_v2_s*/pinn_ekman_test_predictions.npz" \
        --noek-glob "out_noek_v2_s*/pinn_ekman_test_predictions.npz" \
        [--unseen-only]
"""
import glob, argparse
import numpy as np


def unwrap_deg(a, axis=-1):
    return np.degrees(np.unwrap(np.radians(a), axis=axis))


def shape_stats(spd, dirp, z):
    dU = np.diff(spd, axis=1) / np.diff(z)[None, :]
    du = unwrap_deg(dirp, axis=1)
    turn = du[:, -1] - du[:, 0]
    return {
        "frac_dU_negative": float((dU < 0).mean()),
        "roughness_speed":  float(np.mean(np.abs(np.diff(spd, n=2, axis=1))) / np.mean(spd)),
        "roughness_dir":    float(np.mean(np.abs(np.diff(du, n=2, axis=1)))),
        "turning_mean":     float(np.mean(turn)),
        "turning_sd":       float(np.std(turn)),
    }


def load_profiles(path, unseen_only):
    d = np.load(path)
    order = np.argsort(d["altitudes"])
    z = d["altitudes"][order]
    vm = d["valid_mask"][:, order].astype(bool)
    sel_h = np.ones_like(z, dtype=bool)
    if unseen_only:
        seen = d["seen_mask"][order].astype(bool)
        band = d["band_mask"][order].astype(bool) if "band_mask" in d.files else np.zeros_like(seen)
        sel_h = ~seen & ~band
        if sel_h.sum() < 3:
            raise SystemExit("need at least 3 held-out gates for second differences")
    ok = vm[:, sel_h].all(axis=1)
    return (d["spd_pred"][:, order][ok][:, sel_h], d["dir_pred"][:, order][ok][:, sel_h],
            d["spd_true"][:, order][ok][:, sel_h], d["dir_true"][:, order][ok][:, sel_h],
            z[sel_h])


def main(a):
    keys = ["frac_dU_negative", "roughness_speed", "roughness_dir", "turning_mean", "turning_sd"]
    rows = {}
    for name, patt in [("Ekman", a.pinn_glob), ("Physics-free", a.noek_glob)]:
        fs = sorted(glob.glob(patt))
        if not fs:
            raise SystemExit(f"no files for {name}: {patt}")
        st = []
        for f in fs:
            sp, dp, st_, dt_, z = load_profiles(f, a.unseen_only)
            st.append(shape_stats(sp, dp, z))
        rows[name] = {k: (np.mean([s[k] for s in st]),
                          np.std([s[k] for s in st], ddof=1) if len(st) > 1 else 0.0)
                      for k in keys}
    sp, dp, st_, dt_, z = load_profiles(sorted(glob.glob(a.pinn_glob))[0], a.unseen_only)
    meas = shape_stats(st_, dt_, z)

    scope = "held-out gates only" if a.unseen_only else "full profile"
    print(f"\n  Profile shape statistics ({scope})\n")
    print(f"  {'statistic':>18} {'measured':>10} {'Ekman':>18} {'Physics-free':>18}")
    print("  " + "-" * 68)
    for k in keys:
        p, ps = rows["Ekman"][k]; n, ns = rows["Physics-free"][k]
        print(f"  {k:>18} {meas[k]:10.4f} {p:11.4f}+-{ps:<5.4f} {n:11.4f}+-{ns:<5.4f}")
    print("\n  Closer to the measured column = more plausible shape.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pinn-glob", default="out_pinn_v2_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--noek-glob", default="out_noek_v2_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--unseen-only", action="store_true",
                   help="restrict to the held-out gates (needs >=3 of them)")
    main(p.parse_args())
