"""

Compare two arms of the vertical-extrapolation experiment

Loads the pinn_ekman_test_predictions.npz written by
lidar_pinn_ekman_v2_extrap.py (or its tuning script) for the PINN arm and
the physics-off arm, and reports speed/direction errors on the SEEN and
UNSEEN (held-out) gates, restricted to genuine (non-imputed) measurements,
plus a paired Wilcoxon signed-rank test on per-sample MAE at the unseen
gates. Both runs must come from the same dataset / n_past / RANDOM_STATE,
so the test samples are aligned one-to-one.

Usage:

  python3 compare_extrap_arms.py out_pinn_h150/pinn_ekman_test_predictions.npz \
                                 out_noek_h150/pinn_ekman_test_predictions.npz
"""

import os
import sys
import numpy as np


def load(path):
    d = np.load(path)
    need = ["altitudes", "seen_mask", "valid_mask",
            "spd_pred", "spd_true", "dir_pred", "dir_true"]
    missing = [k for k in need if k not in d.files]
    if missing:
        raise SystemExit(f"{path}: missing {missing} — was it produced by the "
                         f"*_extrap scripts with --train-max-height set?")
    out = {k: d[k] for k in need}
    out["band_mask"] = (d["band_mask"] if "band_mask" in d.files
                        else np.zeros_like(out["seen_mask"]))
    return out


def group_metrics(d, gate_sel):
    m = d["valid_mask"] & gate_sel[None, :]
    es = (d["spd_pred"] - d["spd_true"])[m]
    ed = np.abs(((d["dir_pred"] - d["dir_true"] + 180) % 360) - 180)[m]
    return (float(np.sqrt(np.mean(es**2))), float(np.mean(np.abs(es))),
            float(np.mean(ed)), int(m.sum()))


def per_sample_mae(d, gate_sel):
    """Per-test-sample speed MAE over valid gates in the selection.
    Samples with no valid gate in the group get NaN."""
    m = d["valid_mask"] & gate_sel[None, :]
    err = np.abs(d["spd_pred"] - d["spd_true"])
    s = np.where(m, err, 0.0).sum(axis=1)
    n = m.sum(axis=1)
    out = np.full(len(n), np.nan)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    return out


def main(path_pinn, path_off):
    A = load(path_pinn)   # arm A (e.g. Ekman)
    B = load(path_off)    # arm B (physics-off, log-law, ...)
    name_a = os.path.basename(os.path.dirname(path_pinn)) or "arm_A"
    name_b = os.path.basename(os.path.dirname(path_off)) or "arm_B"

    if A["spd_true"].shape != B["spd_true"].shape:
        raise SystemExit("Shape mismatch between arms — different n_past, "
                         "dataset or split. The comparison must be paired.")
    # joint validity: genuine measurements in BOTH files, and finite
    # predictions in BOTH (classical baselines emit NaN at z=0 or, in
    # extrapolation mode, at unsupervised gates). All metrics below are
    # computed on this common support, so the pairing is exact.
    joint = (A["valid_mask"].astype(bool) & B["valid_mask"].astype(bool)
             & np.isfinite(A["spd_pred"]) & np.isfinite(B["spd_pred"]))
    A["valid_mask"] = joint
    B["valid_mask"] = joint
    chk = joint & np.isfinite(A["spd_true"]) & np.isfinite(B["spd_true"])
    if not np.allclose(A["spd_true"][chk], B["spd_true"][chk],
                       rtol=1e-4, atol=1e-3):
        raise SystemExit("Test targets differ between arms on jointly-valid "
                         "entries — different split or dataset. Aborting "
                         "(comparison would be invalid).")
    if not np.array_equal(A["seen_mask"], B["seen_mask"]):
        raise SystemExit("seen_mask differs — the two arms were run with "
                         "different --train-max-height.")
    band_a = A["band_mask"].astype(bool)
    band_b = B["band_mask"].astype(bool)
    if band_a.any() and band_b.any() and not np.array_equal(band_a, band_b):
        raise SystemExit("band_mask differs — the two arms were run with "
                         "different --val-max-height.")
    # union: a gate used for model selection by EITHER arm is excluded from
    # the UNSEEN test (classical baselines carry no band)

    alts = A["altitudes"]; seen = A["seen_mask"].astype(bool)
    bandm = band_a | band_b
    print("=" * 70)
    print("  Vertical-extrapolation paired comparison")
    print(f"  supervised gates: {[int(h) for h in alts[seen]]}")
    if bandm.any():
        print(f"  selection band  : {[int(h) for h in alts[bandm]]}")
    print(f"  HELD-OUT gates  : {[int(h) for h in alts[~seen & ~bandm]]}")
    print("=" * 70)

    unseen_sel = ~seen & ~bandm
    for name, sel in [("SEEN", seen), ("UNSEEN", unseen_sel)]:
        if not sel.any():
            continue
        ra = group_metrics(A, sel); rb = group_metrics(B, sel)
        print(f"\n  {name} gates (n={ra[3]:,} valid entries)")
        print(f"    {'':18s} {'RMSE_spd':>9s} {'MAE_spd':>9s} {'MAE_dir':>9s}")
        print(f"    {name_a[:18]:18s} {ra[0]:9.3f} {ra[1]:9.3f} {ra[2]:9.1f}")
        print(f"    {name_b[:18]:18s} {rb[0]:9.3f} {rb[1]:9.3f} {rb[2]:9.1f}")
        print(f"    {'delta (A-B)':18s} {ra[0]-rb[0]:+9.3f} {ra[1]-rb[1]:+9.3f} "
              f"{ra[2]-rb[2]:+9.1f}")

    # paired Wilcoxon on per-sample MAE at the held-out gates
    if unseen_sel.any():
        a = per_sample_mae(A, unseen_sel)
        b = per_sample_mae(B, unseen_sel)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(a, b)
            better = float(np.mean(a < b)) * 100
            print(f"\n  Paired Wilcoxon on per-sample speed MAE (UNSEEN gates, "
                  f"n={ok.sum():,} samples):")
            print(f"    median MAE  {name_a} {np.median(a):.3f}  "
                  f"{name_b} {np.median(b):.3f}")
            print(f"    {name_a} better in {better:.1f}% of samples | "
                  f"W={stat:.0f}  p={p:.3e}")
        except ImportError:
            print("\n  scipy not available — install it for the Wilcoxon test "
                  "(pip install scipy)")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
