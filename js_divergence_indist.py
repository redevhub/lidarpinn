"""
Jensen-Shannon divergence between predicted and measured wind-speed
distributions, in-distribution (all heights, no held-out gates).

Uses saved predictions only. Supports two or three arms at once.

Usage:

    python3 js_divergence_indist.py \
        --arm "Ekman=out_pinn_indist_s*/pinn_ekman_test_predictions.npz" \
        --arm "Shear=out_shear_indist_s*/pinn_ekman_test_predictions.npz" \
        --arm "No physics=out_noek_indist_s*/pinn_ekman_test_predictions.npz"
"""
import argparse, glob
import numpy as np
from scipy.spatial.distance import jensenshannon

BINS = 60


def js_divergence(true_v, pred_v, bins=BINS):
    lo = float(min(np.nanmin(true_v), np.nanmin(pred_v)))
    hi = float(max(np.nanmax(true_v), np.nanmax(pred_v)))
    edges = np.linspace(lo, hi, bins + 1)
    pt, _ = np.histogram(true_v, bins=edges, density=True)
    pp, _ = np.histogram(pred_v, bins=edges, density=True)
    pt = pt / (pt.sum() + 1e-12); pp = pp / (pp.sum() + 1e-12)
    return float(jensenshannon(pt, pp, base=2) ** 2)


def arm_js(paths):
    out = []
    for p in paths:
        d = np.load(p)
        m = d["valid_mask"].astype(bool)
        out.append(js_divergence(d["spd_true"][m], d["spd_pred"][m]))
    return np.array(out)


def main(a):
    for spec in a.arm:
        name, patt = spec.split("=", 1)
        fs = sorted(glob.glob(patt))
        if not fs:
            print(f"  {name}: no files for {patt}")
            continue
        js = arm_js(fs)
        sd = js.std(ddof=1) if len(js) > 1 else 0.0
        print(f"  {name:<14} JS = {js.mean():.4f} +- {sd:.4f}   "
              f"(n={len(js)})  values: {[round(float(x),4) for x in js]}")
    print("\n  Lower is a closer match to the measured speed distribution.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True,
                   help="Name=glob pattern, repeatable, e.g. --arm 'Ekman=out_pinn_indist_s*/pinn_ekman_test_predictions.npz'")
    main(p.parse_args())
