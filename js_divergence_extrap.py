"""
Jensen-Shannon divergence between predicted and measured wind-speed
distributions on the held-out gates.

The in-distribution comparison already reports JS divergence as a metric that
matters independently of pointwise error: wind-resource and energy-yield
estimates depend on the speed histogram, not on the time-matched error. This
script computes the same quantity in the vertical-extrapolation setting, for each arm and each seed.

Uses saved predictions only. Restricted to the held-out gates and to
genuine (non-imputed) measurements, matching the convention used for the other
extrapolation metrics.

Usage:

    python3 js_divergence_extrap.py \
        --pinn-glob "out_pinnB_s*/pinn_ekman_test_predictions.npz" \
        --noek-glob "out_noekB_s*/pinn_ekman_test_predictions.npz"
"""
import argparse, glob
import numpy as np
from scipy.spatial.distance import jensenshannon

BINS = 60


def js_divergence(true_v, pred_v, bins=BINS):
    """Squared Jensen-Shannon distance (i.e. the divergence, base 2) between
    the two speed histograms over a common support."""
    lo = float(min(np.nanmin(true_v), np.nanmin(pred_v)))
    hi = float(max(np.nanmax(true_v), np.nanmax(pred_v)))
    edges = np.linspace(lo, hi, bins + 1)
    pt, _ = np.histogram(true_v, bins=edges, density=True)
    pp, _ = np.histogram(pred_v, bins=edges, density=True)
    pt = pt / (pt.sum() + 1e-12)
    pp = pp / (pp.sum() + 1e-12)
    return float(jensenshannon(pt, pp, base=2) ** 2)


def arm_js(paths):
    out = []
    for p in paths:
        d = np.load(p)
        seen = d["seen_mask"].astype(bool)
        band = (d["band_mask"].astype(bool) if "band_mask" in d.files
                else np.zeros_like(seen))
        unseen = ~seen & ~band
        m = d["valid_mask"].astype(bool) & unseen[None, :]
        out.append(js_divergence(d["spd_true"][m], d["spd_pred"][m]))
    return np.array(out)


def main(a):
    for name, patt in [("Ekman", a.pinn_glob),
                       ("Physics-off", a.noek_glob)]:
        fs = sorted(glob.glob(patt))
        if not fs:
            print(f"  {name}: no files for {patt}")
            continue
        js = arm_js(fs)
        sd = js.std(ddof=1) if len(js) > 1 else 0.0
        print(f"  {name:<16} JS = {js.mean():.4f} +- {sd:.4f}   "
              f"(n={len(js)})  values: {[round(float(x),4) for x in js]}")
    print("\n  Lower is a closer match to the measured speed distribution.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pinn-glob", default="out_pinnB_s*/pinn_ekman_test_predictions.npz")
    p.add_argument("--noek-glob", default="out_noekB_s*/pinn_ekman_test_predictions.npz")
    main(p.parse_args())
