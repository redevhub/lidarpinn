"""
Paired Wilcoxon signed-rank test between two arms, in-distribution (all
heights, all genuine measurements). Same statistical approach as
compare_extrap_arms.py, without the held-out-gate restriction.

Usage:

    python3 compare_indist_arms.py arm_a/pinn_ekman_test_predictions.npz \
                                    arm_b/pinn_ekman_test_predictions.npz
"""
import sys
import numpy as np
from scipy.stats import wilcoxon


def load(path):
    d = np.load(path)
    return d


def main(path_a, path_b):
    A = load(path_a)
    B = load(path_b)
    vm = A["valid_mask"].astype(bool) & B["valid_mask"].astype(bool)


    # speed
    spd_finite = np.isfinite(A["spd_pred"]) & np.isfinite(B["spd_pred"])
    vm_spd = vm & spd_finite
    n_dropped_spd = int(vm.sum() - vm_spd.sum())


    err_a = np.abs(A["spd_pred"] - A["spd_true"])[vm_spd]
    err_b = np.abs(B["spd_pred"] - B["spd_true"])[vm_spd]
    n = err_a.size
    stat, p = wilcoxon(err_a, err_b)
    better_a = float(np.mean(err_a < err_b)) * 100
 
    print(f"Paired Wilcoxon on per-sample speed MAE (in-distribution, n={n:,} samples"
         f"{f', {n_dropped_spd:,} entries dropped for a non-finite prediction' if n_dropped_spd else ''}):")
    print(f"    median MAE  {path_a} {np.median(err_a):.3f}  {path_b} {np.median(err_b):.3f}")
    print(f"    {path_a} better in {better_a:.1f}% of samples | W={stat:.0f}  p={p:.3e}")
 
    # direction
    dir_finite = np.isfinite(A["dir_pred"]) & np.isfinite(B["dir_pred"])
    vm_dir = vm & dir_finite
    n_dropped_dir = int(vm.sum() - vm_dir.sum())
 
    dir_a = np.abs(((A["dir_pred"] - A["dir_true"] + 180) % 360) - 180)[vm_dir]
    dir_b = np.abs(((B["dir_pred"] - B["dir_true"] + 180) % 360) - 180)[vm_dir]
    n_d = dir_a.size
    stat_d, p_d = wilcoxon(dir_a, dir_b)
    better_a_d = float(np.mean(dir_a < dir_b)) * 100
    print(f"\nPaired Wilcoxon on per-sample direction error (in-distribution, n={n_d:,} samples"
         f"{f', {n_dropped_dir:,} entries dropped for a non-finite prediction' if n_dropped_dir else ''}):")
    print(f"    median dir err  {path_a} {np.median(dir_a):.2f}  {path_b} {np.median(dir_b):.2f}")
    print(f"    {path_a} better in {better_a_d:.1f}% of samples | W={stat_d:.0f}  p={p_d:.3e}")
 
 
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_indist_arms.py A.npz B.npz")
    main(sys.argv[1], sys.argv[2])
 


