"""

LiDAR Wind Temporal Spectra

Loads the .npz produced by lidar_parser.py (or the cluster_results.npz
that contains timestamps_valid) and plots the power spectral density (PSD)
of horizontal wind speed at three selected altitudes using Welch's method.

Three sub-figures are produced:
  1. PSD vs frequency  (log-log, one line per altitude)
  2. PSD vs period     (log-linear, more intuitive for meteorologists)
  3. Normalised cumulative energy vs frequency  (linear-linear)

A -5/3 Kolmogorov reference slope is overlaid on the PSD panel.

Outputs:

  wind_spectra.png   - three-panel figure saved to disk

Usage:

  python plot_wind_spectra.py                        # defaults below
  python plot_wind_spectra.py lidar_dataset.npz
  python plot_wind_spectra.py lidar_dataset.npz --altitudes 100 200 400
  python plot_wind_spectra.py lidar_dataset.npz --dt 600 --nperseg 512
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import welch

# CONFIGURATION  - edit before running

NPZ_FILE         = "lidar_dataset.npz"
OUTPUT_FIG       = "wind_spectra.pdf"

# Indices (0-based) of the three altitudes to analyse.
# Set to None to auto-pick low / mid / high levels.
ALT_INDICES      = None          # e.g. [0, 4, 8]  or  None

# Sampling interval in seconds.
# Set to None to infer from the timestamps array (recommended).
DT_SECONDS       = None         # e.g. 600 for 10-min data

# Welch parameters
NPERSEG          = 128           # samples per segment (power of 2 preferred)
NOVERLAP         = None          # None → Welch default (nperseg // 2)
DETREND          = "linear"      # "linear" removes trend per segment

CMAP             = "tab10"

# I/O

def load_npz(path):
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


# HELPERS

def infer_dt(timestamps):
    """
    Estimate median sampling interval [s] from a 1-D array of timestamps.
    Accepts numpy datetime64 or float (Unix seconds).
    """
    ts = np.asarray(timestamps)
    if np.issubdtype(ts.dtype, np.datetime64):
        ts_s = ts.astype("datetime64[s]").astype(np.float64)
    else:
        ts_s = ts.astype(np.float64)
    diffs = np.diff(ts_s)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Cannot infer dt: no positive time differences found.")
    dt = float(np.median(diffs))
    print(f"  Inferred dt = {dt:.1f} s  ({dt/60:.2f} min)")
    return dt


def pick_alt_indices(altitudes, n=3):
    """Auto-pick n evenly spaced altitude levels."""
    idx = np.round(np.linspace(1, len(altitudes) - 2, n)).astype(int)
    return idx.tolist()


def clean_series(arr):
    """
    Linearly interpolate over NaN values.
    If more than 50 % are NaN, returns None (unusable).
    """
    arr = arr.astype(np.float64)
    nan_frac = np.isnan(arr).mean()
    if nan_frac > 0.5:
        return None
    if nan_frac > 0:
        x = np.arange(len(arr))
        ok = ~np.isnan(arr)
        arr = np.interp(x, x[ok], arr[ok])
    return arr


# SPECTRAL COMPUTATION

def compute_spectra(ws, mask, alt_indices, dt, nperseg, noverlap, detrend):
    """
    Compute Welch PSD for each requested altitude.

    Parameters:
    
    ws         : (N_total, N_alt)  full wind-speed array
    mask       : (N_total,) bool   valid-profile mask (from clustering)
    alt_indices: list of int       altitude column indices
    dt         : float             sampling interval [s]

    Returns:
    
    list of dicts with keys: alt_idx, freqs, psd, label
    """
    results = []
    # Use only the valid (non-masked-out) profiles to preserve temporal order
    ws_valid = ws[mask]
    fs = 1.0 / dt

    for ai in alt_indices:
        series = clean_series(ws_valid[:, ai])
        if series is None:
            warnings.warn(f"  Altitude index {ai}: >50% NaN, skipping.")
            continue

        # Subtract mean (on top of per-segment detrending)
        series -= series.mean()

        seg = min(nperseg, len(series) // 4)
        if seg < 8:
            warnings.warn(f"  Altitude index {ai}: too few samples for Welch.")
            continue

        freqs, psd = welch(
            series, fs=fs,
            nperseg=seg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density",
        )

        # Drop zero-frequency bin
        nonzero = freqs > 0
        results.append(dict(
            alt_idx = ai,
            freqs   = freqs[nonzero],
            psd     = psd[nonzero],
        ))

    return results


# PLOTTING

def plot_spectra(spectra_list, altitudes, dt, out_path):
    """
    Three-panel figure:
      Left   : PSD vs frequency (log-log) with Kolmogorov -5/3 reference
      Centre : PSD vs period (log-linear)
      Right  : Cumulative energy fraction vs frequency (linear-linear)
    """
    n   = len(spectra_list)
    colors = [matplotlib.colormaps[CMAP](i) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_psd, ax_per, ax_cum = axes

    # Panel 1 – PSD vs frequency (log-log)
    
    for sp, color in zip(spectra_list, colors):
        alt_m = altitudes[sp["alt_idx"]]
        label = f"{alt_m:.0f} m"
        ax_psd.loglog(sp["freqs"], sp["psd"],
                      lw=1.6, color=color, label=label)

    # Kolmogorov -5/3 reference line
    f_ref   = np.array([spectra_list[0]["freqs"][5],
                        spectra_list[0]["freqs"][-5]])
    # Position the reference line near the median PSD value of the first series
    mid_psd  = np.median(spectra_list[0]["psd"])
    mid_freq = np.sqrt(f_ref[0] * f_ref[1])
    p_ref    = mid_psd * (f_ref / mid_freq) ** (-5 / 3)
    ax_psd.loglog(f_ref, p_ref, "k--", lw=1.2, alpha=0.7, label="−5/3 slope")

    ax_psd.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_psd.set_ylabel("PSD  [(m/s)² / Hz]", fontsize=12)
    ax_psd.set_title("Power spectral density", fontsize=12, fontweight="bold")
    ax_psd.legend(fontsize=12)
    ax_psd.grid(True, which="both", alpha=0.3)

    # Panel 2 – PSD vs period (log-linear, x-axis inverted)

    for sp, color in zip(spectra_list, colors):
        alt_m = altitudes[sp["alt_idx"]]
        period_h = 1.0 / (sp["freqs"] * 3600)   # convert Hz → hours
        ax_per.semilogx(period_h, sp["freqs"] *sp["psd"],
                        lw=1.6, color=color, label=f"{alt_m:.0f} m")

    # Reference period ticks
    ref_periods = [1/24, 1/12, 0.5, 1, 6, 12, 24, 72, 168]  # hours
    ref_labels  = ["1h",  "2h", "12h", "1d", "6d",
                   "12d", "1mo", "~3mo", "1wk"]
    # Only label ticks that fall within the plotted range
    f_min = spectra_list[0]["freqs"].min()
    f_max = spectra_list[0]["freqs"].max()
    p_min_h = 1 / (f_max * 3600)
    p_max_h = 1 / (f_min * 3600)
    visible = [(p, l) for p, l in zip(ref_periods, ref_labels)
               if p_min_h <= p <= p_max_h]
    if visible:
        ax_per.set_xticks([v[0] for v in visible])
        ax_per.set_xticklabels([v[1] for v in visible], fontsize=12, rotation=30)

    ax_per.set_ylabel(r"$f \cdot S(f)$  [(m/s)²]", fontsize=12)
    ax_per.set_title("Variance-preserving spectrum", fontsize=12, fontweight="bold")
    ax_per.set_title("PSD vs period", fontsize=12, fontweight="bold")
    ax_per.legend(fontsize=12)
    ax_per.grid(True, which="both", alpha=0.3)

    # Panel 3 – Normalised cumulative energy vs frequency

    for sp, color in zip(spectra_list, colors):
        alt_m   = altitudes[sp["alt_idx"]]
        df      = np.diff(sp["freqs"], prepend=sp["freqs"][0])
        energy  = sp["psd"] * df
        cum_e   = np.cumsum(energy)
        cum_e  /= cum_e[-1]          # normalise to [0, 1]
        ax_cum.semilogx(sp["freqs"], cum_e,
                        lw=1.8, color=color, label=f"{alt_m:.0f} m")

    ax_cum.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_cum.set_ylabel("Cumulative energy fraction", fontsize=12)
    ax_cum.set_title("Cumulative energy", fontsize=12, fontweight="bold")
    ax_cum.legend(fontsize=12)
    ax_cum.grid(True, which="both", alpha=0.3)
    ax_cum.set_ylim(0, 1.05)

    # Global styling

    total_dur_h = len(list(
        s for s in spectra_list if s is not None
    )[0]["freqs"]) / (1.0 / dt) / 3600 if spectra_list else 0

    fig.suptitle(
        f"Temporal wind-speed spectra  (Welch's method,  dt = {dt:.0f} s)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# MAIN

def main(npz_file=NPZ_FILE, output_fig=OUTPUT_FIG,
         alt_indices=ALT_INDICES, dt_seconds=DT_SECONDS,
         nperseg=NPERSEG, noverlap=NOVERLAP, detrend=DETREND):

    sep = "=" * 65
    print(f"\n{sep}")
    print("  LiDAR Wind Temporal Spectra")
    print(f"  Input  : {os.path.abspath(npz_file)}")
    print(sep)

    # ---- load ----
    print("  Loading dataset ...")
    data = load_npz(npz_file)
    alts = data["altitudes"]
    N_tot, N_alt = data["wind_speed"].shape
    print(f"  Total records  : {N_tot:,}")
    print(f"  Altitudes ({N_alt}) : {alts.tolist()} m")

    # Scrub sentinel values (consistent with clustering script)
    SENTINELS = [9999.0, 9998.0, -9999.0]
    TOL = 0.5
    ws = data["wind_speed"].astype(np.float64)
    for s in SENTINELS:
        ws[np.abs(ws - s) < TOL] = np.nan
    ws[(ws < 0) | (ws > 80)] = np.nan

    # valid-profile mask
    # Re-use the mask saved by the clustering script if available,
    # otherwise apply the same MIN_VALID_ALTS = 6 rule.
    if "valid_mask" in data:
        mask = data["valid_mask"].astype(bool)
        print(f"  Using stored valid_mask : {mask.sum():,} / {N_tot:,} profiles")
    else:
        min_valid = 6
        valid_per_row = np.sum(~np.isnan(ws), axis=1)
        mask = valid_per_row >= min_valid
        print(f"  Computed valid_mask     : {mask.sum():,} / {N_tot:,} profiles")

    # altitude selection
    if alt_indices is None:
        alt_indices = pick_alt_indices(alts, n=3)
    else:
        # Validate indices
        alt_indices = [int(i) for i in alt_indices]
        if any(i < 0 or i >= N_alt for i in alt_indices):
            raise ValueError(
                f"alt_indices {alt_indices} out of range [0, {N_alt - 1}]"
            )
    print(f"  Altitude indices : {alt_indices}")
    print(f"  Altitudes used   : {[float(alts[i]) for i in alt_indices]} m")

    # sampling interval
    if dt_seconds is None:
        ts_key = "timestamps_valid" if "timestamps_valid" in data else "timestamps"
        dt = infer_dt(data[ts_key])
    else:
        dt = float(dt_seconds)
        print(f"  Using supplied dt = {dt:.1f} s")

    # spectra
    print("\n  Computing Welch spectra ...")
    spectra = compute_spectra(
        ws, mask, alt_indices, dt,
        nperseg=nperseg, noverlap=noverlap, detrend=detrend
    )
    if not spectra:
        print("  ERROR: no valid spectra computed. Check alt_indices and data quality.")
        sys.exit(1)
    for sp in spectra:
        print(f"    alt index {sp['alt_idx']:2d}  "
              f"({alts[sp['alt_idx']]:.0f} m)  "
              f"nfreqs={len(sp['freqs'])}  "
              f"f_min={sp['freqs'][0]:.2e} Hz  "
              f"f_max={sp['freqs'][-1]:.2e} Hz")

    # plot
    print("\n  Generating figure ...")
    plot_spectra(spectra, alts, dt, output_fig)

    print(f"\n{sep}")
    print("  DONE")
    print(f"  Figure : {os.path.abspath(output_fig)}")
    print(sep)


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot temporal wind-speed spectra at three altitudes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npz_file", nargs="?", default=NPZ_FILE,
                        help="Path to lidar_dataset.npz (or cluster_results.npz)")
    parser.add_argument("--altitudes", nargs=3, type=int, metavar="IDX",
                        default=None,
                        help="Three 0-based altitude column indices, e.g. 0 4 8")
    parser.add_argument("--dt", type=float, default=None,
                        help="Sampling interval in seconds (inferred if omitted)")
    parser.add_argument("--nperseg", type=int, default=NPERSEG,
                        help="Welch segment length (samples)")
    parser.add_argument("--noverlap", type=int, default=None,
                        help="Welch overlap (samples, default = nperseg//2)")
    parser.add_argument("--no-detrend", action="store_true",
                        help="Disable per-segment linear detrending")
    parser.add_argument("-o", "--output", default=OUTPUT_FIG,
                        help="Output PNG filename")
    args = parser.parse_args()

    main(
        npz_file   = args.npz_file,
        output_fig = args.output,
        alt_indices= args.altitudes,
        dt_seconds = args.dt,
        nperseg    = args.nperseg,
        noverlap   = args.noverlap,
        detrend    = "constant" if args.no_detrend else DETREND,
    )
