# Physics-informed reconstruction of the vertical wind profile from a single near-surface LiDAR measurement

Code accompanying the paper *[Physics-Informed Neural Networks for Multi-Location ABL
Wind Profiling with Minimal Instrumentation / DOI to be added upon publication]*, submitted to *Remote Sensing* (MDPI).

This repository trains and evaluates three architecture-matched neural networks that reconstruct the full Atmospheric Boundary Layer wind profile (speed and direction, 0-300 m) from a short history of ground-level measurements:

- an **Ekman-constrained** model, embedding the boundary-layer momentum balance closed with a Monin-Obukhov eddy viscosity;
- a **shear-constrained** model, embedding a power-law shear relation with a per-sample inferred exponent;
- a **physics-free** model of identical architecture, with neither physical term active.

All three share the same architecture (selected by Bayesian optimisation for the Ekman-constrained arm and reused unchanged for the other two), so that the effect of each physical constraint can be isolated from that of the network itself. Comparisons are also drawn against four fixed classical/tree-based baselines (logarithmic, power-law and constant wind profiles; random forest).

## Data availability

Raw LiDAR data are not included in this repository. Contact the corresponding author for data access requests.

## Repository structure
```
lidar_pinn_ekman_v2_extrap.py          Model definition, training loop (custom, with the
                                        physics residual evaluated by automatic differentiation),
                                        and the extrapolation protocol.
physics_shear_powerlaw.py              Shear-constrained residual (imported by the file above).
lidar_pinn_ekman_v2_extrap_tuning.py   Bayesian hyperparameter search (Keras Tuner) for the
                                        Ekman-constrained arm; selects on validation-band data
                                        loss, not on how strongly the physics residual is
                                        satisfied.
lidar_baselines_condireccion_extrap.py Classical/tree-based baselines (log law, power law,
                                        constant, random forest).

  Partition alignment against the neural-model arms (identical valid_mask and
  spd_true on the common test partition) was verified for the runs used in
  this paper; the paired Wilcoxon test against these baselines is valid for
  those files. If baselines or model runs are regenerated, re-verify
  alignment (matching valid_mask and spd_true arrays) before trusting any
  paired significance test, since a mismatch in --n-past, --min-valid-alts
  or --k-clusters between runs silently breaks it.

lidar_clustering.py                    K-Means regime clustering (PCA + K-Means/Agglomerative)
                                        used to stratify the train/val/test split.
plot_wind_spectra.py                   Welch power spectral density of the wind-speed time series.
```
Analysis / comparison:
```
compare_extrap_arms.py                 Paired Wilcoxon test between two arms, held-out gates only.
compare_indist_arms.py                 Paired Wilcoxon test between two arms, in-distribution.

  Both scripts assume the two compared arms share the identical test
  partition (verified for the runs used in this paper against every
  classical baseline; re-verify if any run is regenerated). compare_indist_arms.py
  additionally excludes any entry where either arm's prediction is
  non-finite, independently for the speed and direction test, required for
  a valid comparison against the logarithmic and power-law baselines, whose
  speed prediction is undefined at the surface gate (z = 0 m) even though a
  genuine measurement exists there.

js_divergence_extrap.py                Jensen-Shannon divergence, held-out gates (prints the
                                        exact per-seed values used in the paper text).
js_divergence_indist.py                Jensen-Shannon divergence, in-distribution.
profile_physical_consistency.py        Profile-shape statistics (curvature, jaggedness, turning)
                                        against the measured profiles, two arms at a time.
breakdown_extrap_by_regime.py          Held-out error broken down by K-Means regime, up to
                                        three arms.
analyze_error_breakdown_three_arms.py  In-distribution error by local hour, season and regime.
trainsize_sweep_three_arms.py          Training-set size sweep (contiguous months / distributed
                                        days-per-month), all three arms, fixed architecture.
```
Figures:
```
fig_extrapolation_profiles.py          Per-altitude RMSE/MAE/direction error, extrapolation.
fig_extrap_metrics.py                  JS divergence, profile shape and regime breakdown panels,
                                        extrapolation (three arms + classical baselines).
fig_indist_by_altitude.py              Per-altitude error, in-distribution (three arms +
                                        classical baselines).
fig_indist_overall.py                  Aggregate RMSE bars + measured/predicted speed histograms
                                        with JS divergence, in-distribution.
fig_indist_rmse_reduction.py           Per-altitude RMSE reduction of the best arm relative to
                                        each classical baseline.
fig_trainsize_three_arms.py            Training-set-size sweep curves, three arms.
fig_alpha_shear_final.py               Distribution and diurnal cycle of the inferred shear
                                        exponent alpha (loads model weights manually via h5py;
                                        see the note below).
plot_timeseries_by_regime.py           Example time-series windows by K-Means regime.
```

## Environment

```bash
pip install -r requirements.txt
```

Models were trained inside the NVIDIA NGC container `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (TensorFlow 2.17, CUDA/cuDNN built for NVIDIA Blackwell GPUs). Two environment-specific notes:

- If training raises a cuDNN RNN kernel error (`Failed to call DoRnnForward`), set `export PINN_LSTM_NO_CUDNN=1` before running; this forces the LSTM layers to unroll instead of using the fused cuDNN kernel.
- `fig_alpha_shear_final.py` loads saved model weights directly from the `.h5` file via `h5py` rather than `model.load_weights()`, because the standard Keras loader can reject weights saved under `tf_keras` (inside the container) when run under a different Keras 3 installation (e.g. a local, non-containerised environment). If you retrain the models in a single consistent environment, `model.load_weights()` can be used directly instead.

## Reproducing the main results

1. Train the Ekman-constrained arm with the hyperparameters in `lidar_pinn_ekman_v2_extrap_tuning.py`'s output (or rerun the search).
2. Train the shear-constrained and physics-free arms with the identical architecture, varying only `--w-ekman` and the physics term active in `lidar_pinn_ekman_v2_extrap.py`.
3. Run the classical baselines with `lidar_baselines_condireccion_extrap.py`.
4. Use the analysis and figure scripts above, pointing `--pinn-glob` / `--noek-glob` / `--shear-glob` (or equivalent flags) at the seed-wise output folders of steps 1-3.

Each script's docstring documents its exact command-line usage.

## Citation

If you use this code, please cite:

Echeverría, R. et al. *Physics-Informed Neural Networks for Multi-Location ABL Wind Profiling with Minimal Instrumentation*. Remote Sensing (submitted). DOI to be added upon publication.

and the archived code release:

Rubén Echeverría. (2026). redevhub/lidarpinn: Code release for manuscript submission (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21897456

## License

See `LICENSE`.
