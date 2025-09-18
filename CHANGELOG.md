# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-08-29

This release captures the Ax cross-validation integration and plotting/export updates implemented in the Branin demonstration. See PR #9: "Implement Ax-platform cross-validation integration for iteration-by-iteration evaluation metrics with interval score and enhanced multi-axis visualization".

### Added
- Ax cross-validation metrics computed iteration-by-iteration in the Branin demo (`scripts/bo_benchmarks/demonstrations/branin_campaign_demonstration.py`).
  - Ax CV R² and Ax CV mean interval score (95% CI) per iteration.
  - Dedicated CV AxClient that mirrors all completed trials from the main AxClient each iteration (dedup by parameters) and forces a model refit (generate-then-abandon pattern).
  - Interval score computed from the Ax CV Plotly LOO figure’s error bars (interpreted as 95% CI half-widths).
- Per-iteration CV diagnostic plots exported to `scripts/bo_benchmarks/demonstrations/cv_plots/`:
  - Plotly interactive HTML (for inspection).
  - Matplotlib PNG parity plots (observed vs. predicted with error bars) derived from the Plotly trace data.
- Enhanced overall results figure with multi-axis metrics (GP R², Ax CV R², rank τ, LOO NLL, GP interval score, Ax CV interval score):
  - `scripts/bo_benchmarks/demonstrations/branin_campaign_demonstration_results.png`.

### Changed
- Prefer PNG over HTML for per-iteration CV figures; still save the Plotly HTML alongside PNGs. No timestamps in filenames.
- Organized CV outputs under `cv_plots/` for easier browsing.
- Increased Branin demo run length to 50 iterations to observe metric convergence behavior.

### Notes
- PNGs are generated via Matplotlib using Plotly trace data to avoid extra system dependencies for Plotly image export; the interactive Plotly HTML is still saved for reference.

