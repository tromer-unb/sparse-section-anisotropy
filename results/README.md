# Frozen manuscript results

This directory stores lightweight derived tables that connect the numerical experiments to the publication figures.

They are committed intentionally so a reviewer can inspect or re-render the figures without first repeating every expensive simulation or real-rock reference calculation.

## Contents

- `figure3/figure3_blind_rotation_results.csv`: blind-angle recovery cases.
- `figure4/figure4_q3d_recovery_results.csv`: section-count / conditioning benchmark.
- `figure5/figure5_robustness_results.csv`: porosity, anisotropy, field-of-view, angular-sampling and interpolation benchmark.
- `figure6/`: final Bentheimer and Edwards Brown subvolume/summary tables plus the comparison summary/report.

## Policy

These files are **frozen outputs**, not raw source data. Their upstream generation scripts are preserved under `scripts/figures/` and `scripts/validation/`.

For Figure 6, the raw third-party micro-CT volumes are intentionally excluded from Git because each decompressed ROI is 15,625,000,000 bytes. Exact filenames, SHA-256 hashes, DOI, dimensions and phase convention are recorded in `data/manifest.csv` and `docs/DATA_PROVENANCE.md`.
