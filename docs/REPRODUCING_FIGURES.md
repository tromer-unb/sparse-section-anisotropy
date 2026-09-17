# Reproducing the manuscript figures

This page distinguishes **numerical experiment scripts** from **publication renderers** and records the execution checks performed against the archived analysis workspace.

## Environment

Create an isolated Python environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[test]'
```

The historical figure scripts require NumPy, SciPy, and Matplotlib. The real-rock download helper additionally uses the Python standard library plus a system `tar` executable; `curl` is used when available for resumable downloads.

## Reproduction status

The archived production scripts from the analysis workspace were syntax-checked together and each manuscript figure was executed independently on 2026-09-16.

| Figure | Execution status | Reproduced diagnostic values |
|---|---|---|
| 1 | PASS | Figure generated successfully |
| 2 | PASS | representative `A_2D`: isotropic XZ `1.0196`, anisotropic XZ `2.7271` |
| 3 | PASS | median blind axial error `1.150 deg`; p95 `3.101 deg`; median rotation-equivariance error `3.289%` |
| 4 | PASS | representative rank `6`; condition number `1.414`; tensor-shape error `13.75%`; triplet Spearman rho `0.441` |
| 5 | PASS | main median tensor-shape error `16.96%`; median major-axis error `8.47 deg` |
| 6 | PASS | Bentheimer `18.00 -> 6.32%`; Edwards Brown `38.84 -> 16.94%` from 3 to 21 sections |

These checks establish internal executable consistency of the archived figure code. They do not by themselves establish that every prose parameter in the manuscript is identical to every parameter in the final per-figure implementation.

## Important parameter audit

The manuscript methodology describes a broad synthetic protocol including larger benchmark volumes and larger realization counts. The final figure scripts use figure-specific executable settings. For example:

- Figure 2 uses a `96^3` synthetic volume and 20 ensemble realizations.
- Figure 3 uses a `256 x 256` generation canvas, a `160 x 160` analyzed crop, and 60 blind rotations.
- Figure 4 uses a `224^3` synthetic volume and 220 accepted subsets per section count.
- Figure 5 uses `128^3` main volumes with 12 realizations per anisotropy setting and a separate `160^3` sensitivity experiment with 10 realizations.

For auditability, this repository treats the executable script configuration as the provenance of each plotted result. Any manuscript statement intended to describe the exact plotted experiment should be reconciled with these settings before submission.

## Figure 6 pipeline

Figure 6 has three distinct stages:

```text
Figshare+ binary micro-CT RAW
          |
          +--> full-3D numerical reference estimation
          |
          +--> sparse 2D section measurement and reconstruction
          |
          v
final per-subvolume and summary CSV files
          |
          v
figure6 renderer + representative RAW slices
          |
          v
PNG / SVG / comparison summary / text report
```

The Figure-6 renderer is intentionally not the expensive validation program. The final Bentheimer production run uses 27 non-overlapping `512^3` regions, 67 reference directions, 50,000 Monte Carlo points per direction, 5 independent reference seeds, and 200 random sparse-section configurations per section count. Edwards Brown follows the same frozen acquisition/inversion protocol with explicit handling of invalid or rank-deficient sparse configurations.

## Deterministic outputs

Where a script contains stochastic generation or resampling, its fixed seed must remain visible in source control. Derived CSV files that are used as frozen publication inputs should be versioned when their size permits it. Large caches and raw micro-CT volumes should not be committed; use the manifest and downloader instead.

## Recommended release check

Before creating the publication tag:

```bash
pytest -q
python examples/infer_from_sections.py
```

Then reproduce Figures 1--5 from their scripts and render Figure 6 from the frozen real-rock CSVs. Compare the generated quantitative summaries with the values recorded above and in `docs/DATA_PROVENANCE.md`.