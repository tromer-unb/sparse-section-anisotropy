# Data provenance

This document records how the manuscript data flow through the analysis. It is intentionally explicit about the distinction between synthetic benchmarks, third-party real-rock volumes, heavy numerical validation, and lightweight publication plotting.

## Provenance levels

1. **Raw source data** — original binary micro-CT volumes or procedurally generated random fields.
2. **Measurement stage** — directional autocorrelation curves and first-`1/e` correlation lengths.
3. **Inference stage** — 2D or 3D SPD tensor estimates, observability metrics, reconstruction errors, and uncertainty summaries.
4. **Derived tables** — CSV files containing the frozen outputs needed for publication figures.
5. **Publication rendering** — scripts that read frozen tables and generate PNG/SVG figures.

A publication plotting script must not be mistaken for the script that generated its numerical inputs.

## Synthetic figures

### Figure 2 — in-plane detectability

The production script generates thresholded anisotropic Gaussian random fields directly in Python. Representative settings in the archived script are `N=96`, porosity `0.27`, isotropic Gaussian-filter scales `(4.8, 4.8, 4.8)`, anisotropic scales `(4.8, 4.8, 1.8)`, a 5-degree directional increment, and 20 independent ensemble realizations. No external rock dataset is used.

### Figure 3 — blind orientation recovery

The production script generates 2D thresholded anisotropic Gaussian random fields, rotates them by unknown angles, measures directional correlation lengths, and fits the 2D tensor. Archived settings include a `256 x 256` parent canvas, a `160 x 160` analyzed crop, 60 independent blind-angle cases, and deterministic random seeds. The generated CSV is `figure3_blind_rotation_results.csv`.

### Figure 4 — 3D observability and conditioning

The production script generates a stationary 3D Gaussian random field in Fourier space using an ellipsoidal spectral filter. Archived settings include `N=224`, analyzed section width `144`, porosity `0.27`, principal synthetic scales `[7.0, 4.2, 2.3]`, and a non-axis-aligned principal frame. The script then extracts oriented sections, assembles the six-parameter inverse problem, and writes `figure4_q3d_recovery_results.csv`.

### Figure 5 — robustness benchmark

The production script again generates anisotropic Gaussian random fields in Fourier space and thresholds them to prescribed porosity. The archived final script uses a main cube size of `128`, porosities `0.15, 0.25, 0.35`, target anisotropy ratios `1.0, 1.5, 2.0, 3.0, 4.0`, 12 main realizations, plus a separate `160^3` sensitivity benchmark with 10 realizations. It varies field of view, angular increment, and section interpolation. The frozen output is `figure5_robustness_results.csv`.

These per-figure settings are the executable provenance of the archived plotting/benchmark scripts. They should be kept distinct from broader parameter ranges described in the manuscript methodology.

## Figure 6 — real-rock validation

Figure 6 uses **real segmented micro-CT data**, not procedurally generated synthetic rock volumes.

### Local raw files used by the production workflow

- Bentheimer sandstone: `kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw`
- Edwards Brown carbonate: `edb-1_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw`

The local dataset metadata identifies these as segmented binary cubes produced from filtered grayscale micro-CT cubes using a three-class Multi-Otsu segmentation workflow. The production scripts interpret the files as `uint8`, shape `(2500, 2500, 2500)`, at `2.25 micrometre` voxel spacing, with `0 = pore` and `1 = solid` in the validated analysis convention.

### Spatial sampling

Each rock is divided into a `3 x 3 x 3` grid of 27 non-overlapping `512^3` subvolumes. The archived Bentheimer production code uses grid centers `(384, 1250, 2116)` along each computational axis.

### Frozen Bentheimer reference protocol

The final Bentheimer validation script freezes the reference estimator after earlier convergence tests:

- 64 Fibonacci-hemisphere directions plus the three Cartesian axes: 67 total directions;
- 50,000 Monte Carlo points per direction;
- 5 independent random seeds;
- the reference tensor is the SPD-projected Euclidean mean of the five seed tensors;
- maximum lag: 48 voxels;
- sparse 2D angular increment: 5 degrees;
- section counts: 1, 3, 5, and 7 per orientation, corresponding to 3, 9, 15, and 21 total sections;
- 200 random offset configurations at each section count;
- reproducibility seed: `20260907`.

The full-volume tensor is therefore a **numerical reference estimate**, not an analytic ground truth.

### Figure-6 plotting stage

`figure6.py` is deliberately lightweight. It does **not** recompute the 3D reference, directional autocorrelation functions, section measurements, or Monte Carlo reconstructions. It reads the final per-subvolume and summary CSV files and extracts one representative raw slice per rock for the image panel.

Quantitative input tables:

- `bentheimer_final_subvolumes.csv`
- `bentheimer_final_summary.csv`
- `edb1_final_subvolumes.csv`
- `edb1_final_summary.csv`

The final comparison table is `figure6_real_rock_comparison_summary.csv`.

### Reported frozen outputs

The archived final comparison reports the following spatial median tensor-shape errors:

| Rock | 3 sections | 9 sections | 15 sections | 21 sections |
|---|---:|---:|---:|---:|
| Bentheimer | 18.003% | 10.060% | 7.425% | 6.320% |
| Edwards Brown | 38.843% | 23.137% | 18.918% | 16.938% |

The same output reports a decrease in median within-region position-induced error IQR from about 9.15 to 2.63 percentage points for Bentheimer and from about 26.83 to 6.95 percentage points for Edwards Brown.

## Required source metadata before release

Before tagging a publication release, the repository should include a machine-readable manifest containing, for every third-party raw file:

- canonical dataset title;
- authors / data creators;
- DOI or permanent repository URL;
- original archive filename;
- decompressed filename;
- byte size;
- SHA-256 checksum;
- array shape and dtype;
- voxel size and units;
- phase convention;
- license / reuse terms;
- manuscript citation key.

The raw volumes themselves should be kept outside normal Git history unless their license and size make direct redistribution appropriate. A download/verification script or DOI-based acquisition instruction is preferred.