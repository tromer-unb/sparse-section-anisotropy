# Reproducibility workflow

## End-to-end order

The manuscript workflow should be reproduced in this order:

1. run the synthetic scripts for Figures 1-5 with their embedded fixed seeds;
2. query/download the Kocurek 15A and EdB-1 binary ROI-1 files with `scripts/data/download_real_rocks.py`;
3. validate dimensions, binary values, and phase convention;
4. reproduce the Bentheimer 3D reference-estimator convergence study;
5. reproduce the consensus-size calibration and freeze the reference protocol;
6. execute the final Bentheimer validation;
7. execute the final EdB-1 validation using the same frozen protocol, without lithology-specific retuning;
8. construct Figure 6 from the final CSV outputs.

The calibration stages are part of the audit trail. They should not be independently retuned on Edwards Brown when reproducing the manuscript's external-validation design.

## Frozen final real-rock protocol

| Parameter | Value |
|---|---:|
| ROI shape | 2500 x 2500 x 2500 uint8 |
| analysis subvolume | 512^3 voxels |
| spatial grid | 3 x 3 x 3 = 27 subvolumes |
| max correlation lag | 48 voxels |
| reference directional set | 64 Fibonacci directions + 3 Cartesian axes = 67 |
| Monte Carlo points/direction | 50,000 |
| independent reference seeds | 5 |
| 2D angular increment | 5 degrees |
| sections/orientation | 1, 3, 5, 7 |
| total sections | 3, 9, 15, 21 |
| candidate offsets/orientation | 13 |
| random section configurations/count/subvolume | 200 |
| Bentheimer production seed | 20260907 |
| EdB-1 production seed | 20260908 |

The five-seed, 50,000-point protocol was frozen after the Bentheimer convergence/consensus calibration. Its empirical Bentheimer numerical-reference diagnostics were:

- median uncertainty proxy: 0.932%;
- 95th percentile: 1.706%;
- worst-subvolume 95th percentile: 1.881%.

These values are calibration diagnostics, not an Edwards-Brown-specific uncertainty bound.

## Restart-safe caches

The production scripts separately cache:

1. seed-specific 3D reference tensors;
2. the 39 precomputed planar section measurements per subvolume;
3. completed sparse-resampling results.

Cached data are reused only when the stored calculation signature is compatible with the active settings. The released compact CSV outputs are sufficient to audit published summary statistics; full cache directories are computational intermediates and are not required for ordinary repository use.

## Environment recorded during repository audit

```text
Python       3.12.3
NumPy        2.3.2
SciPy        1.16.0
Matplotlib   3.10.3
pandas       2.3.1
Pillow       10.2.0
```

The frozen standalone Figure 1-6 scripts passed Python byte-code compilation under this environment on 2026-09-16.

`PoreSpy` was not installed in this audited workstation environment. It is therefore not listed as a requirement for the core release. The manuscript mentions a secondary PoreSpy generator benchmark; if that benchmark is released later, its environment should be pinned separately rather than silently added to the core inference dependencies.
