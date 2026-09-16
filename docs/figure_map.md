# Figure and data map

| Figure | Scientific role | Frozen script | Principal machine-readable output |
|---|---|---|---|
| 1 | Conceptual pipeline: oriented section, autocorrelation, 1/e length, 2D tensor | `scripts/paper/figures/figure1.py` | figure itself |
| 2 | In-plane anisotropy detectability / negative control | `scripts/paper/figures/figure2.py` | figure itself |
| 3 | Blind in-plane orientation recovery and rotation equivariance | `scripts/paper/figures/figure3.py` | `data/processed/figure3/figure3_blind_rotation_results.csv` |
| 4 | 3D observability, rank, conditioning, sparse tensor recovery | `scripts/paper/figures/figure4.py` | `data/processed/figure4/figure4_q3d_recovery_results.csv` |
| 5 | Synthetic robustness to porosity, anisotropy, FOV, angle increment, interpolation | `scripts/paper/figures/figure5.py` | `data/processed/figure5/figure5_robustness_results.csv` |
| 6 | **Real-rock** validation: Bentheimer vs Edwards Brown | `scripts/paper/figures/figure6.py` | `data/processed/figure6/*_final_*.csv` |

## Figure 6 dependency chain

```text
Figshare+ v6 binary ROI-1 files
          |
          +--> Bentheimer calibration tests 1-5
          |        |
          |        +--> frozen 50k points x 5 seeds reference protocol
          |
          +--> final Bentheimer production validation
          |        +--> bentheimer_final_*.csv
          |
          +--> final EdB-1 production validation (same frozen protocol)
                   +--> edb1_final_*.csv

bentheimer_final_* + edb1_final_*
          |
          +--> scripts/paper/figures/figure6.py
                   |
                   +--> Figure 6
                   +--> figure6_real_rock_comparison_summary.csv
```

The Figure 6 plotting script does not recompute the expensive 3D references. Quantitative panels are constructed from the frozen final CSV outputs; the RAW volumes are only needed to draw representative 2D microstructure panels.

## Calibration-script naming

The working directory used incremental filenames `real_bentheimer_test1.py` through `real_bentheimer_test6.py`. In this repository, tests 1-5 remain under `scripts/paper/real_rocks/calibration/`, while test 6 is published as `scripts/paper/real_rocks/real_bentheimer_final_validation.py` because it is the frozen production analysis.
