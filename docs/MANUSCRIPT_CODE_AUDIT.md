# Manuscript-code audit

This document compares the current manuscript against the executable files recovered from the original analysis workstation and the frozen outputs committed to this repository.

The purpose is not to rewrite provenance after the fact. A manuscript statement is marked **SUPPORTED** only when the archived code/results currently available can reproduce or directly substantiate it.

## Status legend

- **SUPPORTED** - archived executable evidence agrees with the manuscript.
- **PARTIAL** - the scientific statement is broadly supported, but implementation details differ or an explicit artifact is missing.
- **UNSUPPORTED BY ARCHIVED CODE** - no recovered script/result currently substantiates the stated numerical protocol. This does not prove the calculation was never performed; it means it should not be claimed as reproducible until the missing artifact is supplied.

## Main Methods audit (Sections 3.2-3.17)

The main Methods contains several statements that describe a broader or earlier workflow than the final archived production code. The Supplement and Results are substantially closer to the final real-rock implementation.

| Manuscript section | Claim | Archived implementation evidence | Status |
|---|---|---|---|
| 3.2 | pore indicator is 1 for pore | synthetic code uses True/1 for the generated pore phase; real RAW storage is 0=pore but scripts explicitly convert `phase==0` to the pore indicator | **SUPPORTED with storage-convention caveat** |
| 3.3 | overlap-corrected autocorrelation and directional interpolation | 2D scripts use FFT convolution with explicit valid-pair counts and interpolation | **SUPPORTED for 2D** |
| 3.3 | full-volume 3D autocorrelation evaluated by zero-padded FFT | final real-rock reference uses directional Monte Carlo sampling with trilinear off-lattice interpolation; final Fig. 4/5 use prescribed generator tensors as reference | **NOT THE FINAL PRODUCTION IMPLEMENTATION** |
| 3.4 | first 1/e crossing with linear interpolation | implemented throughout final scripts | **SUPPORTED** |
| 3.4 | robustness repeated with alternative thresholds and integral correlation scale | no recovered Python artifact implements these sensitivity experiments | **UNSUPPORTED BY ARCHIVED CODE** |
| 3.4 | real lengths converted to physical units before tensor fitting | final real-rock fitting is carried out in voxel units; voxel size is used for physical scale/display. Shape/anisotropy metrics are scale invariant | **UNSUPPORTED AS WORDED** |
| 3.5 / 3.8 | weighted least squares with positive-definiteness constraint | production code uses unweighted `np.linalg.lstsq` followed by eigenvalue flooring / SPD projection | **PARTIAL - same linear model, different numerical estimator** |
| 3.6 | non-tensorial residual eta | final scripts compute the same relative residual form with unit weights | **SUPPORTED with unit weights** |
| 3.7 | section restriction `Q_s = B_s^T Q_3 B_s` | exact basis of all final inversion scripts and reusable package | **SUPPORTED** |
| 3.8 | simultaneous use of directional observations to recover six tensor components | implemented directly in Fig. 4/5 and real-rock validation | **SUPPORTED** |
| 3.9 | rank hierarchy one/two/three sections = 3/5/6 | reproduced exactly | **SUPPORTED** |
| 3.9 | >3 section orientations selected by maximizing smallest singular value | no recovered implementation of this optimization was found; production benchmarks use fixed orthogonal/oblique bases or random subsets/offsets | **UNSUPPORTED BY ARCHIVED CODE** |
| 3.10 | moving-block bootstrap with 500 resamples per section generates directional weights and CIs | no moving-block bootstrap or `NB=500` implementation was found in the recovered code | **UNSUPPORTED BY ARCHIVED CODE** |
| 3.11 | 256^3, 50 realizations/combination, 512^3 repeat | not present in recovered production scripts; see synthetic audit below | **UNSUPPORTED BY ARCHIVED CODE** |
| 3.11 | every synthetic volume uses a measured full-volume tensor from 3D FFT as ground truth | Fig. 4/5 construct `Q_true` directly from prescribed principal scales and rotations and determinant-normalize it | **UNSUPPORTED AS WORDED** |
| 3.12 | rotation-equivariance test | Fig. 3 directly tests rotation equivariance and reproduces the reported 3.29% median | **SUPPORTED** |
| 3.12 | equivariance repeated with both nearest-neighbor and trilinear section sampling | interpolation order is compared in the separate Fig. 5 oblique-section sensitivity, not in the Fig. 3 equivariance experiment | **PARTIAL** |
| 3.13 | validation on Bentheimer, Doddington, Estaillades, Ketton 1000^3 Imperial College volumes | no such rock names occur in recovered code; final study uses Bentheimer + Edwards Brown from Ferreira et al. / Figshare+, 2500^3 ROI-1 | **STALE / CONTRADICTS FINAL STUDY** |
| 3.14 | tensor error in Eq. 36 uses raw `Q` difference | final scripts use determinant-normalized tensor-shape error; Results/Fig. 6 also explicitly call it determinant-normalized | **NEEDS EQUATION UPDATE** |
| 3.15 | FOV, section count/geometry, offsets, angular sampling, interpolation sensitivities | these components are represented across Figs. 4-6 | **SUPPORTED** |
| 3.15 | voxel-resolution study, alternative correlation threshold, and real-rock segmentation erosion/dilation | no corresponding recovered production artifacts were found | **UNSUPPORTED BY ARCHIVED CODE** |
| 3.16 | synthetic results use median/IQR | Fig. 5 uses median/IQR; Fig. 4 uses distributions/boxplots | **SUPPORTED** |
| 3.16 | synthetic bootstrap 95% confidence intervals | no final synthetic bootstrap-CI implementation was found | **UNSUPPORTED BY ARCHIVED CODE** |
| 3.17 | final implementation uses NumPy/SciPy | confirmed by scripts and audited environment | **SUPPORTED** |
| 3.17 | independent PoreSpy benchmark | exploratory PoreSpy scripts exist under the local `etapa1/` workspace, but no final frozen result artifact links them to the production figures | **PARTIAL** |
| 3.17 | all seeds/origins/orientations/parameters stored for every realization | seeds and fixed parameters are preserved; some final CSVs do not store every generated orientation/subset explicitly, although deterministic RNG allows regeneration | **PARTIAL** |

### High-priority manuscript fixes

Before submission, the most important consistency fixes are:

1. replace the stale Sec. 3.13 four-rock Imperial College protocol with the actual Ferreira et al. Bentheimer/Edwards Brown protocol;
2. reconcile Sec. 3.11 with the actual Figure 2-5 executable settings or supply the missing 256^3/50-realization/512^3 artifacts;
3. revise Eq. 36 to the determinant-normalized tensor-shape error actually used;
4. remove or substantiate the moving-block bootstrap / 500-resample weighting claim;
5. remove or substantiate unexecuted sensitivity claims (alternative correlation thresholds, integral scale, segmentation erosion/dilation, 512^3 synthetic repeat);
6. distinguish the real-rock Monte Carlo 3D reference estimator from any FFT-based synthetic calculation.

## Figures 2-5: executable synthetic settings

| Item | Manuscript / caption | Recovered production script | Status |
|---|---|---|---|
| Fig. 2 displayed anisotropy | A2D ~1.02 isotropic; ~2.73 anisotropic | `figure2.py` reproduces 1.0196 and 2.7271 | **SUPPORTED** |
| Fig. 2 ensemble | independent realizations | `N_ENSEMBLE=20`, `N=96`, porosity 0.27, 5-deg angular sampling | **SUPPORTED for the plotted figure** |
| Fig. 3 blind orientation | 37.0 deg -> 37.9 deg; median 1.15 deg; p95 3.10 deg | `figure3.py`: 256x256 parent canvas, 160x160 crop, 60 blind cases; reproduced all quoted values | **SUPPORTED** |
| Fig. 3 equivariance | median 3.29%, max 7.55% | reproduced 3.289% and 7.547% | **SUPPORTED** |
| Fig. 4 observability | ranks 3, 5, 6 | `figure4.py` reproduces rank sequence | **SUPPORTED** |
| Fig. 4 representative error | EQ=13.7%, axis errors ~8.0, 8.1, 1.8 deg, rho=0.44 | reproduced EQ=13.75%, axes 8.04, 8.10, 1.85 deg, rho=0.441 | **SUPPORTED** |
| Fig. 4 generator | general synthetic GRF description | `N=224`, section=144, porosity=0.27, scales [7.0,4.2,2.3], fixed Euler frame (24,37,19) deg | **SUPPORTED for figure-specific code; differs from Sec. 3.11 global benchmark numbers** |
| Fig. 5 ranges | porosities 0.15/0.25/0.35; anisotropy 1-4; angular increment 2.5-30 deg | exact same ranges in `figure5.py` | **SUPPORTED** |
| Fig. 5 production size/count | Sec. 3.11 states principal 256^3 benchmark and 50 independent realizations per parameter combination | `figure5.py` uses `N_MAIN=128`, 12 main realizations per anisotropy field, and a separate `N_SENS=160`, 10-realization sensitivity benchmark | **UNSUPPORTED BY ARCHIVED CODE as currently worded** |
| 512^3 synthetic repeat | Sec. 3.11 states a subset was repeated at 512^3 | no recovered Python script generating a 512^3 synthetic benchmark was found in the workspace | **UNSUPPORTED BY ARCHIVED CODE** |
| PoreSpy secondary benchmark | Secs. 3.11/3.17 state an independent PoreSpy generator benchmark | exploratory `etapa1/` PoreSpy scripts exist (typically SIZE=256, N_ROCKS=20), but the final Figure 2-5 scripts explicitly do not depend on PoreSpy and no frozen final result table currently links this prototype benchmark to the manuscript claim | **PARTIAL** |

## Section 3.11: key reconciliation required

The current manuscript states that the principal synthetic benchmark used:

- 256^3 volumes;
- porosities 0.15, 0.25, 0.35;
- anisotropy ratios 1.0, 1.5, 2.0, 3.0, 4.0;
- independent SO(3) rotations;
- 50 statistically independent realizations per parameter combination;
- a repeated subset at 512^3.

The recovered final robustness script agrees with the porosity range, anisotropy range, random SO(3) orientation idea, and general spectral-GRF construction, but not with the stated volume size, realization count, or 512^3 repeat.

### Author decision before submission

Choose one of the following and make the manuscript/repository consistent:

1. **Executable-provenance route (recommended):** revise Sec. 3.11 so the principal quantitative claims describe the actual final production scripts and frozen tables in this repository.
2. **Missing-artifact route:** if the 256^3 / 50-realization / 512^3 calculations were genuinely performed and are intended to support the paper, supply the missing scripts and result tables; they should then be added here and independently reproduced before submission.

Do not silently change the historical figure scripts to match the prose.

## Figure 6 / real-rock validation

| Item | Manuscript | Recovered executable evidence | Status |
|---|---|---|---|
| Data source | Ferreira et al., Figshare+ DOI 10.25452/figshare.plus.21375565 | downloader queries article 21375565; manifest records DOI | **SUPPORTED** |
| Raw geometry | 2500^3 binary ROI, 2.25 um voxels | RAW files are uint8, 15,625,000,000 bytes each; expected 2500^3 | **SUPPORTED** |
| Phase convention | 0=pore, 1=solid | validated in production workflow | **SUPPORTED** |
| Spatial replication | 27 non-overlapping 512^3 regions; centers 384,1250,2116 | exact production constants | **SUPPORTED** |
| Section pool | 3 orthogonal families, 13 offsets from -192 to +192 in steps of 32 | exact production constants | **SUPPORTED** |
| Reference calibration | 67 directions, 50,000 points/direction, 5-seed consensus | convergence + consensus scripts and final scripts agree | **SUPPORTED** |
| Sparse replication | 3,9,15,21 total sections; 200 random configurations | exact production constants | **SUPPORTED** |
| Bentheimer error | 18.00% -> 6.32% | reproduced from final CSVs | **SUPPORTED** |
| Edwards Brown error | 38.84% -> 16.94% | reproduced from final CSVs | **SUPPORTED** |
| Figure 6 renderer | final frozen outputs rather than recalculating heavy reference stage | portable and historical renderers both read frozen validation tables | **SUPPORTED** |

## Table S1 filename audit

The manuscript table contains several descriptive filenames that did not exist literally in the recovered directory. The script docstrings reveal the intended canonical names.

| Manuscript / intended name | Recovered workstation filename | Public repository |
|---|---|---|
| `download_real_rocks.py` | `script.py` | `scripts/data/download_real_rocks.py` |
| `real_bentheimer_test4_reference_convergence.py` | `real_bentheimer_test4.py` | canonical intended name used |
| `real_bentheimer_test5_consensus_convergence.py` | `real_bentheimer_test5.py` | canonical intended name used |
| `real_bentheimer_final_validation.py` | `real_bentheimer_test6.py` | canonical production name used |
| `real_edb1_final_validation_v2.py` in Table S1 | `real_edb1_final_validation.py`; no `*_v2.py` file found anywhere in the recovered workspace | `real_edb1_final_validation.py` |

**Recommended manuscript correction:** change the EdB-1 Table S1 filename to `real_edb1_final_validation.py` unless an actual v2 artifact is supplied.

Table S1 also omits the final Figure-6 comparison renderer. For auditability, consider adding `figure6.py` as the lightweight script that reads the frozen final CSVs and generates the comparison figure.

## Reproduction audit completed

On the original workstation:

- all recovered Figure 1-6 scripts compiled;
- Figures 1-6 were executed;
- the quoted numerical values for Figures 2-6 were reproduced;
- the real-rock RAW SHA-256 values were calculated and stored in `data/manifest.csv`;
- the portable Figure-6 renderer reproduced the same comparison using project-relative frozen tables and an external raw-data root;
- the reusable package tests pass and the end-user tensor inversion example recovers a known tensor to machine precision.

## Release gate

Before merging/tagging a manuscript release:

- [ ] reconcile Sec. 3.11 with executable synthetic provenance, or supply the missing benchmark artifacts;
- [ ] decide whether the exploratory PoreSpy benchmark should be formalized/preserved or the claim narrowed;
- [ ] correct the EdB-1 filename in Table S1;
- [ ] consider adding `figure6.py` to Table S1;
- [ ] wait for CI to be green;
- [ ] only then mark the PR ready for review and merge to `main`.
