# Manuscript-code audit

This document compares the current manuscript against the executable files recovered from the original analysis workstation and the frozen outputs committed to this repository.

The purpose is not to rewrite provenance after the fact. A manuscript statement is marked **SUPPORTED** only when the archived code/results currently available can reproduce or directly substantiate it.

## Status legend

- **SUPPORTED** - archived executable evidence agrees with the manuscript.
- **PARTIAL** - the scientific statement is broadly supported, but implementation details differ or an explicit artifact is missing.
- **UNSUPPORTED BY ARCHIVED CODE** - no recovered script/result currently substantiates the stated numerical protocol. This does not prove the calculation was never performed; it means it should not be claimed as reproducible until the missing artifact is supplied.

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
