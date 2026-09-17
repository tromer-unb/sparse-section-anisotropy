# Manuscript revision audit

This directory contains the author-approved, code-aligned manuscript snapshot produced during the September 2026 reproducibility audit.

The original author-side `main.tex` and `bibliography.bib` were preserved unchanged on the workstation during the revision process. In this repository, `manuscript/main.tex` and `manuscript/bibliography.bib` are the revised, publication-facing copies.

## Scientific corrections

The revised manuscript was aligned to the archived production scripts and frozen outputs that reproduce Figs. 1--6.

- Removed the duplicated Introduction section heading.
- Rewrote the 2D autocorrelation description to match the overlap-normalized FFT implementation.
- Replaced weighted/constrained tensor fitting with the production estimator: ordinary least squares followed by SPD projection.
- Replaced weighted observability expressions with the unweighted design matrix actually used.
- Replaced the unsupported moving-block bootstrap description with the actual uncertainty hierarchy: independent synthetic realizations/subsets, section-position resampling, and Monte Carlo reference calibration.
- Replaced the generic synthetic benchmark protocol with the exact Fig. 2--5 production settings.
- Corrected Fig. 3 rotation equivariance to the actual 2D in-plane tensor test.
- Replaced the stale four-rock Imperial College real-rock Methods section with the final Ferreira et al. Bentheimer + Edwards Brown protocol.
- Clarified that the real-rock 3D reference is a numerical Monte Carlo directional estimate, not analytic ground truth and not a full-volume 3D FFT.
- Replaced the tensor error definition with the determinant-normalized tensor-shape error used by Figs. 4--6.
- Narrowed sensitivity-analysis claims to experiments actually represented by the archived production code.
- Clarified the real-rock statistical hierarchy: 200 position configurations within each subvolume, then 27 spatial subvolumes within one ROI.
- Narrowed the PoreSpy statement: exploratory code exists, but PoreSpy is not part of the frozen Fig. 2--5 quantitative pipeline.

## Supplementary Information

- Replaced the generic synthetic-generation description with exact figure-specific sizes, seeds, realization counts, and reference construction.
- Corrected `real_edb1_final_validation_v2.py` to the archived production name `real_edb1_final_validation.py`.
- Added `figure6.py` to the computational-script inventory.
- Recorded the clean-environment reproduction versions used in September 2026.
- Converted Table S1 to a multipage `longtable` and enabled breakable path formatting for long script names.

## Bibliography repair

The exported bibliography contained four duplicate keys: `Jiao2007`, `Inglis2003`, `Cowin2004`, and `AlRaoush2010`. It also contained trailing orphaned field blocks duplicating already valid entries for `Singh2020`, `Zubov2024`, and `Ferreira2023`.

The repository bibliography preserves one valid copy of each entry and removes only those duplicate/orphaned blocks.

## Build and quantitative validation

The author-side Overleaf bundle completed `pdflatex -> bibtex -> pdflatex -> pdflatex` with no LaTeX compilation errors, no BibTeX errors/warnings, no undefined citations, and no undefined internal references.

A final Results-to-CSV audit also confirmed the reported real-rock summary values, including Bentheimer `18.003% -> 6.320%`, Edwards Brown `38.843% -> 16.938%`, the position-IQR values, the anisotropy differences, the orientation-reliability counts, and the Bentheimer 15-section `E_Q <= 10%` success fraction (25/27 = 92.6%, reported as 93%).

The remaining overfull-box warnings are typographic and do not indicate broken references or scientific-content errors.
