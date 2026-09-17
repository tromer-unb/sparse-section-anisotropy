# Manuscript revision audit

Source preserved unchanged:
- `main.tex`
- `bibliography.bib`

Review copies:
- `main_revised.tex`
- `bibliography_revised.bib`
- `main_revised.pdf`
- `manuscript_revision.diff`

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

The original bibliography contained four duplicate keys:
`Jiao2007`, `Inglis2003`, `Cowin2004`, and `AlRaoush2010`.

It also contained trailing orphaned field blocks duplicating already valid entries for `Singh2020`, `Zubov2024`, and `Ferreira2023`.

`bibliography_revised.bib` preserves one valid copy of each entry and removes only the duplicate/orphaned blocks. The original `bibliography.bib` is unchanged.

## Build validation

Full build sequence:
`pdflatex -> bibtex -> pdflatex -> pdflatex`

Result:
- no LaTeX compilation errors;
- no BibTeX errors or warnings;
- no undefined citations;
- no undefined internal references;
- Table S1 no longer exceeds the page as a single float.

The remaining overfull-box warnings are typographic and do not indicate broken references or scientific-content errors.