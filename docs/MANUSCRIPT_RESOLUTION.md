# Manuscript reconciliation status

The manuscript-code discrepancies identified in `MANUSCRIPT_CODE_AUDIT.md` were reviewed against the archived production scripts, frozen CSV outputs, and the Supplementary Information. The author approved the executable-provenance route: the manuscript was revised to describe the calculations that actually produced the reported figures, rather than modifying historical scripts to fit earlier prose.

## Resolved items

- The stale four-rock Imperial College real-rock Methods protocol was replaced by the final Bentheimer + Edwards Brown Ferreira et al./Figshare+ protocol.
- The synthetic Methods now use the actual figure-specific settings: Fig. 2 (`96^3`, 20 realizations), Fig. 3 (`256x256` parent canvas, `160x160` crop, 60 blind cases), Fig. 4 (`224^3`, 20-section pool, 220 accepted subsets per section count), and Fig. 5 (`128^3`, 12 main realizations plus `160^3`, 10 sensitivity realizations).
- The unsupported `256^3` / 50-realization / `512^3` synthetic claims were removed.
- Tensor estimation is now described as ordinary least squares followed by SPD projection, matching the production implementation.
- The unsupported moving-block bootstrap / directional weighting claim was replaced by the actual uncertainty hierarchy.
- The Fig. 3 equivariance test is correctly described as a 2D in-plane tensor test; interpolation sensitivity remains a separate Fig. 5 experiment.
- The real-rock reference is described as a finite-volume numerical Monte Carlo directional estimate, not analytic ground truth and not a 3D FFT field.
- Tensor-shape error is defined after determinant normalization, matching Figs. 4--6.
- Sensitivity claims were narrowed to archived production experiments.
- The EdB-1 script name in Table S1 was corrected to `real_edb1_final_validation.py`, and `figure6.py` was added.
- The PoreSpy statement was narrowed to acknowledge exploratory development code without treating it as part of the frozen Fig. 2--5 quantitative benchmark.
- Duplicate/orphaned bibliography entries were repaired in the repository manuscript bibliography.

## Final verification

The revised manuscript source is under `manuscript/`. The author-side Overleaf-ready bundle completed the full LaTeX/BibTeX build with no undefined citations or internal references and no BibTeX errors. A final Results-to-CSV audit reproduced the numerical real-rock claims, including the section-count medians, position-IQR values, anisotropy differences, orientation-reliability counts, and success fractions.

The historical audit document is retained as evidence of what was found before reconciliation; this file records how those findings were resolved.
