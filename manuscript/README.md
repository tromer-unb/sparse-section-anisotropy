# Manuscript source

This directory stores the author-approved, code-aligned manuscript snapshot produced during the September 2026 reproducibility audit.

- `main.tex` is the revised manuscript and Supplementary Information.
- `bibliography.bib` is the cleaned bibliography used by the revised source.
- `REVISION_NOTES.md` records the scientific and bibliographic corrections applied.

The original author files were preserved outside this repository during the audit. The revision changes the prose and equations where necessary to describe the executable production workflow; it does not modify the historical production scripts to force agreement with the prose.

## Figures

The six publication figures are intentionally not duplicated here as binary assets. Their production code is preserved under `scripts/figures/`, and the frozen quantitative tables are under `results/`.

For an Overleaf-ready bundle, place the generated publication images `figure1.png` through `figure6.png` beside `main.tex` and `bibliography.bib`.

## Build

With the six PNG files present:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The audited local bundle completed this build sequence with no undefined citations, no undefined internal references, and no BibTeX errors.
