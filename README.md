# Sparse-Section Anisotropy

**Direct inference of 3D pore-space anisotropy from sparse, oriented 2D sections.**

This repository accompanies the manuscript *Beyond 2D-to-3D Reconstruction: Direct Inference of Pore-Space Anisotropy from Sparse Sections* (LCCMat, University of Brasilia, September 2026).

The method does **not** reconstruct a full 3D pore geometry. Instead, it estimates a symmetric positive-definite (SPD) 3D correlation tensor from directional two-point correlation lengths measured on oriented 2D sections.

## Core model

For a unit 3D direction `v`, the directional correlation length is modeled as

```text
1 / ell(v)^2 = v.T @ Q @ v
```

where `Q` is a 3x3 SPD tensor. If a 2D section has orthonormal basis `B` (shape 3x2) and in-plane unit direction `u`, then `v = B @ u` and the section observes

```text
Q_section = B.T @ Q @ B
```

Each directional measurement therefore contributes one linear equation for the six independent entries of `Q`.

## Repository map

```text
src/sparse_section_anisotropy/   reusable inference code
examples/                        minimal user-facing workflows
scripts/                         manuscript figure / validation entry points
data/                            data manifest only; large raw volumes are not committed
docs/                            methods, provenance, and reproducibility notes
results/                         lightweight derived tables used by figures
tests/                           numerical and geometry checks
```

## Scientific workflow

```text
binary 2D sections + orientation matrices
        |
        v
directional unbiased autocorrelation C(u,r)
        |
        v
first 1/e crossing -> ell(u)
        |
        v
map u -> v = B @ u
        |
        v
assemble design matrix A and y = 1/ell^2
        |
        v
least-squares tensor estimate + SPD projection
        |
        v
eigenvectors/eigenvalues -> principal directions and correlation scales
```

Three suitably complementary section orientations are the minimum geometry for a generic full-rank 3D symmetric tensor inversion. Additional spatially separated sections do not add tensor degrees of freedom once rank six is achieved, but improve representativity and reduce finite-sampling variability.

## Manuscript figures

| Figure | Purpose | Data type |
|---|---|---|
| Fig. 1 | Method schematic / tensor construction | illustrative / synthetic |
| Fig. 2 | In-plane detectability | synthetic Gaussian random fields |
| Fig. 3 | Blind orientation recovery and rotation equivariance | synthetic Gaussian random fields |
| Fig. 4 | 3D observability, conditioning, and recovery | synthetic spectral Gaussian random field |
| Fig. 5 | Robustness to porosity, anisotropy, field of view, angular sampling, interpolation | synthetic spectral Gaussian random fields |
| Fig. 6 | Bentheimer sandstone vs Edwards Brown carbonate | real segmented micro-CT data |

**Important:** Figure 6 is not based on synthetic rock data. It is a real-rock validation. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/infer_from_sections.py
```

The example constructs a known SPD tensor, creates synthetic directional section measurements, and verifies that the six tensor components are recovered from multiple oriented planes. Replace those measurements with correlation lengths extracted from your own segmented 2D rock images to use the same inversion stage on experimental data.

## Reproducibility policy

- Random seeds, section origins, section orientations, and reconstruction parameters must be stored with every generated result.
- Raw third-party image volumes are referenced by filename, source, expected dimensions, voxel size, and checksum rather than silently copied into the repository.
- Publication figures should read derived CSV tables whenever a heavy numerical stage has already been frozen.
- A figure-generation script must state whether it recomputes the numerical experiment or only renders existing results.
- Absolute workstation paths are not allowed in reusable code.

## Data availability

The real-rock workflow uses segmented `2500 x 2500 x 2500` uint8 binary volumes at `2.25 micrometre` voxel size in the local production pipeline. The manuscript validation uses 27 non-overlapping `512^3` subvolumes per rock. Large raw image files are intentionally excluded from Git history; exact acquisition/file provenance and expected local layout are documented in `docs/DATA_PROVENANCE.md` and `data/README.md`.

## Status

The reproducibility branch now preserves the recovered historical Figure 1--6 scripts, the Bentheimer/Edwards Brown validation pipeline, frozen manuscript CSV outputs, third-party data checksums, and a separate reusable inference API. The historical code is retained as provenance; portable wrappers and reusable modules are kept separate so auditability is not lost during cleanup.

The remaining author-level task before submission is to reconcile the manuscript's broad synthetic benchmark description with the exact figure-specific settings recorded in `docs/DATA_PROVENANCE.md`.

## License

See [`LICENSE`](LICENSE). Third-party datasets retain their original licenses and citation requirements.