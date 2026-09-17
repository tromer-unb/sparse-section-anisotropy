# Proposed manuscript corrections based on executable provenance

This file is an **author-review draft**. It does not modify the manuscript. The wording below is based on the recovered production scripts and the numerical reproduction audit.

## 1. Replace the stale real-rock Methods section (current Sec. 3.13)

### Proposed replacement

**Validation on real micro-CT rock images.** Real-rock validation used publicly available segmented micro-computed-tomography data from Ferreira et al. (Scientific Data 10, 368, 2023; Figshare+ DOI 10.25452/figshare.plus.21375565). The analysis used the binary ROI-1 volumes of Bentheimer sandstone (Kocurek 15A) and Edwards Brown carbonate (EdB-1). Both files are stored as 2500 x 2500 x 2500 uint8 voxel arrays with a voxel size of 2.25 micrometres. The validated storage convention in the production workflow is 0 = pore and 1 = solid.

Each ROI was sampled on a 3 x 3 x 3 grid of 27 non-overlapping 512^3 subvolumes, with center coordinates 384, 1250, and 2116 along each computational array axis. The axes were denoted a0, a1, and a2 unless an independent mapping to physical plug coordinates was available.

For each subvolume, a numerical full-3D reference tensor was estimated from directional correlation lengths sampled along 67 ambient directions (64 Fibonacci-hemisphere directions plus the three computational axes). The frozen reference protocol used 50,000 Monte Carlo base points per direction and five independent seeds. The five seed-specific tensors were averaged in Q-space and projected to the SPD cone. This reference is a numerical estimate rather than an analytic ground truth.

Sparse reconstructions used three mutually orthogonal section families (a0-a1, a0-a2, a1-a2). Thirteen candidate offsets per family were precomputed from -192 to +192 voxels in steps of 32. Reconstructions used 1, 3, 5, or 7 sections per orientation, corresponding to 3, 9, 15, or 21 total sections. At each section count, 200 random section-position configurations were evaluated per subvolume. The numerical protocol was calibrated on Bentheimer and transferred unchanged to Edwards Brown; no lithology-specific retuning was performed.

## 2. Replace / qualify the current Sec. 3.11 synthetic benchmark paragraph

The current manuscript states a 256^3, 50-realization-per-combination benchmark plus a 512^3 repeat. Those artifacts are not present in the recovered workspace.

### Proposed executable-provenance wording

**Synthetic porous-media benchmarks.** Synthetic validation was organized as a sequence of controlled, figure-specific experiments. The in-plane detectability experiment (Fig. 2) used thresholded three-dimensional Gaussian random fields of size 96^3 at porosity 0.27. The isotropic field used Gaussian filter scales (4.8, 4.8, 4.8), whereas the anisotropic field used (4.8, 4.8, 1.8); 20 independent realizations were used for the ensemble comparison.

The blind orientation experiment (Fig. 3) used two-dimensional thresholded anisotropic Gaussian random fields generated on a 256 x 256 parent canvas and analyzed after rotation and central cropping to 160 x 160 pixels. Sixty independently generated blind-angle cases were used to quantify orientation recovery.

The three-dimensional observability experiment (Fig. 4) used a stationary spectral Gaussian random field on a 224^3 periodic cube with porosity 0.27 and prescribed principal correlation scales (7.0, 4.2, 2.3) in a rotated principal frame. The reference tensor shape was defined from the prescribed principal scales and rotation and determinant-normalized. A pool of 20 oriented sections was used for subset experiments, with 220 accepted full-rank subsets at each section count from three to six.

The robustness experiment (Fig. 5) used 128^3 spectral Gaussian random fields for the main porosity-by-anisotropy benchmark. Porosities 0.15, 0.25, and 0.35 and prescribed anisotropy ratios 1.0, 1.5, 2.0, 3.0, and 4.0 were evaluated with 12 independent field realizations for each anisotropy setting; the same continuous field realization was thresholded at the three prescribed porosities. A separate 160^3 sensitivity benchmark with 10 realizations tested section field of view, directional angular increment, and oblique-section interpolation.

If the earlier 256^3 / 50-realization / 512^3 benchmark is scientifically important to retain, its missing scripts and result tables should instead be recovered and added to the repository before submission.

## 3. Revise the tensor-error equation (current Eq. 36)

The production scripts compare **shape** after determinant normalization.

### Proposed definition

Let

```text
Q_tilde = Q / det(Q)^(1/3).
```

Then define

```text
E_Q = || Q_tilde_sparse - Q_tilde_ref ||_F / || Q_tilde_ref ||_F.
```

For synthetic experiments, `Q_ref` is the prescribed determinant-normalized tensor used by the corresponding production script. For real rocks, `Q_ref` is the frozen numerical full-3D reference tensor for that subvolume.

This definition matches the Figure 4-6 code and avoids conflating anisotropy shape error with an overall correlation-length scale difference.

## 4. Replace the current block-bootstrap weighting statement unless missing code is supplied

No production implementation of a 500-realization moving-block bootstrap or bootstrap-derived directional weights was found.

### Proposed wording consistent with the final pipeline

Directional tensor fits used unit weights in the production scripts. For real rocks, uncertainty associated with section position was quantified by repeated sparse reconstructions over random section-offset configurations within each fixed 512^3 subvolume. Numerical uncertainty of the full-3D reference estimator was calibrated separately using independent Monte Carlo seeds and bootstrap resampling of seed-specific tensors during the consensus-size study. These two uncertainty sources were reported separately and were not interpreted as independent geological replication.

If a moving-block bootstrap was in fact used for a separate result, the script and its generated tables should be supplied before retaining the current statement.

## 5. Narrow the sensitivity-analysis statement (current Sec. 3.15)

### Proposed wording

Robustness was evaluated with respect to section field of view, section number and geometry, section position, directional angular sampling, and nearest-neighbor versus trilinear-plus-threshold interpolation for oblique sections. The controlled synthetic benchmark in Fig. 5 isolates field-of-view, angular-sampling, and interpolation effects, whereas the real-rock validation quantifies section-position and replication effects over 27 spatially separated subvolumes.

The archived production code does not currently support claims of a dedicated voxel-resolution study, alternative correlation-threshold / integral-scale study, or real-rock erosion/dilation segmentation sensitivity. These clauses should be removed unless the corresponding artifacts are supplied.

## 6. Clarify numerical estimator wording in Secs. 3.5 and 3.8

The production implementation solves the linear system in `1/ell^2` by unweighted least squares and then symmetrizes/projects the estimated matrix to the SPD cone by eigenvalue flooring.

Suggested phrasing:

> In the production implementation, all valid directional observations were assigned unit weight. The unconstrained least-squares estimate of the symmetric tensor components was subsequently projected to the SPD cone by flooring non-positive or numerically negligible eigenvalues. The weighted constrained formulation provides a natural extension when independently estimated directional uncertainties are available, but bootstrap-derived weights were not used in the frozen production results reported here.

## 7. Update Table S1

Recommended filename corrections:

- `download_real_rocks.py`
- `real_bentheimer_test4_reference_convergence.py`
- `real_bentheimer_test5_consensus_convergence.py`
- `real_bentheimer_final_validation.py`
- `real_edb1_final_validation.py` (no `*_v2.py` artifact was found)
- add `figure6.py` as the lightweight final real-rock comparison renderer

## 8. PoreSpy statement

Exploratory PoreSpy scripts were recovered from an earlier `etapa1/` workspace (typically 256^3 volumes and 20 realizations), but the frozen Figure 2-5 production scripts explicitly do not depend on PoreSpy.

Two defensible options are available:

- preserve and formalize the exploratory PoreSpy benchmark with its exact outputs and describe it explicitly as a separate implementation check; or
- narrow Secs. 3.11/3.17 so PoreSpy is not presented as part of the frozen quantitative benchmark.

The repository should not imply a stronger PoreSpy validation than the retained artifacts support.

## Release recommendation

Do not alter historical production scripts merely to make them match manuscript prose. Reconcile the prose to the executable provenance, or supply the missing artifacts and reproduce them before release.
