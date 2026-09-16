#!/usr/bin/env python3
"""
Bentheimer real-rock validation: test 2
========================================

Goal
----
Quantify the effect of SECTION REPRESENTATIVITY while keeping geometric
observability fixed.

The same central 512^3 Bentheimer subvolume used in test 1 is analyzed in two
ways:

    complete 3D subvolume  -> Q3D_REF

and

    multiple 2D sections from 3 orthogonal plane families -> Q3D_SPARSE

The number of parallel section offsets per orientation is varied as

    1, 3, 5, 7 sections per orientation

which corresponds to

    3, 9, 15, 21 total sections.

Two complementary experiments are performed:

1) SYSTEMATIC CONFIGURATIONS
   Equally spaced offsets spanning the same central interval are used for each
   section count.  This gives one transparent deterministic reconstruction for
   each n.

2) RANDOM POSITION RESAMPLING
   A pool of 13 precomputed offsets per plane family is created.  For each n,
   many random subsets are drawn from this same pool without recomputing the
   section correlations.  This quantifies sensitivity to section position and
   how that uncertainty changes as more parallel sections are included.

Scientific interpretation
-------------------------
Three appropriately oriented planes are sufficient for tensor observability
(rank 6), but a single section at each orientation need not be statistically
representative of a heterogeneous rock.  This script isolates that distinction:

    orientation diversity -> observability
    repeated offsets       -> representativity / reduced sampling uncertainty

IMPORTANT TERMINOLOGY
---------------------
For the real rock there is no analytic tensor ground truth.  The tensor
estimated from directional correlations throughout the complete selected 3D
subvolume is therefore called Q3D_REF (reference), not Q3D_GT.

RAW convention already validated for this dataset:

    RAW value 0 = pore
    RAW value 1 = solid

The computational array axes are called a0, a1, a2 until the physical axis
mapping is confirmed from metadata.

Outputs
-------
  bentheimer_test2.png
  bentheimer_test2.svg
  bentheimer_test2_report.txt
  bentheimer_test2_results.csv
  bentheimer_test2_section_measurements.csv

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

Run
---
  python3 real_bentheimer_test2.py

The script opens the 2500^3 RAW with numpy.memmap and copies only the central
512^3 subvolume into RAM.
"""

from pathlib import Path
import csv
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.stats import spearmanr


# =============================================================================
# USER SETTINGS
# =============================================================================

RAW_FILE = Path(
    "kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)

RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8

# Same central subvolume as test 1.
SUBVOL_N = 512

# Directional-correlation settings.
MAX_LAG = 48
THETA_STEP_2D_DEG = 5.0

# Full-3D reference estimator -- intentionally kept identical to test 1 so
# Q3D_REF is directly comparable to the first real-rock experiment.
N_REF_DIRECTIONS = 64
N_REF_POINTS = 12_000

# Section-position experiment.
SECTIONS_PER_ORIENTATION = (1, 3, 5, 7)

# Precomputed position pool.  For a 512^3 subvolume, this creates offsets
# -192, -160, ..., 0, ..., +160, +192 (13 positions per plane family).
OFFSET_MAX = 192
OFFSET_STEP = 32

# Number of random configurations for each section count.  Once the section
# pool has been measured, these reconstructions are inexpensive.
N_RANDOM_CONFIGS = 400

# Major-axis orientation is not treated as a primary metric when the reference
# tensor is close to spectrally degenerate.  This threshold is based on the
# relative separation between the two largest principal correlation lengths.
ORIENTATION_GAP_MIN = 0.10

SEED = 20260907
DPI = 800


# =============================================================================
# OUTPUTS / STYLE
# =============================================================================

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "bentheimer_test2.png"
SVG_PATH = OUTDIR / "bentheimer_test2.svg"
REPORT_PATH = OUTDIR / "bentheimer_test2_report.txt"
RESULTS_CSV_PATH = OUTDIR / "bentheimer_test2_results.csv"
SECTION_CSV_PATH = OUTDIR / "bentheimer_test2_section_measurements.csv"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "font.weight": "bold",
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# BASIC TENSOR UTILITIES
# =============================================================================

def normalize_det_spd(Q):
    """Symmetrize, project to SPD, and normalize det(Q)=1."""
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-8 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = V @ np.diag(w) @ V.T
    return Q / np.linalg.det(Q) ** (1.0 / Q.shape[0])


def qvec_to_matrix(q):
    return np.array([
        [q[0], q[3], q[4]],
        [q[3], q[1], q[5]],
        [q[4], q[5], q[2]],
    ], dtype=float)


def design_rows_from_vectors(v):
    """Design rows for v^T Q v with a symmetric 3x3 tensor."""
    v = np.asarray(v, dtype=float)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.column_stack([
        x*x,
        y*y,
        z*z,
        2*x*y,
        2*x*z,
        2*y*z,
    ])


def ell_from_q3d(Q, vectors):
    vectors = np.asarray(vectors, dtype=float)
    val = np.einsum("ni,ij,nj->n", vectors, Q, vectors)
    val = np.maximum(val, 1e-14)
    return 1.0 / np.sqrt(val)


def fit_q3d_from_directional_lengths(vectors, ell):
    """
    Unweighted least squares in y=1/ell^2, followed by SPD projection.

    Returns
    -------
    Q, rank, condition_number, eta, vectors_used, lengths_used
    """
    vectors = np.asarray(vectors, dtype=float)
    ell = np.asarray(ell, dtype=float)

    good = np.isfinite(ell) & (ell > 0)
    if np.sum(good) < 6:
        raise RuntimeError("Fewer than six valid directional lengths.")

    V = vectors[good]
    L = ell[good]
    A = design_rows_from_vectors(V)
    y = 1.0 / (L * L)

    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    if rank < 6:
        raise RuntimeError(f"3D tensor design matrix is rank deficient: rank={rank}")

    q, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = qvec_to_matrix(q)

    # Finite-section noise can make the unconstrained LS estimate slightly
    # indefinite.  The SPD projection is kept identical to test 1.
    Q = 0.5 * (Q + Q.T)
    w, E = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-7 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = E @ np.diag(w) @ E.T

    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1])

    ell_pred = ell_from_q3d(Q, V)
    eta = float(np.sqrt(np.sum((L - ell_pred)**2) / np.sum(L**2)))

    return Q, rank, cond, eta, V, L


def principal_lengths_and_axes(Q):
    """Principal correlation lengths from largest to smallest."""
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    w = np.maximum(w, 1e-14)
    ell = 1.0 / np.sqrt(w)
    order = np.argsort(ell)[::-1]
    return ell[order], V[:, order]


def anisotropy_ratio(Q):
    ell, _ = principal_lengths_and_axes(Q)
    return float(ell[0] / ell[-1])


def axial_angle_error_deg(v1, v2):
    """Acute angle between undirected axes."""
    c = np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))


def tensor_shape_error(Q_sparse, Q_ref):
    Qs = normalize_det_spd(Q_sparse)
    Qr = normalize_det_spd(Q_ref)
    return float(
        np.linalg.norm(Qs - Qr, ord="fro") /
        np.linalg.norm(Qr, ord="fro")
    )


def reference_major_axis_gap(Q_ref):
    """
    Relative separation of the two largest reference principal lengths.

    Small values mean the major-axis orientation is intrinsically unstable.
    """
    ell, _ = principal_lengths_and_axes(normalize_det_spd(Q_ref))
    return float((ell[0] - ell[1]) / ell[0])


# =============================================================================
# RAW / SUBVOLUME HANDLING
# =============================================================================

def validate_raw(path):
    expected = int(np.prod(RAW_SHAPE)) * np.dtype(RAW_DTYPE).itemsize
    actual = path.stat().st_size

    print("=" * 78)
    print("RAW VALIDATION")
    print("=" * 78)
    print(f"File:     {path}")
    print(f"Shape:    {RAW_SHAPE}")
    print(f"dtype:    {np.dtype(RAW_DTYPE)}")
    print(f"Expected: {expected:,} bytes")
    print(f"Actual:   {actual:,} bytes")

    if actual != expected:
        raise RuntimeError(
            f"RAW size mismatch: expected {expected}, found {actual}."
        )
    print("Status:   PERFECT MATCH\n")


def central_subvolume(raw, n):
    if n > min(raw.shape):
        raise ValueError("SUBVOL_N is larger than the RAW volume.")

    starts = [(d - n) // 2 for d in raw.shape]
    stops = [s + n for s in starts]
    sl = tuple(slice(s, e) for s, e in zip(starts, stops))

    print("=" * 78)
    print("EXTRACTING CENTRAL SUBVOLUME")
    print("=" * 78)
    print(f"Start indices: {tuple(starts)}")
    print(f"Stop indices:  {tuple(stops)}")
    print(f"Shape:         {(n, n, n)}")
    print("Copying only this subvolume to RAM...")

    phase = np.array(raw[sl], dtype=np.uint8, copy=True)
    values = np.unique(phase)
    if not np.all(np.isin(values, [0, 1])):
        raise RuntimeError(f"Unexpected values in subvolume: {values.tolist()}")

    print(f"Observed values: {values.tolist()}")
    print("Done.\n")
    return phase, tuple(starts), tuple(stops)


# =============================================================================
# FULL-3D REFERENCE DIRECTIONAL CORRELATION
# =============================================================================

def fibonacci_hemisphere(n):
    """Approximately uniform axial directions over one hemisphere."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n, dtype=float)
    z = (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    phi = golden * i

    dirs = np.column_stack([
        r * np.cos(phi),
        r * np.sin(phi),
        z,
    ])

    dirs = np.vstack([dirs, np.eye(3)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    key = np.round(dirs, 8)
    _, idx = np.unique(key, axis=0, return_index=True)
    return dirs[np.sort(idx)]


def correlation_length(r, C, threshold=np.exp(-1.0)):
    """First interpolated 1/e crossing."""
    for i in range(1, len(C)):
        if C[i] <= threshold:
            x1, x2 = r[i-1], r[i]
            y1, y2 = C[i-1], C[i]
            if np.isclose(y1, y2):
                return float(x2)
            return float(
                x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            )
    return np.nan


def directional_correlations_3d(phase, directions, max_lag, n_points, seed):
    """
    Monte-Carlo directional covariance estimator over the complete 3D
    subvolume, identical in construction to test 1.
    """
    rng = np.random.default_rng(seed)
    n = phase.shape[0]
    margin = int(np.ceil(max_lag)) + 2

    if 2 * margin >= n:
        raise ValueError("MAX_LAG is too large for this subvolume.")

    phi_pore = float(np.mean(phase == 0))
    variance = phi_pore * (1.0 - phi_pore)
    if variance <= 0:
        raise RuntimeError("Degenerate pore/solid volume.")

    base = rng.integers(
        low=margin,
        high=n - margin,
        size=(3, n_points),
        endpoint=False,
        dtype=np.int32,
    )

    chi0 = 1.0 - phase[base[0], base[1], base[2]].astype(np.float32)
    f0 = chi0 - phi_pore

    lags = np.arange(max_lag + 1, dtype=np.float32)
    all_C = np.empty((len(directions), len(lags)), dtype=np.float64)
    all_ell = np.empty(len(directions), dtype=np.float64)

    print("=" * 78)
    print("FULL-3D REFERENCE DIRECTIONAL CORRELATION")
    print("=" * 78)
    print(f"Directions:    {len(directions)}")
    print(f"Points/dir:    {n_points:,}")
    print(f"Maximum lag:   {max_lag} voxels")
    print(f"Subvolume phi: {phi_pore:.6f}")
    print()

    t0 = time.time()

    for j, u in enumerate(directions):
        coords = (
            base[:, :, None].astype(np.float32)
            + u[:, None, None].astype(np.float32) * lags[None, None, :]
        )

        sampled_solid = map_coordinates(
            phase,
            coords.reshape(3, -1),
            order=1,
            mode="nearest",
            prefilter=False,
            output=np.float32,
        ).reshape(n_points, len(lags))

        chi_r = 1.0 - sampled_solid
        fr = chi_r - phi_pore
        C = np.mean(f0[:, None] * fr, axis=0) / variance
        C[0] = 1.0

        all_C[j] = C
        all_ell[j] = correlation_length(lags, C)

        if (j + 1) % max(1, len(directions) // 10) == 0 or j == len(directions) - 1:
            elapsed = time.time() - t0
            print(
                f"  {j+1:3d}/{len(directions)} directions "
                f"({100*(j+1)/len(directions):5.1f}%)   "
                f"elapsed {elapsed:6.1f} s"
            )

    print()
    return lags.astype(float), all_C, all_ell, phi_pore


# =============================================================================
# 2D SECTION ESTIMATOR
# =============================================================================

def autocorrelation_field_2d(pore_binary):
    """Overlap-normalized finite-domain covariance field."""
    x = pore_binary.astype(float)
    m = np.ones_like(x, dtype=float)
    phi = float(x.mean())
    f = x - phi

    numerator = fftconvolve(f, f[::-1, ::-1], mode="full")
    pairs = fftconvolve(m, m[::-1, ::-1], mode="full")

    cov = np.zeros_like(numerator, dtype=float)
    ok = pairs > 0.5
    cov[ok] = numerator[ok] / pairs[ok]

    center = (pore_binary.shape[0] - 1, pore_binary.shape[1] - 1)
    c0 = cov[center]
    if not np.isfinite(c0) or c0 <= 0:
        raise RuntimeError("Invalid 2D zero-lag covariance.")

    return cov / c0, center, phi


def directional_acf_2d(acf_field, center, theta_deg, max_lag):
    theta = np.deg2rad(theta_deg)
    r = np.arange(max_lag + 1, dtype=float)

    rr = center[0] + r * np.cos(theta)
    cc = center[1] + r * np.sin(theta)

    vals = map_coordinates(
        acf_field,
        np.vstack([rr, cc]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    vals[0] = 1.0
    return r, vals


def measure_section_2d(phase_section, B, theta_step_deg, max_lag,
                       family, offset, absolute_index):
    pore = (phase_section == 0)
    acf, center, phi = autocorrelation_field_2d(pore)

    theta = np.arange(0.0, 180.0, theta_step_deg, dtype=float)
    ell = np.empty_like(theta)

    for i, th in enumerate(theta):
        r, C = directional_acf_2d(acf, center, th, max_lag)
        ell[i] = correlation_length(r, C)

    u = np.column_stack([
        np.cos(np.deg2rad(theta)),
        np.sin(np.deg2rad(theta)),
    ])
    v = (B @ u.T).T

    return {
        "family": family,
        "offset": int(offset),
        "absolute_index": int(absolute_index),
        "B": np.asarray(B, dtype=float),
        "theta": theta,
        "vectors": v,
        "ell": ell,
        "phi": phi,
    }


def plane_family_definitions():
    """Return plane-family metadata in computational array coordinates."""
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    return {
        # fixed a2 index
        "a0-a1": {
            "fixed_axis": 2,
            "B": np.column_stack([e0, e1]),
        },
        # fixed a1 index
        "a0-a2": {
            "fixed_axis": 1,
            "B": np.column_stack([e0, e2]),
        },
        # fixed a0 index
        "a1-a2": {
            "fixed_axis": 0,
            "B": np.column_stack([e1, e2]),
        },
    }


def extract_axis_aligned_section(phase, fixed_axis, index):
    if fixed_axis == 0:
        return phase[index, :, :]
    if fixed_axis == 1:
        return phase[:, index, :]
    if fixed_axis == 2:
        return phase[:, :, index]
    raise ValueError("fixed_axis must be 0, 1, or 2")


def make_offset_pool():
    offsets = np.arange(-OFFSET_MAX, OFFSET_MAX + 1, OFFSET_STEP, dtype=int)
    if 0 not in offsets:
        raise RuntimeError("Offset pool must include zero.")
    return offsets


def systematic_offsets(n_per_orientation):
    """
    Deterministic offsets spanning the same +/- OFFSET_MAX interval.

    n=1 uses the center.  For n>1 the positions are equally spaced.  The
    chosen settings (1,3,5,7 with OFFSET_MAX=192 and OFFSET_STEP=32) all lie
    exactly on the precomputed offset pool.
    """
    if n_per_orientation == 1:
        return np.array([0], dtype=int)

    vals = np.rint(
        np.linspace(-OFFSET_MAX, OFFSET_MAX, n_per_orientation)
    ).astype(int)

    pool = set(make_offset_pool().tolist())
    if not all(int(v) in pool for v in vals):
        raise RuntimeError(
            f"Systematic offsets {vals.tolist()} are not all present in the "
            "precomputed offset pool. Adjust OFFSET_STEP/OFFSET_MAX."
        )
    return vals


def precompute_section_pool(phase):
    """
    Measure directional lengths for every plane family and offset once.

    The expensive 2D FFT correlations are therefore not repeated during the
    random-resampling experiment.
    """
    definitions = plane_family_definitions()
    offsets = make_offset_pool()
    c = phase.shape[0] // 2

    if c - OFFSET_MAX < 0 or c + OFFSET_MAX >= phase.shape[0]:
        raise ValueError("OFFSET_MAX places a section outside the subvolume.")

    cache = {}

    print("=" * 78)
    print("PRECOMPUTING SECTION-OFFSET POOL")
    print("=" * 78)
    print(f"Plane families:      {len(definitions)}")
    print(f"Offsets/family:      {len(offsets)}")
    print(f"Total 2D sections:   {len(definitions) * len(offsets)}")
    print(f"Offsets:             {offsets.tolist()}")
    print()

    t0 = time.time()
    counter = 0
    total = len(definitions) * len(offsets)

    for family, meta in definitions.items():
        for offset in offsets:
            idx = c + int(offset)
            img = extract_axis_aligned_section(
                phase,
                fixed_axis=meta["fixed_axis"],
                index=idx,
            )

            result = measure_section_2d(
                img,
                meta["B"],
                theta_step_deg=THETA_STEP_2D_DEG,
                max_lag=MAX_LAG,
                family=family,
                offset=offset,
                absolute_index=idx,
            )
            cache[(family, int(offset))] = result

            counter += 1
            n_valid = int(np.sum(np.isfinite(result["ell"])))
            print(
                f"  {counter:2d}/{total}  {family:6s}  "
                f"offset={offset:+4d}  phi={result['phi']:.6f}  "
                f"valid={n_valid}/{len(result['ell'])}"
            )

    print(f"Section pool completed in {time.time() - t0:.1f} s.\n")
    return cache, offsets


# =============================================================================
# CONFIGURATION RECONSTRUCTION / METRICS
# =============================================================================

def collect_sections(cache, offsets_by_family):
    sections = []
    for family, offsets in offsets_by_family.items():
        for offset in offsets:
            sections.append(cache[(family, int(offset))])
    return sections


def reconstruct_sparse_q3d(section_results):
    vectors = []
    ell = []

    for s in section_results:
        good = np.isfinite(s["ell"]) & (s["ell"] > 0)
        vectors.append(s["vectors"][good])
        ell.append(s["ell"][good])

    vectors = np.vstack(vectors)
    ell = np.concatenate(ell)

    Q, rank, cond, eta, used_v, used_ell = (
        fit_q3d_from_directional_lengths(vectors, ell)
    )
    return Q, rank, cond, eta, used_v, used_ell


def configuration_metrics(sections, Q_ref, phi_sub, orientation_reliable):
    Q, rank, cond, eta, vectors, ell = reconstruct_sparse_q3d(sections)

    E_Q = tensor_shape_error(Q, Q_ref)
    A_sparse = anisotropy_ratio(Q)

    ell_ref_n, axes_ref = principal_lengths_and_axes(normalize_det_spd(Q_ref))
    ell_sparse_n, axes_sparse = principal_lengths_and_axes(normalize_det_spd(Q))

    major_axis_error = axial_angle_error_deg(axes_ref[:, 0], axes_sparse[:, 0])

    # Porosity representativity metrics.
    phi_all = float(np.mean([s["phi"] for s in sections]))

    family_names = sorted(set(s["family"] for s in sections))
    family_means = {}
    for family in family_names:
        vals = [s["phi"] for s in sections if s["family"] == family]
        family_means[family] = float(np.mean(vals))

    phi_family_rms = float(np.sqrt(np.mean([
        (family_means[f] - phi_sub)**2 for f in family_names
    ])))

    return {
        "Q": Q,
        "rank": rank,
        "condition": cond,
        "eta": eta,
        "E_Q": E_Q,
        "A_sparse": A_sparse,
        "ell_sparse_n": ell_sparse_n,
        "major_axis_error_deg": major_axis_error,
        "orientation_reliable": bool(orientation_reliable),
        "phi_all": phi_all,
        "phi_family_rms": phi_family_rms,
        "family_means": family_means,
        "vectors": vectors,
        "ell": ell,
    }


def run_systematic_configs(cache, Q_ref, phi_sub, orientation_reliable):
    results = []
    families = list(plane_family_definitions().keys())

    print("=" * 78)
    print("SYSTEMATIC OFFSET CONFIGURATIONS")
    print("=" * 78)

    for n in SECTIONS_PER_ORIENTATION:
        offsets = systematic_offsets(n)
        offsets_by_family = {f: offsets for f in families}
        sections = collect_sections(cache, offsets_by_family)
        metrics = configuration_metrics(
            sections, Q_ref, phi_sub, orientation_reliable
        )

        row = {
            "mode": "systematic",
            "trial": 0,
            "n_per_orientation": int(n),
            "total_sections": int(3*n),
            "offsets_by_family": {
                f: [int(v) for v in offsets] for f in families
            },
            **metrics,
        }
        results.append(row)

        print(
            f"n/orientation={n:2d}  total={3*n:2d}  "
            f"offsets={offsets.tolist()}  "
            f"E_Q={100*metrics['E_Q']:6.2f}%  "
            f"A={metrics['A_sparse']:.4f}  "
            f"eta={100*metrics['eta']:.2f}%  "
            f"phi-RMS={100*metrics['phi_family_rms']:.3f} pp"
        )

    print()
    return results


def run_random_configs(cache, offset_pool, Q_ref, phi_sub,
                       orientation_reliable):
    rng = np.random.default_rng(SEED + 2000)
    families = list(plane_family_definitions().keys())
    results = []

    print("=" * 78)
    print("RANDOM POSITION RESAMPLING")
    print("=" * 78)
    print(f"Random configurations/count: {N_RANDOM_CONFIGS}")
    print()

    for n in SECTIONS_PER_ORIENTATION:
        t0 = time.time()
        local = []

        for trial in range(N_RANDOM_CONFIGS):
            offsets_by_family = {}
            for family in families:
                chosen = np.sort(
                    rng.choice(offset_pool, size=n, replace=False)
                ).astype(int)
                offsets_by_family[family] = chosen

            sections = collect_sections(cache, offsets_by_family)
            metrics = configuration_metrics(
                sections, Q_ref, phi_sub, orientation_reliable
            )

            row = {
                "mode": "random",
                "trial": int(trial + 1),
                "n_per_orientation": int(n),
                "total_sections": int(3*n),
                "offsets_by_family": {
                    f: [int(v) for v in offsets_by_family[f]]
                    for f in families
                },
                **metrics,
            }
            results.append(row)
            local.append(row)

        err = np.array([r["E_Q"] for r in local])
        Avec = np.array([r["A_sparse"] for r in local])
        phi_rms = np.array([r["phi_family_rms"] for r in local])

        print(
            f"n/orientation={n:2d}  total={3*n:2d}  "
            f"E_Q median={100*np.median(err):6.2f}%  "
            f"IQR=[{100*np.quantile(err,0.25):.2f}, "
            f"{100*np.quantile(err,0.75):.2f}]%  "
            f"A median={np.median(Avec):.4f}  "
            f"phi-RMS median={100*np.median(phi_rms):.3f} pp  "
            f"elapsed={time.time()-t0:.1f}s"
        )

    print()
    return results


# =============================================================================
# CSV / REPORT
# =============================================================================

def offsets_to_text(offsets_by_family):
    parts = []
    for family in sorted(offsets_by_family):
        vals = ",".join(str(int(v)) for v in offsets_by_family[family])
        parts.append(f"{family}:{vals}")
    return " | ".join(parts)


def write_results_csv(all_results):
    fields = [
        "mode",
        "trial",
        "n_per_orientation",
        "total_sections",
        "offsets",
        "tensor_shape_error",
        "anisotropy_sparse",
        "tensor_fit_residual",
        "rank",
        "condition_number",
        "mean_section_porosity",
        "family_porosity_rms",
        "major_axis_error_deg",
        "orientation_reliable",
        "phi_mean_a0_a1",
        "phi_mean_a0_a2",
        "phi_mean_a1_a2",
    ]

    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in all_results:
            fm = r["family_means"]
            writer.writerow({
                "mode": r["mode"],
                "trial": r["trial"],
                "n_per_orientation": r["n_per_orientation"],
                "total_sections": r["total_sections"],
                "offsets": offsets_to_text(r["offsets_by_family"]),
                "tensor_shape_error": f"{r['E_Q']:.10g}",
                "anisotropy_sparse": f"{r['A_sparse']:.10g}",
                "tensor_fit_residual": f"{r['eta']:.10g}",
                "rank": r["rank"],
                "condition_number": f"{r['condition']:.10g}",
                "mean_section_porosity": f"{r['phi_all']:.10g}",
                "family_porosity_rms": f"{r['phi_family_rms']:.10g}",
                "major_axis_error_deg": f"{r['major_axis_error_deg']:.10g}",
                "orientation_reliable": int(r["orientation_reliable"]),
                "phi_mean_a0_a1": f"{fm['a0-a1']:.10g}",
                "phi_mean_a0_a2": f"{fm['a0-a2']:.10g}",
                "phi_mean_a1_a2": f"{fm['a1-a2']:.10g}",
            })


def write_section_measurements_csv(cache):
    fields = [
        "family",
        "offset",
        "absolute_index",
        "section_porosity",
        "theta_deg",
        "v0",
        "v1",
        "v2",
        "ell_pixels",
    ]

    with open(SECTION_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for key in sorted(cache):
            s = cache[key]
            for th, v, ell in zip(s["theta"], s["vectors"], s["ell"]):
                writer.writerow({
                    "family": s["family"],
                    "offset": s["offset"],
                    "absolute_index": s["absolute_index"],
                    "section_porosity": f"{s['phi']:.10g}",
                    "theta_deg": f"{th:.10g}",
                    "v0": f"{v[0]:.10g}",
                    "v1": f"{v[1]:.10g}",
                    "v2": f"{v[2]:.10g}",
                    "ell_pixels": f"{ell:.10g}",
                })


def summarize_random(random_results, n):
    rows = [r for r in random_results if r["n_per_orientation"] == n]

    def q(field, p):
        vals = np.array([r[field] for r in rows], dtype=float)
        return float(np.quantile(vals, p))

    return {
        "E_med": q("E_Q", 0.50),
        "E_q1": q("E_Q", 0.25),
        "E_q3": q("E_Q", 0.75),
        "A_med": q("A_sparse", 0.50),
        "A_q1": q("A_sparse", 0.25),
        "A_q3": q("A_sparse", 0.75),
        "eta_med": q("eta", 0.50),
        "phi_rms_med": q("phi_family_rms", 0.50),
    }


def write_report(starts, stops, phi_sub, Q_ref, rank_ref, cond_ref, eta_ref,
                 A_ref, major_gap, orientation_reliable,
                 systematic_results, random_results, rho, rho_p):
    ell_ref_n, _ = principal_lengths_and_axes(normalize_det_spd(Q_ref))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Bentheimer real-rock tensor validation: test 2\n")
        f.write("=" * 76 + "\n\n")
        f.write(f"RAW file: {RAW_FILE}\n")
        f.write(f"RAW shape: {RAW_SHAPE}\n")
        f.write("RAW convention: 0=pore, 1=solid\n")
        f.write("Axes a0/a1/a2 are computational array coordinates.\n\n")

        f.write(f"Subvolume shape: {(SUBVOL_N, SUBVOL_N, SUBVOL_N)}\n")
        f.write(f"Subvolume start: {starts}\n")
        f.write(f"Subvolume stop:  {stops}\n")
        f.write(f"Exact subvolume porosity: {phi_sub:.8f}\n\n")

        f.write("Reference tensor Q3D_REF\n")
        f.write(np.array2string(Q_ref, precision=8) + "\n")
        f.write(
            f"rank={rank_ref}, condition={cond_ref:.6g}, "
            f"eta={eta_ref:.8f}, A_ref={A_ref:.8f}\n"
        )
        f.write(
            "Normalized principal lengths: "
            + np.array2string(ell_ref_n, precision=6) + "\n"
        )
        f.write(f"Major-axis relative spectral gap: {major_gap:.8f}\n")
        f.write(f"Orientation reliable by configured criterion: {orientation_reliable}\n")
        f.write(
            "Orientation errors are not primary when the principal spectral "
            "gap is small.\n\n"
        )

        f.write("Systematic configurations\n")
        f.write("-" * 76 + "\n")
        for r in systematic_results:
            f.write(
                f"n/orientation={r['n_per_orientation']}, "
                f"total={r['total_sections']}, "
                f"offsets={offsets_to_text(r['offsets_by_family'])}\n"
            )
            f.write(
                f"  E_Q={r['E_Q']:.8f} ({100*r['E_Q']:.3f}%), "
                f"A={r['A_sparse']:.8f}, eta={r['eta']:.8f}, "
                f"phi_mean={r['phi_all']:.8f}, "
                f"phi_family_RMS={r['phi_family_rms']:.8f}\n"
            )
            f.write(
                "  family porosities: "
                + ", ".join(
                    f"{k}={v:.8f}" for k, v in sorted(r["family_means"].items())
                )
                + "\n"
            )
            f.write(
                f"  major-axis error={r['major_axis_error_deg']:.4f} deg "
                f"(reliable={r['orientation_reliable']})\n"
            )

        f.write("\nRandom position resampling\n")
        f.write("-" * 76 + "\n")
        f.write(f"Configurations per n: {N_RANDOM_CONFIGS}\n")
        for n in SECTIONS_PER_ORIENTATION:
            s = summarize_random(random_results, n)
            f.write(
                f"n/orientation={n}, total={3*n}: "
                f"E_Q median={100*s['E_med']:.3f}% "
                f"IQR=[{100*s['E_q1']:.3f}, {100*s['E_q3']:.3f}]%; "
                f"A median={s['A_med']:.6f} "
                f"IQR=[{s['A_q1']:.6f}, {s['A_q3']:.6f}]; "
                f"eta median={100*s['eta_med']:.3f}%; "
                f"phi-family-RMS median={100*s['phi_rms_med']:.3f} pp\n"
            )

        f.write("\nRepresentativity/error association\n")
        f.write("-" * 76 + "\n")
        f.write(
            "Spearman correlation between family-porosity RMS mismatch and "
            f"tensor-shape error across all random configurations: "
            f"rho={rho:.6f}, p={rho_p:.6g}\n"
        )
        f.write(
            "This association is diagnostic only: porosity is a scalar and "
            "cannot by itself explain all tensor-shape variability.\n"
        )


# =============================================================================
# FIGURE
# =============================================================================

def panel_label(ax, label, x=0.01, y=0.99):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )


def grouped_random_values(random_results, n, field, scale=1.0):
    return np.array([
        r[field] * scale
        for r in random_results
        if r["n_per_orientation"] == n
    ], dtype=float)


def build_figure(cache, offset_pool, phi_sub, Q_ref, A_ref,
                 systematic_results, random_results, rho):
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

    families = list(plane_family_definitions().keys())
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    # ------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, 0])
    for i, family in enumerate(families):
        phi = np.array([cache[(family, int(o))]["phi"] for o in offset_pool])
        ax.plot(
            offset_pool,
            100.0 * phi,
            marker="o",
            markersize=3.7,
            label=family,
            color=colors[i],
        )
    ax.axhline(100.0 * phi_sub, linestyle="--", color="black", linewidth=1.2,
               label="3D subvolume")
    ax.set_xlabel("Section offset from center (pixels)")
    ax.set_ylabel("Section porosity (%)")
    ax.set_title("Spatial variability of section porosity")
    ax.legend(frameon=False, fontsize=7.4)
    panel_label(ax, "(a)")

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    nvals = np.array([r["n_per_orientation"] for r in systematic_results])
    phi_rms = 100.0 * np.array([r["phi_family_rms"] for r in systematic_results])
    ax.plot(nvals, phi_rms, marker="o")
    ax.set_xticks(nvals)
    ax.set_xlabel("Sections per orientation")
    ax.set_ylabel("Family porosity RMS mismatch (pp)")
    ax.set_title("Systematic representativity")
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    total = np.array([r["total_sections"] for r in systematic_results])
    err = 100.0 * np.array([r["E_Q"] for r in systematic_results])
    ax.plot(total, err, marker="o")
    ax.set_xticks(total)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel(r"Tensor-shape error $E_Q$ (%)")
    ax.set_title("Systematic sparse reconstruction")
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0])
    box_E = [
        grouped_random_values(random_results, n, "E_Q", scale=100.0)
        for n in SECTIONS_PER_ORIENTATION
    ]
    ax.boxplot(
        box_E,
        tick_labels=[str(n) for n in SECTIONS_PER_ORIENTATION],
        widths=0.58,
        showfliers=False,
        medianprops=dict(linewidth=1.7, color="black"),
        boxprops=dict(linewidth=1.0, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )
    sys_E = 100.0 * np.array([r["E_Q"] for r in systematic_results])
    ax.scatter(
        np.arange(1, len(SECTIONS_PER_ORIENTATION) + 1),
        sys_E,
        marker="D",
        s=28,
        color="black",
        label="Systematic",
        zorder=4,
    )
    ax.set_xlabel("Sections per orientation")
    ax.set_ylabel(r"Tensor-shape error $E_Q$ (%)")
    ax.set_title("Position uncertainty decreases with replication")
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    box_A = [
        grouped_random_values(random_results, n, "A_sparse")
        for n in SECTIONS_PER_ORIENTATION
    ]
    ax.boxplot(
        box_A,
        tick_labels=[str(n) for n in SECTIONS_PER_ORIENTATION],
        widths=0.58,
        showfliers=False,
        medianprops=dict(linewidth=1.7, color="black"),
        boxprops=dict(linewidth=1.0, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )
    ax.axhline(A_ref, linestyle="--", color="black", linewidth=1.2,
               label=f"3D reference ({A_ref:.3f})")
    sys_A = np.array([r["A_sparse"] for r in systematic_results])
    ax.scatter(
        np.arange(1, len(SECTIONS_PER_ORIENTATION) + 1),
        sys_A,
        marker="D",
        s=28,
        color="black",
        zorder=4,
        label="Systematic",
    )
    ax.set_xlabel("Sections per orientation")
    ax.set_ylabel(r"Recovered anisotropy $A_{sparse}$")
    ax.set_title("Anisotropy estimate stabilizes with offsets")
    ax.legend(frameon=False, fontsize=7.4)
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    for i, n in enumerate(SECTIONS_PER_ORIENTATION):
        rows = [r for r in random_results if r["n_per_orientation"] == n]
        x = 100.0 * np.array([r["phi_family_rms"] for r in rows])
        y = 100.0 * np.array([r["E_Q"] for r in rows])
        ax.scatter(
            x, y,
            s=16,
            alpha=0.45,
            edgecolors="none",
            label=f"n={n}",
            color=colors[i % len(colors)],
        )
    ax.set_xlabel("Family porosity RMS mismatch (pp)")
    ax.set_ylabel(r"Tensor-shape error $E_Q$ (%)")
    ax.set_title("Representativity and tensor error")
    ax.text(
        0.97, 0.95,
        rf"Spearman $\rho$ = {rho:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=7.2)
    panel_label(ax, "(f)")

    for ax in fig.axes:
        ax.grid(False)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.965)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\nRAW file not found:\n{RAW_FILE.resolve()}\n\n"
            "Place this script in the same directory as the RAW file or edit "
            "RAW_FILE at the top of the script."
        )

    validate_raw(RAW_FILE)

    raw = np.memmap(
        RAW_FILE,
        dtype=RAW_DTYPE,
        mode="r",
        shape=RAW_SHAPE,
        order="C",
    )

    phase, starts, stops = central_subvolume(raw, SUBVOL_N)
    del raw

    phi_sub = float(np.mean(phase == 0))

    print("=" * 78)
    print("SUBVOLUME SUMMARY")
    print("=" * 78)
    print(f"Exact porosity: {phi_sub:.8f} ({100*phi_sub:.3f}%)")
    print(f"Pore voxels:    {np.count_nonzero(phase == 0):,}")
    print(f"Solid voxels:   {np.count_nonzero(phase == 1):,}")
    print()

    # ------------------------------------------------------------------
    # 1) Full-3D reference tensor -- same settings as test 1
    # ------------------------------------------------------------------
    ref_dirs = fibonacci_hemisphere(N_REF_DIRECTIONS)
    _, _, ref_ell, phi_check = directional_correlations_3d(
        phase,
        ref_dirs,
        max_lag=MAX_LAG,
        n_points=N_REF_POINTS,
        seed=SEED,
    )

    if not np.isclose(phi_sub, phi_check):
        raise RuntimeError("Internal porosity consistency check failed.")

    valid_ref = np.isfinite(ref_ell) & (ref_ell > 0)
    print(f"Valid 3D correlation lengths: {np.sum(valid_ref)}/{len(ref_ell)}")
    if np.sum(valid_ref) < 0.80 * len(ref_ell):
        print(
            "WARNING: fewer than 80% of reference directions reached the 1/e "
            "threshold. Consider increasing MAX_LAG."
        )

    Q_ref, rank_ref, cond_ref, eta_ref, _, _ = (
        fit_q3d_from_directional_lengths(ref_dirs, ref_ell)
    )
    A_ref = anisotropy_ratio(Q_ref)
    major_gap = reference_major_axis_gap(Q_ref)
    orientation_reliable = bool(major_gap >= ORIENTATION_GAP_MIN)

    print("=" * 78)
    print("REFERENCE TENSOR")
    print("=" * 78)
    print(Q_ref)
    print(f"rank / condition:            {rank_ref} / {cond_ref:.4f}")
    print(f"tensor fit residual:         {100*eta_ref:.3f}%")
    print(f"A_ref:                       {A_ref:.4f}")
    print(f"major-axis spectral gap:     {100*major_gap:.3f}%")
    print(f"orientation reliable flag:   {orientation_reliable}")
    if not orientation_reliable:
        print(
            "NOTE: major-axis orientation is not treated as a primary metric "
            "because the reference principal-length gap is small."
        )
    print()

    # ------------------------------------------------------------------
    # 2) Precompute 13 offsets x 3 plane families
    # ------------------------------------------------------------------
    cache, offset_pool = precompute_section_pool(phase)

    # ------------------------------------------------------------------
    # 3) Deterministic systematic configurations
    # ------------------------------------------------------------------
    systematic_results = run_systematic_configs(
        cache, Q_ref, phi_sub, orientation_reliable
    )

    # ------------------------------------------------------------------
    # 4) Random position-resampling configurations
    # ------------------------------------------------------------------
    random_results = run_random_configs(
        cache, offset_pool, Q_ref, phi_sub, orientation_reliable
    )

    # ------------------------------------------------------------------
    # 5) Diagnostic association: scalar representativity vs tensor error
    # ------------------------------------------------------------------
    x_all = np.array([r["phi_family_rms"] for r in random_results])
    y_all = np.array([r["E_Q"] for r in random_results])
    rho, rho_p = spearmanr(x_all, y_all)

    print("=" * 78)
    print("OVERALL INTERPRETATION METRICS")
    print("=" * 78)
    print(
        "Spearman correlation between family-porosity RMS mismatch and "
        f"E_Q: rho={rho:.4f}, p={rho_p:.4g}"
    )
    print(
        "NOTE: this correlation is diagnostic only; scalar porosity cannot "
        "fully describe tensorial representativity."
    )
    print()

    # ------------------------------------------------------------------
    # 6) Save data / report / figure
    # ------------------------------------------------------------------
    all_results = systematic_results + random_results
    write_results_csv(all_results)
    write_section_measurements_csv(cache)
    write_report(
        starts, stops, phi_sub,
        Q_ref, rank_ref, cond_ref, eta_ref, A_ref,
        major_gap, orientation_reliable,
        systematic_results, random_results,
        rho, rho_p,
    )
    build_figure(
        cache, offset_pool, phi_sub, Q_ref, A_ref,
        systematic_results, random_results, rho,
    )

    print("=" * 78)
    print("FILES GENERATED")
    print("=" * 78)
    print(PNG_PATH)
    print(SVG_PATH)
    print(REPORT_PATH)
    print(RESULTS_CSV_PATH)
    print(SECTION_CSV_PATH)
    print()
    print("Bentheimer test 2 completed successfully.")


if __name__ == "__main__":
    main()