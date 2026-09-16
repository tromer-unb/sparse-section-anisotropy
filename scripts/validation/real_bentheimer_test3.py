#!/usr/bin/env python3
"""
Bentheimer real-rock validation: test 3
========================================

Goal
----
Test whether the section-replication behavior observed in one central 512^3
subvolume is spatially reproducible throughout the full 2500^3 Bentheimer ROI.

The ROI is sampled by a non-overlapping 3 x 3 x 3 grid of 27 subvolumes.  For
each subvolume we compute:

    complete 3D subvolume -> Q3D_REF

and reconstruct Q3D_SPARSE from three orthogonal section families using

    1, 3, 5, 7 sections per orientation
    = 3, 9, 15, 21 total sections.

For every section count, random offset resampling is performed inside each
subvolume.  The median error of those random configurations is then treated as
the position-robust sparse estimate for that spatial replicate.

Scientific distinction
-----------------------
    orientation diversity  -> tensor observability
    offset replication     -> within-subvolume representativity
    spatial subvolumes      -> reproducibility across the real rock ROI

IMPORTANT
---------
These 27 subvolumes are non-overlapping spatial replicates from ONE rock plug /
ONE ROI.  They are not 27 independent rock specimens.  Statistical summaries
are therefore spatial-replicate summaries, not specimen-level population
inference.

RAW convention already validated:
    RAW value 0 = pore
    RAW value 1 = solid

Array axes are called a0, a1, a2 until the physical axis mapping is confirmed
from dataset metadata.

Outputs
-------
  bentheimer_test3.png
  bentheimer_test3.svg
  bentheimer_test3_report.txt
  bentheimer_test3_subvolumes.csv
  bentheimer_test3_summary.csv
  bentheimer_test3_random.csv
  bentheimer_test3_cache/        (restart-safe per-subvolume JSON cache)

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

Run
---
  python3 real_bentheimer_test3.py

The full 2500^3 RAW is opened with numpy.memmap.  Only one 512^3 subvolume is
copied into RAM at a time.
"""

from pathlib import Path
import csv
import gc
import json
import os
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


# =============================================================================
# USER SETTINGS
# =============================================================================

RAW_FILE = Path(
    "kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)
RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8

# Spatial-replication design.
SUBVOL_N = 512
GRID_CENTERS = (384, 1250, 2116)

# Directional-correlation settings.  Kept identical to tests 1 and 2.
MAX_LAG = 48
THETA_STEP_2D_DEG = 5.0
N_REF_DIRECTIONS = 64
N_REF_POINTS = 12_000

# Section-position experiment inside each spatial subvolume.
SECTIONS_PER_ORIENTATION = (1, 3, 5, 7)
OFFSET_MAX = 192
OFFSET_STEP = 32
N_RANDOM_CONFIGS = 200

# Spectral-gap criterion for interpreting the major-axis orientation.
ORIENTATION_GAP_MIN = 0.10

SEED = 20260907
DPI = 800

# False is manuscript-quality mode.  True keeps all 27 subvolumes but reduces
# stochastic sampling for a faster pipeline check.
QUICK_MODE = False

# Restart behavior.  Completed subvolumes are stored as JSON files.  With
# FORCE_RECOMPUTE=False, rerunning the script skips completed subvolumes.
FORCE_RECOMPUTE = False


# =============================================================================
# OUTPUTS / STYLE
# =============================================================================

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "bentheimer_test3.png"
SVG_PATH = OUTDIR / "bentheimer_test3.svg"
REPORT_PATH = OUTDIR / "bentheimer_test3_report.txt"
SUBVOL_CSV_PATH = OUTDIR / "bentheimer_test3_subvolumes.csv"
SUMMARY_CSV_PATH = OUTDIR / "bentheimer_test3_summary.csv"
RANDOM_CSV_PATH = OUTDIR / "bentheimer_test3_random.csv"
CACHE_DIR = OUTDIR / "bentheimer_test3_cache"

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
    """OLS in 1/ell^2 followed by the same SPD projection as tests 1/2."""
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

    Q = 0.5 * (Q + Q.T)
    w, E = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-7 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = E @ np.diag(w) @ E.T

    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1])

    pred = ell_from_q3d(Q, V)
    eta = float(np.sqrt(np.sum((L - pred)**2) / np.sum(L**2)))

    return Q, rank, cond, eta, V, L


def principal_lengths_and_axes(Q):
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    w = np.maximum(w, 1e-14)
    ell = 1.0 / np.sqrt(w)
    order = np.argsort(ell)[::-1]
    return ell[order], V[:, order]


def anisotropy_ratio(Q):
    ell, _ = principal_lengths_and_axes(Q)
    return float(ell[0] / ell[-1])


def axial_angle_error_deg(v1, v2):
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
    ell, _ = principal_lengths_and_axes(normalize_det_spd(Q_ref))
    return float((ell[0] - ell[1]) / ell[0])


# =============================================================================
# RAW / SPATIAL GRID
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


def spatial_grid():
    """Return metadata for the 3 x 3 x 3 non-overlapping subvolume grid."""
    half = SUBVOL_N // 2
    rows = []
    counter = 0

    for i, c0 in enumerate(GRID_CENTERS):
        for j, c1 in enumerate(GRID_CENTERS):
            for k, c2 in enumerate(GRID_CENTERS):
                center = (int(c0), int(c1), int(c2))
                start = tuple(c - half for c in center)
                stop = tuple(s + SUBVOL_N for s in start)

                if any(s < 0 for s in start) or any(e > RAW_SHAPE[d] for d, e in enumerate(stop)):
                    raise ValueError(f"Subvolume {center} lies outside the RAW volume.")

                rows.append({
                    "id": f"G{counter:02d}",
                    "grid_i": i,
                    "grid_j": j,
                    "grid_k": k,
                    "center": center,
                    "start": start,
                    "stop": stop,
                })
                counter += 1

    # Verify non-overlap of axis intervals.
    intervals = [(c - half, c + half) for c in GRID_CENTERS]
    for a in range(len(intervals) - 1):
        if intervals[a][1] > intervals[a+1][0]:
            raise RuntimeError("Configured spatial subvolumes overlap along an axis.")

    return rows


def extract_subvolume(raw, meta):
    sl = tuple(slice(s, e) for s, e in zip(meta["start"], meta["stop"]))
    phase = np.array(raw[sl], dtype=np.uint8, copy=True)
    values = np.unique(phase)
    if not np.all(np.isin(values, [0, 1])):
        raise RuntimeError(
            f"Unexpected values in {meta['id']}: {values.tolist()}"
        )
    return phase


# =============================================================================
# FULL-3D REFERENCE DIRECTIONAL CORRELATION
# =============================================================================

def fibonacci_hemisphere(n):
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
    rng = np.random.default_rng(seed)
    n = phase.shape[0]
    margin = int(np.ceil(max_lag)) + 2

    if 2 * margin >= n:
        raise ValueError("MAX_LAG is too large for this subvolume.")

    phi_pore = float(np.mean(phase == 0))
    variance = phi_pore * (1.0 - phi_pore)
    if variance <= 0:
        raise RuntimeError("Degenerate pore/solid subvolume.")

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

    all_ell = np.empty(len(directions), dtype=np.float64)

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
        all_ell[j] = correlation_length(lags, C)

    return all_ell, phi_pore


# =============================================================================
# 2D SECTION ESTIMATOR
# =============================================================================

def autocorrelation_field_2d(pore_binary):
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


def measure_section_2d(phase_section, B, family, offset, absolute_index):
    pore = (phase_section == 0)
    acf, center, phi = autocorrelation_field_2d(pore)

    theta = np.arange(0.0, 180.0, THETA_STEP_2D_DEG, dtype=float)
    ell = np.empty_like(theta)

    for i, th in enumerate(theta):
        r, C = directional_acf_2d(acf, center, th, MAX_LAG)
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
        "theta": theta,
        "vectors": v,
        "ell": ell,
        "phi": float(phi),
    }


def plane_family_definitions():
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    return {
        "a0-a1": {
            "fixed_axis": 2,
            "B": np.column_stack([e0, e1]),
        },
        "a0-a2": {
            "fixed_axis": 1,
            "B": np.column_stack([e0, e2]),
        },
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
    if n_per_orientation == 1:
        return np.array([0], dtype=int)

    vals = np.rint(
        np.linspace(-OFFSET_MAX, OFFSET_MAX, n_per_orientation)
    ).astype(int)

    pool = set(make_offset_pool().tolist())
    if not all(int(v) in pool for v in vals):
        raise RuntimeError(
            f"Systematic offsets {vals.tolist()} are not all in the pool."
        )
    return vals


def precompute_section_pool(phase):
    definitions = plane_family_definitions()
    offsets = make_offset_pool()
    c = phase.shape[0] // 2

    if c - OFFSET_MAX < 0 or c + OFFSET_MAX >= phase.shape[0]:
        raise ValueError("OFFSET_MAX places a section outside the subvolume.")

    cache = {}
    for family, meta in definitions.items():
        for offset in offsets:
            idx = c + int(offset)
            img = extract_axis_aligned_section(
                phase, meta["fixed_axis"], idx
            )
            cache[(family, int(offset))] = measure_section_2d(
                img,
                meta["B"],
                family=family,
                offset=offset,
                absolute_index=idx,
            )
    return cache, offsets


# =============================================================================
# CONFIGURATION RECONSTRUCTION
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
    return fit_q3d_from_directional_lengths(vectors, ell)


def configuration_metrics(sections, Q_ref, phi_sub, orientation_reliable):
    Q, rank, cond, eta, _, _ = reconstruct_sparse_q3d(sections)

    E_Q = tensor_shape_error(Q, Q_ref)
    A_sparse = anisotropy_ratio(Q)

    _, axes_ref = principal_lengths_and_axes(normalize_det_spd(Q_ref))
    _, axes_sparse = principal_lengths_and_axes(normalize_det_spd(Q))
    major_axis_error = axial_angle_error_deg(axes_ref[:, 0], axes_sparse[:, 0])

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
        "E_Q": float(E_Q),
        "A_sparse": float(A_sparse),
        "eta": float(eta),
        "rank": int(rank),
        "condition": float(cond),
        "phi_all": float(phi_all),
        "phi_family_rms": float(phi_family_rms),
        "major_axis_error_deg": float(major_axis_error),
        "orientation_reliable": bool(orientation_reliable),
        "family_means": family_means,
    }


def run_systematic_configs(cache, Q_ref, phi_sub, orientation_reliable):
    families = list(plane_family_definitions().keys())
    rows = []

    for n in SECTIONS_PER_ORIENTATION:
        offsets = systematic_offsets(n)
        offsets_by_family = {f: offsets.copy() for f in families}
        sections = collect_sections(cache, offsets_by_family)
        metrics = configuration_metrics(
            sections, Q_ref, phi_sub, orientation_reliable
        )
        rows.append({
            "n_per_orientation": int(n),
            "total_sections": int(3*n),
            "offsets": [int(v) for v in offsets],
            **metrics,
        })
    return rows


def run_random_configs(cache, offset_pool, Q_ref, phi_sub,
                       orientation_reliable, rng, n_random):
    families = list(plane_family_definitions().keys())
    rows = []

    for n in SECTIONS_PER_ORIENTATION:
        for trial in range(n_random):
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
            rows.append({
                "trial": int(trial + 1),
                "n_per_orientation": int(n),
                "total_sections": int(3*n),
                **metrics,
            })
    return rows


def summarize_random(random_rows):
    out = []
    for n in SECTIONS_PER_ORIENTATION:
        rows = [r for r in random_rows if r["n_per_orientation"] == n]

        def arr(field):
            return np.array([r[field] for r in rows], dtype=float)

        E = arr("E_Q")
        A = arr("A_sparse")
        eta = arr("eta")
        phi_rms = arr("phi_family_rms")
        angle = arr("major_axis_error_deg")

        out.append({
            "n_per_orientation": int(n),
            "total_sections": int(3*n),
            "E_med": float(np.median(E)),
            "E_q1": float(np.quantile(E, 0.25)),
            "E_q3": float(np.quantile(E, 0.75)),
            "A_med": float(np.median(A)),
            "A_q1": float(np.quantile(A, 0.25)),
            "A_q3": float(np.quantile(A, 0.75)),
            "eta_med": float(np.median(eta)),
            "phi_rms_med": float(np.median(phi_rms)),
            "major_axis_error_med": float(np.median(angle)),
        })
    return out


# =============================================================================
# PER-SUBVOLUME PROCESSING / RESTART CACHE
# =============================================================================

def json_atomic_write(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    os.replace(tmp, path)


def process_subvolume(raw, meta, ref_dirs, n_ref_points, n_random):
    t0 = time.time()
    print("=" * 78)
    print(
        f"SUBVOLUME {meta['id']}  grid=({meta['grid_i']},{meta['grid_j']},{meta['grid_k']})  "
        f"center={meta['center']}"
    )
    print("=" * 78)
    print(f"start={meta['start']}  stop={meta['stop']}")
    print("Copying 512^3 subvolume to RAM...")

    phase = extract_subvolume(raw, meta)
    phi_sub = float(np.mean(phase == 0))
    print(f"porosity={100*phi_sub:.3f}%")

    # Full-3D reference tensor.
    ref_ell, phi_check = directional_correlations_3d(
        phase,
        ref_dirs,
        max_lag=MAX_LAG,
        n_points=n_ref_points,
        seed=SEED + 1000 + int(meta["id"][1:]),
    )
    if not np.isclose(phi_sub, phi_check):
        raise RuntimeError("Internal porosity consistency check failed.")

    valid = np.isfinite(ref_ell) & (ref_ell > 0)
    if np.sum(valid) < 0.80 * len(ref_ell):
        print(
            f"WARNING: only {np.sum(valid)}/{len(ref_ell)} reference directions "
            "reached the 1/e threshold."
        )

    Q_ref, rank_ref, cond_ref, eta_ref, _, _ = (
        fit_q3d_from_directional_lengths(ref_dirs, ref_ell)
    )
    A_ref = anisotropy_ratio(Q_ref)
    major_gap = reference_major_axis_gap(Q_ref)
    orientation_reliable = bool(major_gap >= ORIENTATION_GAP_MIN)

    print(
        f"Q_REF: A={A_ref:.4f}, eta={100*eta_ref:.2f}%, "
        f"gap={100*major_gap:.2f}%, orientation_reliable={orientation_reliable}"
    )

    # Section pool.
    t_sections = time.time()
    cache, offset_pool = precompute_section_pool(phase)
    print(f"39 section measurements completed in {time.time()-t_sections:.1f} s")

    # Deterministic and random offset configurations.
    systematic = run_systematic_configs(
        cache, Q_ref, phi_sub, orientation_reliable
    )
    rng = np.random.default_rng(
        SEED + 50000 + 1000 * int(meta["id"][1:])
    )
    random_rows = run_random_configs(
        cache, offset_pool, Q_ref, phi_sub,
        orientation_reliable, rng, n_random
    )
    summary = summarize_random(random_rows)

    sys_by_n = {r["n_per_orientation"]: r for r in systematic}
    for row in summary:
        s = sys_by_n[row["n_per_orientation"]]
        row.update({
            "systematic_E_Q": s["E_Q"],
            "systematic_A": s["A_sparse"],
            "systematic_eta": s["eta"],
            "systematic_phi_rms": s["phi_family_rms"],
        })

    print("Random-median sparse recovery:")
    for row in summary:
        print(
            f"  n/orient={row['n_per_orientation']} total={row['total_sections']:2d}  "
            f"E_Q={100*row['E_med']:6.2f}% "
            f"IQR=[{100*row['E_q1']:.2f},{100*row['E_q3']:.2f}]%  "
            f"A={row['A_med']:.4f}  "
            f"phi-RMS={100*row['phi_rms_med']:.3f} pp"
        )

    Q_ref_list = [[float(v) for v in row] for row in Q_ref]
    result = {
        "meta": meta,
        "phi_sub": phi_sub,
        "Q_ref": Q_ref_list,
        "rank_ref": int(rank_ref),
        "condition_ref": float(cond_ref),
        "eta_ref": float(eta_ref),
        "A_ref": float(A_ref),
        "major_gap": float(major_gap),
        "orientation_reliable": bool(orientation_reliable),
        "valid_ref_lengths": int(np.sum(valid)),
        "n_ref_lengths": int(len(ref_ell)),
        "systematic": systematic,
        "summary": summary,
        "random": random_rows,
        "elapsed_seconds": float(time.time() - t0),
    }

    print(f"Subvolume {meta['id']} completed in {result['elapsed_seconds']:.1f} s.\n")

    del cache
    del phase
    gc.collect()
    return result


# =============================================================================
# FLATTEN / CSV
# =============================================================================

def load_all_cached(grid):
    results = []
    missing = []
    for meta in grid:
        path = CACHE_DIR / f"{meta['id']}.json"
        if not path.exists():
            missing.append(meta["id"])
            continue
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    if missing:
        raise RuntimeError(f"Missing cached results for: {missing}")
    return results


def write_csv_outputs(results):
    # Reference / spatial subvolume table.
    fields = [
        "subvolume_id", "grid_i", "grid_j", "grid_k",
        "center_a0", "center_a1", "center_a2",
        "start_a0", "start_a1", "start_a2",
        "stop_a0", "stop_a1", "stop_a2",
        "porosity", "A_ref", "eta_ref", "major_gap",
        "orientation_reliable", "rank_ref", "condition_ref",
        "q00", "q11", "q22", "q01", "q02", "q12",
    ]
    with open(SUBVOL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            m = r["meta"]
            Q = np.asarray(r["Q_ref"], dtype=float)
            w.writerow({
                "subvolume_id": m["id"],
                "grid_i": m["grid_i"], "grid_j": m["grid_j"], "grid_k": m["grid_k"],
                "center_a0": m["center"][0], "center_a1": m["center"][1], "center_a2": m["center"][2],
                "start_a0": m["start"][0], "start_a1": m["start"][1], "start_a2": m["start"][2],
                "stop_a0": m["stop"][0], "stop_a1": m["stop"][1], "stop_a2": m["stop"][2],
                "porosity": f"{r['phi_sub']:.10g}",
                "A_ref": f"{r['A_ref']:.10g}",
                "eta_ref": f"{r['eta_ref']:.10g}",
                "major_gap": f"{r['major_gap']:.10g}",
                "orientation_reliable": int(r["orientation_reliable"]),
                "rank_ref": r["rank_ref"],
                "condition_ref": f"{r['condition_ref']:.10g}",
                "q00": f"{Q[0,0]:.10g}", "q11": f"{Q[1,1]:.10g}", "q22": f"{Q[2,2]:.10g}",
                "q01": f"{Q[0,1]:.10g}", "q02": f"{Q[0,2]:.10g}", "q12": f"{Q[1,2]:.10g}",
            })

    # One spatial-replicate summary per section count.
    fields = [
        "subvolume_id", "n_per_orientation", "total_sections",
        "porosity", "A_ref", "eta_ref", "major_gap", "orientation_reliable",
        "random_E_median", "random_E_q1", "random_E_q3", "random_E_iqr_width",
        "random_A_median", "random_A_q1", "random_A_q3",
        "random_eta_median", "random_phi_rms_median", "random_major_axis_error_median",
        "systematic_E_Q", "systematic_A", "systematic_eta", "systematic_phi_rms",
    ]
    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for s in r["summary"]:
                w.writerow({
                    "subvolume_id": r["meta"]["id"],
                    "n_per_orientation": s["n_per_orientation"],
                    "total_sections": s["total_sections"],
                    "porosity": f"{r['phi_sub']:.10g}",
                    "A_ref": f"{r['A_ref']:.10g}",
                    "eta_ref": f"{r['eta_ref']:.10g}",
                    "major_gap": f"{r['major_gap']:.10g}",
                    "orientation_reliable": int(r["orientation_reliable"]),
                    "random_E_median": f"{s['E_med']:.10g}",
                    "random_E_q1": f"{s['E_q1']:.10g}",
                    "random_E_q3": f"{s['E_q3']:.10g}",
                    "random_E_iqr_width": f"{(s['E_q3']-s['E_q1']):.10g}",
                    "random_A_median": f"{s['A_med']:.10g}",
                    "random_A_q1": f"{s['A_q1']:.10g}",
                    "random_A_q3": f"{s['A_q3']:.10g}",
                    "random_eta_median": f"{s['eta_med']:.10g}",
                    "random_phi_rms_median": f"{s['phi_rms_med']:.10g}",
                    "random_major_axis_error_median": f"{s['major_axis_error_med']:.10g}",
                    "systematic_E_Q": f"{s['systematic_E_Q']:.10g}",
                    "systematic_A": f"{s['systematic_A']:.10g}",
                    "systematic_eta": f"{s['systematic_eta']:.10g}",
                    "systematic_phi_rms": f"{s['systematic_phi_rms']:.10g}",
                })

    # Full random-resampling table.
    fields = [
        "subvolume_id", "trial", "n_per_orientation", "total_sections",
        "tensor_shape_error", "anisotropy_sparse", "tensor_fit_residual",
        "rank", "condition_number", "mean_section_porosity",
        "family_porosity_rms", "major_axis_error_deg", "orientation_reliable",
    ]
    with open(RANDOM_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            sid = r["meta"]["id"]
            for x in r["random"]:
                w.writerow({
                    "subvolume_id": sid,
                    "trial": x["trial"],
                    "n_per_orientation": x["n_per_orientation"],
                    "total_sections": x["total_sections"],
                    "tensor_shape_error": f"{x['E_Q']:.10g}",
                    "anisotropy_sparse": f"{x['A_sparse']:.10g}",
                    "tensor_fit_residual": f"{x['eta']:.10g}",
                    "rank": x["rank"],
                    "condition_number": f"{x['condition']:.10g}",
                    "mean_section_porosity": f"{x['phi_all']:.10g}",
                    "family_porosity_rms": f"{x['phi_family_rms']:.10g}",
                    "major_axis_error_deg": f"{x['major_axis_error_deg']:.10g}",
                    "orientation_reliable": int(x["orientation_reliable"]),
                })


# =============================================================================
# AGGREGATE SUMMARIES
# =============================================================================

def summaries_for_n(results, n):
    rows = []
    for r in results:
        s = next(x for x in r["summary"] if x["n_per_orientation"] == n)
        rows.append((r, s))
    return rows


def aggregate_table(results):
    table = []
    for n in SECTIONS_PER_ORIENTATION:
        rows = summaries_for_n(results, n)
        E = np.array([s["E_med"] for _, s in rows])
        Aerr = np.array([abs(s["A_med"] - r["A_ref"]) for r, s in rows])
        width = np.array([s["E_q3"] - s["E_q1"] for _, s in rows])
        phi_rms = np.array([s["phi_rms_med"] for _, s in rows])

        table.append({
            "n": n,
            "total": 3*n,
            "E_median": float(np.median(E)),
            "E_q1": float(np.quantile(E, 0.25)),
            "E_q3": float(np.quantile(E, 0.75)),
            "Aerr_median": float(np.median(Aerr)),
            "IQRwidth_median": float(np.median(width)),
            "phi_rms_median": float(np.median(phi_rms)),
            "pct_E_le_10": float(100*np.mean(E <= 0.10)),
            "pct_E_le_15": float(100*np.mean(E <= 0.15)),
            "pct_E_le_20": float(100*np.mean(E <= 0.20)),
        })
    return table


# =============================================================================
# REPORT
# =============================================================================

def write_report(results, agg):
    phi = np.array([r["phi_sub"] for r in results])
    Aref = np.array([r["A_ref"] for r in results])
    eta = np.array([r["eta_ref"] for r in results])
    gaps = np.array([r["major_gap"] for r in results])
    reliable = np.array([r["orientation_reliable"] for r in results], dtype=bool)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Bentheimer real-rock tensor validation: test 3\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"RAW file: {RAW_FILE}\n")
        f.write(f"RAW shape: {RAW_SHAPE}\n")
        f.write("RAW convention: 0=pore, 1=solid\n")
        f.write("Axes a0/a1/a2 are computational array coordinates.\n\n")

        f.write("Spatial design\n")
        f.write("-" * 78 + "\n")
        f.write(f"Subvolume shape: {SUBVOL_N}^3\n")
        f.write(f"Grid centers along each axis: {GRID_CENTERS}\n")
        f.write(f"Number of non-overlapping subvolumes: {len(results)}\n")
        f.write(
            "These are spatial replicates within one Bentheimer ROI, not "
            "independent rock specimens.\n\n"
        )

        f.write("3D reference heterogeneity across spatial subvolumes\n")
        f.write("-" * 78 + "\n")
        f.write(
            f"Porosity: median={100*np.median(phi):.3f}%, "
            f"range=[{100*np.min(phi):.3f}, {100*np.max(phi):.3f}]%\n"
        )
        f.write(
            f"A_ref: median={np.median(Aref):.5f}, "
            f"range=[{np.min(Aref):.5f}, {np.max(Aref):.5f}]\n"
        )
        f.write(
            f"Reference tensor residual eta: median={100*np.median(eta):.3f}%, "
            f"range=[{100*np.min(eta):.3f}, {100*np.max(eta):.3f}]%\n"
        )
        f.write(
            f"Major-axis gap: median={100*np.median(gaps):.3f}%\n"
        )
        f.write(
            f"Orientation reliable in {np.sum(reliable)}/{len(reliable)} "
            f"subvolumes using gap >= {100*ORIENTATION_GAP_MIN:.1f}%.\n\n"
        )

        f.write("Sparse reconstruction across spatial replicates\n")
        f.write("-" * 78 + "\n")
        for a in agg:
            f.write(
                f"n/orientation={a['n']} (total={a['total']}): "
                f"spatial median of random-median E_Q={100*a['E_median']:.3f}% "
                f"IQR=[{100*a['E_q1']:.3f}, {100*a['E_q3']:.3f}]%; "
                f"median |A_sparse-A_ref|={a['Aerr_median']:.5f}; "
                f"median within-subvolume E_Q-IQR width={100*a['IQRwidth_median']:.3f} pp; "
                f"median phi-family-RMS={100*a['phi_rms_median']:.3f} pp; "
                f"E_Q<=10%: {a['pct_E_le_10']:.1f}%, "
                f"<=15%: {a['pct_E_le_15']:.1f}%, "
                f"<=20%: {a['pct_E_le_20']:.1f}%\n"
            )

        f.write("\nInterpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "The random-median reconstruction within each subvolume reduces "
            "dependence on one arbitrary offset configuration.  The distribution "
            "of those subvolume medians across the 27 non-overlapping regions "
            "then measures spatial reproducibility within this real-rock ROI.\n"
        )
        f.write(
            "No specimen-level population inference should be made from these "
            "27 regions because they all belong to the same plug/ROI.\n"
        )


# =============================================================================
# FIGURE
# =============================================================================

def panel_label(ax, label, x=0.01, y=0.99):
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(
        x, y, label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )


def draw_roi_cube(ax):
    n = RAW_SHAPE[0]
    edges = [
        ((0,0,0),(n,0,0)), ((0,n,0),(n,n,0)),
        ((0,0,n),(n,0,n)), ((0,n,n),(n,n,n)),
        ((0,0,0),(0,n,0)), ((n,0,0),(n,n,0)),
        ((0,0,n),(0,n,n)), ((n,0,n),(n,n,n)),
        ((0,0,0),(0,0,n)), ((n,0,0),(n,0,n)),
        ((0,n,0),(0,n,n)), ((n,n,0),(n,n,n)),
    ]
    for p0, p1 in edges:
        ax.plot(*zip(p0, p1), color="black", linewidth=0.7, alpha=0.5)


def paired_boxplot(ax, data_by_n, ylabel, title, scale=1.0):
    nvals = list(SECTIONS_PER_ORIENTATION)
    data = [np.asarray(data_by_n[n], dtype=float) * scale for n in nvals]

    ax.boxplot(
        data,
        tick_labels=[str(3*n) for n in nvals],
        widths=0.55,
        showfliers=False,
        medianprops=dict(linewidth=1.7, color="black"),
        boxprops=dict(linewidth=1.0, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )

    # Same spatial subvolume connected across section counts.
    n_sub = len(data[0])
    x = np.arange(1, len(nvals)+1)
    for i in range(n_sub):
        y = [data[j][i] for j in range(len(nvals))]
        ax.plot(x, y, linewidth=0.55, alpha=0.22, color="0.35")

    ax.set_xlabel("Total number of sections")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def build_figure(results, agg):
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # --------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    centers = np.array([r["meta"]["center"] for r in results], dtype=float)
    phi = 100.0 * np.array([r["phi_sub"] for r in results], dtype=float)
    sc = ax.scatter(
        centers[:,0], centers[:,1], centers[:,2],
        c=phi, s=45, cmap="viridis", depthshade=False,
    )
    draw_roi_cube(ax)
    ax.set_xlim(0, RAW_SHAPE[0]); ax.set_ylim(0, RAW_SHAPE[1]); ax.set_zlim(0, RAW_SHAPE[2])
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=22, azim=-52)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("a0", labelpad=-6); ax.set_ylabel("a1", labelpad=-6); ax.set_zlabel("a2", labelpad=-4)
    ax.set_title("Spatial replication across the ROI")
    ax.grid(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Subvolume porosity (%)", fontsize=8.5, fontweight="bold")
    cb.ax.tick_params(labelsize=7.5)
    panel_label(ax, "(a)")

    # --------------------------------------------------------------- (b)
    ax = fig.add_subplot(gs[0, 1])
    Aref = np.array([r["A_ref"] for r in results])
    ax.scatter(phi, Aref, s=30, alpha=0.82)
    ax.axvline(np.median(phi), linestyle=":", color="black", linewidth=0.9)
    ax.axhline(np.median(Aref), linestyle=":", color="black", linewidth=0.9)
    ax.set_xlabel("Subvolume porosity (%)")
    ax.set_ylabel(r"Reference anisotropy $A_{ref}$")
    ax.set_title("Reference-tensor spatial heterogeneity")
    panel_label(ax, "(b)")

    # Prepare per-n spatial replicate arrays in fixed subvolume order.
    E_by_n = {}
    Aerr_by_n = {}
    IQR_by_n = {}
    for n in SECTIONS_PER_ORIENTATION:
        rows = summaries_for_n(results, n)
        E_by_n[n] = [s["E_med"] for _, s in rows]
        Aerr_by_n[n] = [abs(s["A_med"] - r["A_ref"]) for r, s in rows]
        IQR_by_n[n] = [s["E_q3"] - s["E_q1"] for _, s in rows]

    # --------------------------------------------------------------- (c)
    ax = fig.add_subplot(gs[0, 2])
    paired_boxplot(
        ax, E_by_n,
        ylabel=r"Tensor-shape error $E_Q$ (%)",
        title="Spatial reproducibility of sparse recovery",
        scale=100.0,
    )
    panel_label(ax, "(c)")

    # --------------------------------------------------------------- (d)
    ax = fig.add_subplot(gs[1, 0])
    paired_boxplot(
        ax, Aerr_by_n,
        ylabel=r"$|A_{sparse}-A_{ref}|$",
        title="Anisotropy-magnitude error",
        scale=1.0,
    )
    panel_label(ax, "(d)")

    # --------------------------------------------------------------- (e)
    ax = fig.add_subplot(gs[1, 1])
    paired_boxplot(
        ax, IQR_by_n,
        ylabel=r"Within-subvolume $E_Q$ IQR width (pp)",
        title="Section-position uncertainty",
        scale=100.0,
    )
    panel_label(ax, "(e)")

    # --------------------------------------------------------------- (f)
    ax = fig.add_subplot(gs[1, 2])
    totals = [a["total"] for a in agg]
    p10 = [a["pct_E_le_10"] for a in agg]
    p15 = [a["pct_E_le_15"] for a in agg]
    p20 = [a["pct_E_le_20"] for a in agg]
    ax.plot(totals, p10, marker="o", label=r"$E_Q\leq10\%$")
    ax.plot(totals, p15, marker="s", label=r"$E_Q\leq15\%$")
    ax.plot(totals, p20, marker="^", label=r"$E_Q\leq20\%$")
    ax.set_xticks(totals)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel("Spatial subvolumes meeting criterion (%)")
    ax.set_title("Reliability across the real-rock ROI")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "(f)")

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.96)
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
    grid = spatial_grid()

    print("=" * 78)
    print("SPATIAL REPLICATION DESIGN")
    print("=" * 78)
    print(f"Subvolume shape:      {SUBVOL_N}^3")
    print(f"Centers / axis:       {GRID_CENTERS}")
    print(f"Spatial subvolumes:   {len(grid)} (3 x 3 x 3)")
    print("Subvolumes overlap:   NO")
    print(f"Sections / orient.:   {SECTIONS_PER_ORIENTATION}")
    print(f"Random configs / n:   {50 if QUICK_MODE else N_RANDOM_CONFIGS}")
    print(f"Reference points/dir: {4_000 if QUICK_MODE else N_REF_POINTS:,}")
    print()

    CACHE_DIR.mkdir(exist_ok=True)
    if FORCE_RECOMPUTE:
        for p in CACHE_DIR.glob("G*.json"):
            p.unlink()

    raw = np.memmap(
        RAW_FILE,
        dtype=RAW_DTYPE,
        mode="r",
        shape=RAW_SHAPE,
        order="C",
    )

    ref_dirs = fibonacci_hemisphere(N_REF_DIRECTIONS)
    n_ref_points = 4_000 if QUICK_MODE else N_REF_POINTS
    n_random = 50 if QUICK_MODE else N_RANDOM_CONFIGS

    global_t0 = time.time()

    for idx, meta in enumerate(grid, start=1):
        cache_path = CACHE_DIR / f"{meta['id']}.json"
        print(f"[{idx:02d}/{len(grid)}] {meta['id']} center={meta['center']}")

        if cache_path.exists() and not FORCE_RECOMPUTE:
            print(f"  cache found -> skipping computation ({cache_path.name})\n")
            continue

        result = process_subvolume(
            raw,
            meta,
            ref_dirs,
            n_ref_points=n_ref_points,
            n_random=n_random,
        )
        json_atomic_write(cache_path, result)
        print(f"Cached: {cache_path}\n")

    del raw
    gc.collect()

    results = load_all_cached(grid)
    write_csv_outputs(results)
    agg = aggregate_table(results)
    write_report(results, agg)
    build_figure(results, agg)

    print("=" * 78)
    print("SPATIAL-REPLICATION SUMMARY")
    print("=" * 78)
    phi = np.array([r["phi_sub"] for r in results])
    Aref = np.array([r["A_ref"] for r in results])
    eta = np.array([r["eta_ref"] for r in results])
    reliable = np.array([r["orientation_reliable"] for r in results], dtype=bool)

    print(
        f"Subvolume porosity: median={100*np.median(phi):.3f}%  "
        f"range=[{100*np.min(phi):.3f}, {100*np.max(phi):.3f}]%"
    )
    print(
        f"A_ref:              median={np.median(Aref):.4f}  "
        f"range=[{np.min(Aref):.4f}, {np.max(Aref):.4f}]"
    )
    print(
        f"Reference residual: median={100*np.median(eta):.2f}%"
    )
    print(
        f"Reliable major axis: {np.sum(reliable)}/{len(reliable)} subvolumes"
    )
    print()

    for a in agg:
        print(
            f"n/orientation={a['n']} total={a['total']:2d}  "
            f"spatial median E_Q={100*a['E_median']:6.2f}%  "
            f"IQR=[{100*a['E_q1']:.2f},{100*a['E_q3']:.2f}]%  "
            f"median |dA|={a['Aerr_median']:.4f}  "
            f"median position-IQR={100*a['IQRwidth_median']:.2f} pp  "
            f"E<=10/15/20%: {a['pct_E_le_10']:.0f}/{a['pct_E_le_15']:.0f}/{a['pct_E_le_20']:.0f}%"
        )

    print()
    print("=" * 78)
    print("FILES GENERATED")
    print("=" * 78)
    print(PNG_PATH)
    print(SVG_PATH)
    print(REPORT_PATH)
    print(SUBVOL_CSV_PATH)
    print(SUMMARY_CSV_PATH)
    print(RANDOM_CSV_PATH)
    print(CACHE_DIR)
    print()
    print(f"Total elapsed time: {(time.time()-global_t0)/60:.2f} min")
    print("Bentheimer test 3 completed successfully.")


if __name__ == "__main__":
    main()