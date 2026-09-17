#!/usr/bin/env python3
"""
Final Edwards Brown (EdB-1) real-rock validation.

This is the production script after calibration.  It freezes the reference
protocol established by the preceding convergence experiments:

    67 ambient directions
    50,000 Monte Carlo points per direction
    5 independent seeds
    Q3D_REF = SPD-projected Euclidean mean of the five seed tensors

For each of 27 non-overlapping 512^3 subvolumes in a 3 x 3 x 3 grid, the
script compares that full-3D reference tensor with sparse reconstructions from
axis-aligned 2D sections.  It evaluates 1, 3, 5, and 7 sections per orientation
(3, 9, 15, and 21 total sections), using 200 random offset configurations at
each section count.

IMPORTANT SCIENTIFIC TERMINOLOGY
--------------------------------
The full-3D tensor is a numerical reference estimate, not an analytic ground
truth.  The sparse tensor is reconstructed only from 2D sections.

The calibrated uncertainty of the frozen 5-seed x 50k reference protocol is
carried as a separate benchmark uncertainty; it is NOT subtracted from sparse
reconstruction errors.

EdB-1 RAW convention verified before production:
    0 = pore
    1 = solid

The published EdB-1 ROI-1 computed porosity is about 0.20, matching the
observed zero fraction in the downloaded binary ROI-1.

Computational axes are called a0/a1/a2 until the physical acquisition-axis
mapping is confirmed from metadata.

Outputs
-------
  edb1_final_validation.png
  edb1_final_validation.svg
  edb1_final_validation_report.txt
  edb1_final_subvolumes.csv
  edb1_final_summary.csv
  edb1_final_random.csv
  edb1_final_systematic.csv
  edb1_final_reference_seeds.csv
  edb1_final_reference_cache/
  edb1_final_section_cache/
  edb1_final_result_cache/

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

Run
---
  python3 real_edb1_final_validation.py

Restartability
--------------
Reference seeds, 2D section measurements, and completed subvolume results are
cached.  If the run is interrupted, run the same command again.

EdB-1 has its own reference/section/result caches. No Bentheimer calibration
cache is reused; the frozen numerical protocol is transferred unchanged.
"""

from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


# =============================================================================
# FROZEN PROTOCOL / USER SETTINGS
# =============================================================================

RAW_FILE = Path(
    "/home/tromer/Isarael/pushforward/data/real_rocks/EdwardsBrown_EdB1/ROI1/"
    "edb-1_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)
RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8

# Phase convention independently supported by the published dataset table:
# EdB-1 ROI-1 computed porosity is approximately 0.20, while the downloaded
# binary RAW has approximately 20% zeros.  The segmentation description also
# assigns pixels >= threshold to 1, consistent with 0=void/pore, 1=solid here.
PORE_VALUE = 0
EXPECTED_ROI1_COMPUTED_POROSITY = 0.20
PHASE_SANITY_TOL = 0.05

SUBVOL_N = 512
GRID_CENTERS = (384, 1250, 2116)

# Full-3D reference protocol -- FROZEN after Test 5.
MAX_LAG = 48
N_REF_DIRECTIONS = 64        # Fibonacci directions; 3 axes are added -> 67.
N_REF_POINTS = 50_000
N_REF_SEEDS = 5

# 2D section protocol.
THETA_STEP_2D_DEG = 5.0
SECTIONS_PER_ORIENTATION = (1, 3, 5, 7)
OFFSET_MAX = 192
OFFSET_STEP = 32
N_RANDOM_CONFIGS = 200

# Spectral gap used only to flag whether a principal-axis angle is interpretable.
ORIENTATION_GAP_MIN = 0.10

# Reproducibility seed for new final-reference jobs and random section choices.
SEED = 20260908

# Calibrated uncertainty of the FINAL 5-seed x 50k protocol from Test 5.
# These are empirical bootstrap-pair uncertainty proxies and are reported as
# benchmark uncertainty, not subtracted from E_Q.
CALIBRATED_REF_U_MEDIAN = 0.00932
CALIBRATED_REF_U_P95 = 0.01706
CALIBRATED_REF_U_WORST_P95 = 0.01881

# Publication output.
DPI = 800

# Normal production mode.
QUICK_MODE = False

# QUICK_MODE is only for checking the pipeline; do not use its numbers in paper.
QUICK_REF_POINTS = 8_000
QUICK_REF_SEEDS = 2
QUICK_RANDOM_CONFIGS = 40

# Restart / reuse controls.
FORCE_RECOMPUTE_REFERENCES = False
FORCE_RECOMPUTE_SECTIONS = False
FORCE_RECOMPUTE_RESULTS = False
REUSE_TEST4_CACHE = False


# =============================================================================
# OUTPUTS / STYLE
# =============================================================================

OUTDIR = RAW_FILE.parent / "final_validation"
PNG_PATH = OUTDIR / "edb1_final_validation.png"
SVG_PATH = OUTDIR / "edb1_final_validation.svg"
REPORT_PATH = OUTDIR / "edb1_final_validation_report.txt"
SUBVOL_CSV_PATH = OUTDIR / "edb1_final_subvolumes.csv"
SUMMARY_CSV_PATH = OUTDIR / "edb1_final_summary.csv"
RANDOM_CSV_PATH = OUTDIR / "edb1_final_random.csv"
SYSTEMATIC_CSV_PATH = OUTDIR / "edb1_final_systematic.csv"
REFSEED_CSV_PATH = OUTDIR / "edb1_final_reference_seeds.csv"

REF_CACHE_DIR = OUTDIR / "edb1_final_reference_cache"
SECTION_CACHE_DIR = OUTDIR / "edb1_final_section_cache"
RESULT_CACHE_DIR = OUTDIR / "edb1_final_result_cache"
TEST4_CACHE_DIR = OUTDIR / "unused_test4_cache"

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
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# ACTIVE SETTINGS
# =============================================================================

def active_settings():
    if QUICK_MODE:
        return {
            "ref_points": int(QUICK_REF_POINTS),
            "ref_seeds": int(QUICK_REF_SEEDS),
            "random_configs": int(QUICK_RANDOM_CONFIGS),
            "is_final": False,
        }
    return {
        "ref_points": int(N_REF_POINTS),
        "ref_seeds": int(N_REF_SEEDS),
        "random_configs": int(N_RANDOM_CONFIGS),
        "is_final": True,
    }


# =============================================================================
# BASIC TENSOR UTILITIES
# =============================================================================

def project_spd(Q):
    """Symmetrize and project a matrix to SPD by eigenvalue flooring."""
    Q = 0.5 * (np.asarray(Q, dtype=float) + np.asarray(Q, dtype=float).T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-8 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T


def normalize_det_spd(Q):
    """Project to SPD and normalize det(Q)=1."""
    Q = project_spd(Q)
    return Q / np.linalg.det(Q) ** (1.0 / 3.0)


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
    val = np.maximum(val, 1e-15)
    return 1.0 / np.sqrt(val)


def fit_q3d_from_directional_lengths(vectors, ell):
    """Unweighted LS in 1/ell^2, followed by SPD projection."""
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
        raise RuntimeError(f"3D design matrix is rank deficient: rank={rank}")

    q, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = project_spd(qvec_to_matrix(q))

    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1])

    pred = ell_from_q3d(Q, V)
    eta = float(np.sqrt(np.sum((L - pred)**2) / np.sum(L**2)))

    return Q, rank, cond, eta, V, L


def principal_lengths_and_axes(Q):
    w, V = np.linalg.eigh(project_spd(Q))
    w = np.maximum(w, 1e-14)
    ell = 1.0 / np.sqrt(w)
    order = np.argsort(ell)[::-1]
    return ell[order], V[:, order]


def anisotropy_ratio(Q):
    ell, _ = principal_lengths_and_axes(Q)
    return float(ell[0] / ell[-1])


def geometric_correlation_length(Q):
    """Geometric mean of principal lengths = det(Q)^(-1/6)."""
    Q = project_spd(Q)
    return float(np.linalg.det(Q) ** (-1.0 / 6.0))


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


def tensor_pair_error(Q1, Q2):
    """Symmetric determinant-normalized pair disagreement."""
    A = normalize_det_spd(Q1)
    B = normalize_det_spd(Q2)
    den = 0.5 * (np.linalg.norm(A, ord="fro") + np.linalg.norm(B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / den)


def reference_major_axis_gap(Q_ref):
    ell, _ = principal_lengths_and_axes(normalize_det_spd(Q_ref))
    return float((ell[0] - ell[1]) / ell[0])


def consensus_Q(Q_list):
    """Frozen protocol: Euclidean mean in Q-space, projected back to SPD."""
    return project_spd(np.mean(np.stack(Q_list, axis=0), axis=0))


def q_components(Q):
    Q = np.asarray(Q, dtype=float)
    return {
        "q00": float(Q[0, 0]),
        "q11": float(Q[1, 1]),
        "q22": float(Q[2, 2]),
        "q01": float(Q[0, 1]),
        "q02": float(Q[0, 2]),
        "q12": float(Q[1, 2]),
    }


def q_from_components(row):
    return np.array([
        [row["q00"], row["q01"], row["q02"]],
        [row["q01"], row["q11"], row["q12"]],
        [row["q02"], row["q12"], row["q22"]],
    ], dtype=float)


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
    print("Status:   PERFECT MATCH")

    # Lightweight phase-identity sanity check from distributed axis-2 planes.
    # This is not used as an estimator in the validation; it only prevents an
    # accidental pore/solid inversion before expensive processing.
    raw = np.memmap(path, dtype=RAW_DTYPE, mode="r", shape=RAW_SHAPE, order="C")
    idx = np.linspace(125, RAW_SHAPE[2] - 126, 9).round().astype(int)
    counts = {0: 0, 1: 0}
    total = 0
    for k in idx:
        plane = np.asarray(raw[:, :, int(k)])
        counts[0] += int(np.count_nonzero(plane == 0))
        counts[1] += int(np.count_nonzero(plane == 1))
        total += int(plane.size)
    del raw
    f0 = counts[0] / total
    f1 = counts[1] / total
    d0 = abs(f0 - EXPECTED_ROI1_COMPUTED_POROSITY)
    d1 = abs(f1 - EXPECTED_ROI1_COMPUTED_POROSITY)
    print(f"Phase sanity sample: value 0={100*f0:.3f}%  value 1={100*f1:.3f}%")
    print(f"Published EdB-1 ROI-1 computed porosity: ~{100*EXPECTED_ROI1_COMPUTED_POROSITY:.1f}%")
    if PORE_VALUE != 0 or d0 > PHASE_SANITY_TOL or d0 >= d1:
        raise RuntimeError(
            "Phase sanity check failed. Do not run final validation until "
            "the EdB-1 binary convention is re-verified."
        )
    print("Phase convention: 0=pore, 1=solid  [VERIFIED]\n")


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

                if any(s < 0 for s in start) or any(
                    e > RAW_SHAPE[d] for d, e in enumerate(stop)
                ):
                    raise ValueError(f"Subvolume {center} lies outside RAW volume.")

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
# FULL-3D REFERENCE ESTIMATOR
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
    """First linearly interpolated 1/e crossing."""
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


def directional_lengths_3d(phase, directions, max_lag, n_points, seed):
    """
    Monte Carlo directional 3D correlation lengths without a 3D FFT.

    Off-lattice displaced points are sampled with trilinear interpolation.
    The same base points are reused across directions for a given seed.
    """
    rng = np.random.default_rng(seed)
    n = phase.shape[0]
    margin = int(np.ceil(max_lag)) + 2
    if 2 * margin >= n:
        raise ValueError("MAX_LAG is too large for the subvolume.")

    phi_pore = float(np.mean(phase == PORE_VALUE))
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
    f0 = chi0 - np.float32(phi_pore)
    lags = np.arange(max_lag + 1, dtype=np.float32)
    all_ell = np.full(len(directions), np.nan, dtype=np.float64)

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
        fr = chi_r - np.float32(phi_pore)
        C = np.mean(f0[:, None] * fr, axis=0, dtype=np.float64) / variance
        C[0] = 1.0
        all_ell[j] = correlation_length(lags, C)

        if ((j + 1) % max(1, len(directions) // 10) == 0
                or j == len(directions) - 1):
            elapsed = time.time() - t0
            print(
                f"      {j+1:2d}/{len(directions)} directions "
                f"({100*(j+1)/len(directions):5.1f}%)  elapsed={elapsed:6.1f}s"
            )

        del coords, sampled_solid, chi_r, fr, C

    return all_ell, phi_pore, time.time() - t0


# =============================================================================
# REFERENCE CACHE / TEST-4 REUSE
# =============================================================================

def save_json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def final_ref_signature(ref_points, ref_seeds):
    return {
        "raw_file": RAW_FILE.name,
        "pore_value": int(PORE_VALUE),
        "raw_shape": list(RAW_SHAPE),
        "raw_dtype": str(np.dtype(RAW_DTYPE)),
        "subvol_n": int(SUBVOL_N),
        "max_lag": int(MAX_LAG),
        "n_ref_directions": int(N_REF_DIRECTIONS),
        "ref_points": int(ref_points),
        "ref_seeds": int(ref_seeds),
        "estimator_version": 1,
        "consensus_version": 1,
    }


def ref_cache_path(subvol_id, seed_index):
    return REF_CACHE_DIR / f"{subvol_id}_seed{seed_index:02d}.json"


def load_final_ref_cache(path, signature, meta):
    if FORCE_RECOMPUTE_REFERENCES or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("signature") != signature:
            return None
        if d.get("subvolume", {}).get("id") != meta["id"]:
            return None
        if tuple(d.get("subvolume", {}).get("center", [])) != tuple(meta["center"]):
            return None
        Q = np.asarray(d["Q"], dtype=float)
        if Q.shape != (3, 3) or not np.all(np.isfinite(Q)):
            return None
        return d
    except Exception:
        return None


def try_import_test4_seed(meta, seed_index, signature, ref_points):
    """Reuse a compatible 50k Test-4 seed if one exists."""
    if not REUSE_TEST4_CACHE or QUICK_MODE or FORCE_RECOMPUTE_REFERENCES:
        return None
    if ref_points != 50_000:
        return None

    path = TEST4_CACHE_DIR / f"{meta['id']}_seed{seed_index:02d}.json"
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        sub = d.get("subvolume", {})
        sig = d.get("signature", {})
        pr = d.get("point_results", {}).get("50000")
        if pr is None:
            return None

        if sub.get("id") != meta["id"]:
            return None
        if tuple(sub.get("center", [])) != tuple(meta["center"]):
            return None
        if int(sig.get("subvol_n", -1)) != SUBVOL_N:
            return None
        if int(sig.get("max_lag", -1)) != MAX_LAG:
            return None
        if int(sig.get("n_ref_directions", -1)) != N_REF_DIRECTIONS:
            return None

        Q = np.asarray(pr.get("Q"), dtype=float)
        if Q.shape != (3, 3) or not np.all(np.isfinite(Q)):
            return None

        imported = {
            "signature": signature,
            "subvolume": {
                "id": meta["id"],
                "center": list(meta["center"]),
                "start": list(meta["start"]),
                "stop": list(meta["stop"]),
            },
            "seed_index": int(seed_index),
            "random_seed": int(d.get("random_seed", -1)),
            "phi_sub": float(d["phi_sub"]),
            "elapsed_sec": float(d.get("elapsed_sec", 0.0)),
            "source": "imported_test4_cache",
            "Q": [[float(v) for v in row] for row in Q],
            "rank": int(pr.get("rank", 6)),
            "condition": float(pr.get("condition", np.nan)),
            "eta": float(pr.get("eta", np.nan)),
            "A": float(pr.get("A", anisotropy_ratio(Q))),
            "ell_geom": float(pr.get("ell_geom", geometric_correlation_length(Q))),
            "valid_directions": int(pr.get("valid_directions", 67)),
            "total_directions": int(pr.get("total_directions", 67)),
        }
        return imported
    except Exception:
        return None


def final_random_seed(meta, seed_index):
    gid = int(meta["id"][1:])
    return int(SEED + 1_000_000 + 10_000 * gid + seed_index)


def compute_final_reference_seed(phase, meta, seed_index, signature, ref_points):
    random_seed = final_random_seed(meta, seed_index)
    directions = fibonacci_hemisphere(N_REF_DIRECTIONS)

    print(
        f"    reference seed {seed_index+1:02d}/{signature['ref_seeds']:02d}  "
        f"rng_seed={random_seed}  points={ref_points:,}"
    )

    ell, phi_check, elapsed = directional_lengths_3d(
        phase,
        directions,
        max_lag=MAX_LAG,
        n_points=ref_points,
        seed=random_seed,
    )

    phi_sub = float(np.mean(phase == PORE_VALUE))
    if not np.isclose(phi_sub, phi_check):
        raise RuntimeError("Internal porosity consistency check failed.")

    valid = np.isfinite(ell) & (ell > 0)
    Q, rank, cond, eta, _, _ = fit_q3d_from_directional_lengths(
        directions, ell
    )

    return {
        "signature": signature,
        "subvolume": {
            "id": meta["id"],
            "center": list(meta["center"]),
            "start": list(meta["start"]),
            "stop": list(meta["stop"]),
        },
        "seed_index": int(seed_index),
        "random_seed": int(random_seed),
        "phi_sub": phi_sub,
        "elapsed_sec": float(elapsed),
        "source": "computed_final",
        "Q": [[float(v) for v in row] for row in Q],
        "rank": int(rank),
        "condition": float(cond),
        "eta": float(eta),
        "A": float(anisotropy_ratio(Q)),
        "ell_geom": float(geometric_correlation_length(Q)),
        "valid_directions": int(np.sum(valid)),
        "total_directions": int(len(ell)),
    }


def reference_consensus(seed_results):
    Qs = [np.asarray(d["Q"], dtype=float) for d in seed_results]
    Q_ref = consensus_Q(Qs)

    pair = []
    for i in range(len(Qs)):
        for j in range(i + 1, len(Qs)):
            pair.append(tensor_pair_error(Qs[i], Qs[j]))

    etas = np.array([d["eta"] for d in seed_results], dtype=float)
    As = np.array([d["A"] for d in seed_results], dtype=float)
    ellg = np.array([d["ell_geom"] for d in seed_results], dtype=float)

    return {
        "Q_ref": Q_ref,
        "A_ref": anisotropy_ratio(Q_ref),
        "ell_geom_ref": geometric_correlation_length(Q_ref),
        "major_gap": reference_major_axis_gap(Q_ref),
        "orientation_reliable": bool(reference_major_axis_gap(Q_ref) >= ORIENTATION_GAP_MIN),
        "seed_eta_median": float(np.median(etas)),
        "seed_eta_q1": float(np.percentile(etas, 25)),
        "seed_eta_q3": float(np.percentile(etas, 75)),
        "seed_A_sd": float(np.std(As, ddof=1)) if len(As) > 1 else 0.0,
        "seed_ell_geom_cv": float(np.std(ellg, ddof=1) / np.mean(ellg)) if len(ellg) > 1 else 0.0,
        "seed_pair_median": float(np.median(pair)) if pair else 0.0,
        "seed_pair_max": float(np.max(pair)) if pair else 0.0,
    }


# =============================================================================
# 2D SECTION ESTIMATOR / CACHE
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


def measure_section_2d(phase_section, B, family, offset, absolute_index):
    pore = (phase_section == PORE_VALUE)
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
        "phi": float(phi),
        "theta": theta,
        "vectors": v,
        "ell": ell,
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


def section_signature():
    return {
        "subvol_n": int(SUBVOL_N),
        "max_lag": int(MAX_LAG),
        "theta_step_deg": float(THETA_STEP_2D_DEG),
        "offset_max": int(OFFSET_MAX),
        "offset_step": int(OFFSET_STEP),
        "estimator_version": 1,
    }


def section_cache_path(subvol_id):
    return SECTION_CACHE_DIR / f"{subvol_id}.json"


def serialize_section(s):
    return {
        "family": s["family"],
        "offset": int(s["offset"]),
        "absolute_index": int(s["absolute_index"]),
        "phi": float(s["phi"]),
        "theta": [float(v) for v in s["theta"]],
        "vectors": [[float(v) for v in row] for row in s["vectors"]],
        "ell": [float(v) if np.isfinite(v) else None for v in s["ell"]],
    }


def deserialize_section(d):
    ell = np.array([
        np.nan if v is None else float(v) for v in d["ell"]
    ], dtype=float)
    return {
        "family": d["family"],
        "offset": int(d["offset"]),
        "absolute_index": int(d["absolute_index"]),
        "phi": float(d["phi"]),
        "theta": np.asarray(d["theta"], dtype=float),
        "vectors": np.asarray(d["vectors"], dtype=float),
        "ell": ell,
    }


def load_section_cache(path, meta):
    if FORCE_RECOMPUTE_SECTIONS or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("signature") != section_signature():
            return None
        if d.get("subvolume_id") != meta["id"]:
            return None
        if tuple(d.get("center", [])) != tuple(meta["center"]):
            return None

        cache = {}
        for item in d["sections"]:
            s = deserialize_section(item)
            cache[(s["family"], s["offset"])] = s
        expected = 3 * len(make_offset_pool())
        if len(cache) != expected:
            return None
        return cache
    except Exception:
        return None


def precompute_section_pool(phase, meta):
    definitions = plane_family_definitions()
    offsets = make_offset_pool()
    c = phase.shape[0] // 2

    if c - OFFSET_MAX < 0 or c + OFFSET_MAX >= phase.shape[0]:
        raise ValueError("OFFSET_MAX places a section outside the subvolume.")

    cache = {}
    count = 0
    total = 3 * len(offsets)
    t0 = time.time()

    for family, fmeta in definitions.items():
        for offset in offsets:
            idx = c + int(offset)
            img = extract_axis_aligned_section(
                phase, fmeta["fixed_axis"], idx
            )
            s = measure_section_2d(
                img,
                fmeta["B"],
                family=family,
                offset=offset,
                absolute_index=idx,
            )
            cache[(family, int(offset))] = s
            count += 1
            if count % 13 == 0 or count == total:
                print(
                    f"    sections {count:2d}/{total:2d}  "
                    f"elapsed={time.time()-t0:5.1f}s"
                )

    payload = {
        "signature": section_signature(),
        "subvolume_id": meta["id"],
        "center": list(meta["center"]),
        "sections": [
            serialize_section(cache[key])
            for key in sorted(cache.keys(), key=lambda x: (x[0], x[1]))
        ],
    }
    save_json_atomic(section_cache_path(meta["id"]), payload)
    return cache


# =============================================================================
# SPARSE CONFIGURATION RECONSTRUCTION
# =============================================================================

def collect_sections(cache, offsets_by_family):
    sections = []
    for family, offsets in offsets_by_family.items():
        for offset in offsets:
            sections.append(cache[(family, int(offset))])
    return sections


def sparse_design_diagnostics(section_results):
    """Diagnostics for the actually observed directional lengths in a configuration.

    A geometrically valid three-plane design can lose rank if a selected section has
    too few finite 1/e crossings within MAX_LAG.  This helper distinguishes that
    measurement-window issue from the nominal plane geometry.
    """
    vectors = []
    ell = []
    valid_by_family = {}

    for s in section_results:
        good = np.isfinite(s["ell"]) & (s["ell"] > 0)
        family = s["family"]
        valid_by_family[family] = valid_by_family.get(family, 0) + int(np.sum(good))
        if np.any(good):
            vectors.append(s["vectors"][good])
            ell.append(s["ell"][good])

    if not vectors:
        return {
            "rank": 0,
            "condition": np.nan,
            "valid_directions": 0,
            "valid_by_family": valid_by_family,
        }

    V = np.vstack(vectors)
    L = np.concatenate(ell)
    A = design_rows_from_vectors(V)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    condition = np.nan
    if rank == 6:
        svals = np.linalg.svd(A, compute_uv=False)
        if svals[-1] > 0:
            condition = float(svals[0] / svals[-1])

    return {
        "rank": rank,
        "condition": condition,
        "valid_directions": int(len(L)),
        "valid_by_family": valid_by_family,
    }


def reconstruct_sparse_q3d(section_results):
    vectors = []
    ell = []
    for s in section_results:
        good = np.isfinite(s["ell"]) & (s["ell"] > 0)
        if np.any(good):
            vectors.append(s["vectors"][good])
            ell.append(s["ell"][good])

    if not vectors:
        raise RuntimeError("No valid directional lengths in sparse configuration.")

    vectors = np.vstack(vectors)
    ell = np.concatenate(ell)
    return fit_q3d_from_directional_lengths(vectors, ell)


def configuration_metrics(sections, Q_ref, phi_sub, orientation_reliable):
    """Compute sparse metrics without aborting on a rank-deficient random draw.

    Rank-deficient configurations are scientifically meaningful failures of that
    particular section-position draw.  They are retained with valid=False and are
    excluded from conditional E_Q percentile summaries; their frequency is reported
    separately instead of silently resampling them.
    """
    phi_all = float(np.mean([s["phi"] for s in sections]))
    family_names = sorted(set(s["family"] for s in sections))
    family_means = {}
    for family in family_names:
        vals = [s["phi"] for s in sections if s["family"] == family]
        family_means[family] = float(np.mean(vals))

    phi_family_rms = float(np.sqrt(np.mean([
        (family_means[f] - phi_sub)**2 for f in family_names
    ])))

    diag = sparse_design_diagnostics(sections)

    try:
        Q, rank, cond, eta, _, _ = reconstruct_sparse_q3d(sections)
        E_Q = tensor_shape_error(Q, Q_ref)
        A_sparse = anisotropy_ratio(Q)

        _, axes_ref = principal_lengths_and_axes(normalize_det_spd(Q_ref))
        _, axes_sparse = principal_lengths_and_axes(normalize_det_spd(Q))
        major_axis_error = axial_angle_error_deg(axes_ref[:, 0], axes_sparse[:, 0])

        out = {
            "valid": True,
            "failure_reason": "",
            "E_Q": float(E_Q),
            "A_sparse": float(A_sparse),
            "eta": float(eta),
            "rank": int(rank),
            "condition": float(cond),
            "valid_directions": int(diag["valid_directions"]),
            "valid_by_family": diag["valid_by_family"],
            "phi_all": float(phi_all),
            "phi_family_rms": float(phi_family_rms),
            "major_axis_error_deg": float(major_axis_error),
            "orientation_reliable": bool(orientation_reliable),
            "family_means": family_means,
        }
        out.update(q_components(Q))
        return out

    except RuntimeError as exc:
        # Do not invent a tensor with a pseudoinverse when the observed design has
        # rank < 6.  Preserve the attempted configuration as a non-identifiable draw.
        out = {
            "valid": False,
            "failure_reason": str(exc),
            "E_Q": np.nan,
            "A_sparse": np.nan,
            "eta": np.nan,
            "rank": int(diag["rank"]),
            "condition": float(diag["condition"]),
            "valid_directions": int(diag["valid_directions"]),
            "valid_by_family": diag["valid_by_family"],
            "phi_all": float(phi_all),
            "phi_family_rms": float(phi_family_rms),
            "major_axis_error_deg": np.nan,
            "orientation_reliable": bool(orientation_reliable),
            "family_means": family_means,
            "q00": np.nan, "q11": np.nan, "q22": np.nan,
            "q01": np.nan, "q02": np.nan, "q12": np.nan,
        }
        return out


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
        if not metrics["valid"]:
            print(
                f"      NOTE systematic n/orient={n}: non-identifiable "
                f"(rank={metrics['rank']}; {metrics['failure_reason']})"
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
        invalid_count = 0
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
            if not metrics["valid"]:
                invalid_count += 1
                if invalid_count <= 3:
                    famtxt = ", ".join(
                        f"{k}:{v}" for k, v in sorted(metrics["valid_by_family"].items())
                    )
                    print(
                        f"      rank-deficient random draw: n/orient={n}, "
                        f"trial={trial+1}, rank={metrics['rank']}, "
                        f"finite directions by family=[{famtxt}]"
                    )

            rows.append({
                "trial": int(trial + 1),
                "n_per_orientation": int(n),
                "total_sections": int(3*n),
                "offsets_a0_a1": [int(v) for v in offsets_by_family["a0-a1"]],
                "offsets_a0_a2": [int(v) for v in offsets_by_family["a0-a2"]],
                "offsets_a1_a2": [int(v) for v in offsets_by_family["a1-a2"]],
                **metrics,
            })

        n_valid = n_random - invalid_count
        print(
            f"      random identifiability n/orient={n}: "
            f"{n_valid}/{n_random} rank-6 ({100*n_valid/n_random:.1f}%)"
        )
    return rows


def percentile_summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "median": np.nan, "q1": np.nan, "q3": np.nan,
            "p05": np.nan, "p90": np.nan, "p95": np.nan,
            "min": np.nan, "max": np.nan,
        }
    return {
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "p05": float(np.percentile(x, 5)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def summarize_random(random_rows):
    out = []
    for n in SECTIONS_PER_ORIENTATION:
        attempted = [r for r in random_rows if r["n_per_orientation"] == n]
        rows = [
            r for r in attempted
            if r.get("valid", True) and np.isfinite(r.get("E_Q", np.nan))
        ]

        n_attempted = len(attempted)
        n_valid = len(rows)
        valid_fraction = float(n_valid / n_attempted) if n_attempted else np.nan

        E = percentile_summary([r["E_Q"] for r in rows])
        A = percentile_summary([r["A_sparse"] for r in rows])
        eta = percentile_summary([r["eta"] for r in rows])
        phi_rms = percentile_summary([r["phi_family_rms"] for r in rows])

        out.append({
            "n_per_orientation": int(n),
            "total_sections": int(3*n),
            "n_attempted": int(n_attempted),
            "n_valid": int(n_valid),
            "n_invalid": int(n_attempted - n_valid),
            "valid_fraction": float(valid_fraction),
            "E_med": E["median"],
            "E_q1": E["q1"],
            "E_q3": E["q3"],
            "E_p05": E["p05"],
            "E_p95": E["p95"],
            "A_med": A["median"],
            "A_q1": A["q1"],
            "A_q3": A["q3"],
            "eta_med": eta["median"],
            "eta_q1": eta["q1"],
            "eta_q3": eta["q3"],
            "phi_rms_med": phi_rms["median"],
            "phi_rms_q1": phi_rms["q1"],
            "phi_rms_q3": phi_rms["q3"],
        })
    return out


# =============================================================================
# FINAL RESULT CACHE
# =============================================================================

def q_hash(Q):
    arr = np.asarray(Q, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:20]


def result_signature(Q_ref, random_configs):
    return {
        "section_signature": section_signature(),
        "q_ref_hash": q_hash(Q_ref),
        "sections_per_orientation": list(SECTIONS_PER_ORIENTATION),
        "random_configs": int(random_configs),
        "seed": int(SEED),
        "result_version": 2,
    }


def result_cache_path(subvol_id):
    return RESULT_CACHE_DIR / f"{subvol_id}.json"


def load_result_cache(path, signature):
    if FORCE_RECOMPUTE_RESULTS or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("signature") != signature:
            return None
        return d
    except Exception:
        return None


def process_sparse_results(meta, section_cache, Q_ref, phi_sub,
                           orientation_reliable, random_configs):
    signature = result_signature(Q_ref, random_configs)
    path = result_cache_path(meta["id"])
    cached = load_result_cache(path, signature)
    if cached is not None:
        print("    sparse reconstruction loaded from final result cache")
        return cached

    systematic = run_systematic_configs(
        section_cache, Q_ref, phi_sub, orientation_reliable
    )

    # Same random-section seed convention as Test 3, enabling direct design
    # continuity while replacing only the calibrated 3D reference.
    rng = np.random.default_rng(
        SEED + 50_000 + 1000 * int(meta["id"][1:])
    )
    random_rows = run_random_configs(
        section_cache,
        make_offset_pool(),
        Q_ref,
        phi_sub,
        orientation_reliable,
        rng,
        random_configs,
    )
    summary = summarize_random(random_rows)

    payload = {
        "signature": signature,
        "subvolume_id": meta["id"],
        "systematic": systematic,
        "random": random_rows,
        "summary": summary,
    }
    save_json_atomic(path, payload)
    return payload


# =============================================================================
# SPATIAL AGGREGATION
# =============================================================================

def summary_for_n(result, n):
    for row in result["sparse"]["summary"]:
        if row["n_per_orientation"] == n:
            return row
    raise KeyError(n)


def aggregate_spatial(results):
    agg = []
    for n in SECTIONS_PER_ORIENTATION:
        rows = []
        for r in results:
            s = summary_for_n(r, n)
            rows.append((r, s))

        E = np.array([s["E_med"] for r, s in rows], dtype=float)
        Aerr = np.array([abs(s["A_med"] - r["A_ref"]) for r, s in rows], dtype=float)
        IQRw = np.array([s["E_q3"] - s["E_q1"] for r, s in rows], dtype=float)
        phi_rms = np.array([s["phi_rms_med"] for r, s in rows], dtype=float)
        eta = np.array([s["eta_med"] for r, s in rows], dtype=float)
        valid_fraction = np.array([s.get("valid_fraction", 1.0) for r, s in rows], dtype=float)

        e = percentile_summary(E)
        a = percentile_summary(Aerr)
        iw = percentile_summary(IQRw)
        pr = percentile_summary(phi_rms)
        et = percentile_summary(eta)
        vf = percentile_summary(valid_fraction)

        agg.append({
            "n": int(n),
            "total": int(3*n),
            "E_median": e["median"],
            "E_q1": e["q1"],
            "E_q3": e["q3"],
            "E_p05": e["p05"],
            "E_p95": e["p95"],
            "Aerr_median": a["median"],
            "Aerr_q1": a["q1"],
            "Aerr_q3": a["q3"],
            "IQRwidth_median": iw["median"],
            "IQRwidth_q1": iw["q1"],
            "IQRwidth_q3": iw["q3"],
            "phi_rms_median": pr["median"],
            "eta_sparse_median": et["median"],
            "valid_fraction_median": vf["median"],
            "valid_fraction_min": vf["min"],
            "pct_E_le_10": float(100*np.mean(E <= 0.10)),
            "pct_E_le_15": float(100*np.mean(E <= 0.15)),
            "pct_E_le_20": float(100*np.mean(E <= 0.20)),
        })
    return agg


# =============================================================================
# CSV OUTPUTS
# =============================================================================

def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def flatten_subvolume_rows(results):
    rows = []
    for r in results:
        Q = np.asarray(r["Q_ref"], dtype=float)
        row = {
            "subvolume_id": r["meta"]["id"],
            "grid_i": r["meta"]["grid_i"],
            "grid_j": r["meta"]["grid_j"],
            "grid_k": r["meta"]["grid_k"],
            "center_a0": r["meta"]["center"][0],
            "center_a1": r["meta"]["center"][1],
            "center_a2": r["meta"]["center"][2],
            "porosity": r["phi_sub"],
            "A_ref": r["A_ref"],
            "ell_geom_ref": r["ell_geom_ref"],
            "major_gap": r["major_gap"],
            "orientation_reliable": r["orientation_reliable"],
            "seed_eta_median": r["seed_eta_median"],
            "seed_pair_median": r["seed_pair_median"],
            "seed_pair_max": r["seed_pair_max"],
            "seed_A_sd": r["seed_A_sd"],
            "seed_ell_geom_cv": r["seed_ell_geom_cv"],
        }
        row.update(q_components(Q))
        rows.append(row)
    return rows


def flatten_summary_rows(results):
    rows = []
    for r in results:
        for s in r["sparse"]["summary"]:
            rows.append({
                "subvolume_id": r["meta"]["id"],
                "n_per_orientation": s["n_per_orientation"],
                "total_sections": s["total_sections"],
                "E_median": s["E_med"],
                "E_q1": s["E_q1"],
                "E_q3": s["E_q3"],
                "E_p05": s["E_p05"],
                "E_p95": s["E_p95"],
                "A_ref": r["A_ref"],
                "A_sparse_median": s["A_med"],
                "A_sparse_q1": s["A_q1"],
                "A_sparse_q3": s["A_q3"],
                "abs_A_error": abs(s["A_med"] - r["A_ref"]),
                "eta_sparse_median": s["eta_med"],
                "phi_rms_median": s["phi_rms_med"],
                "orientation_reliable": r["orientation_reliable"],
                "n_attempted": s.get("n_attempted", N_RANDOM_CONFIGS),
                "n_valid": s.get("n_valid", N_RANDOM_CONFIGS),
                "n_invalid": s.get("n_invalid", 0),
                "valid_fraction": s.get("valid_fraction", 1.0),
            })
    return rows


def flatten_random_rows(results):
    rows = []
    for r in results:
        sid = r["meta"]["id"]
        for x in r["sparse"]["random"]:
            vbf = x.get("valid_by_family", {})
            row = {
                "subvolume_id": sid,
                "trial": x["trial"],
                "n_per_orientation": x["n_per_orientation"],
                "total_sections": x["total_sections"],
                "offsets_a0_a1": ";".join(map(str, x["offsets_a0_a1"])),
                "offsets_a0_a2": ";".join(map(str, x["offsets_a0_a2"])),
                "offsets_a1_a2": ";".join(map(str, x["offsets_a1_a2"])),
                "valid": x.get("valid", True),
                "failure_reason": x.get("failure_reason", ""),
                "valid_directions": x.get("valid_directions", ""),
                "valid_dirs_a0_a1": vbf.get("a0-a1", ""),
                "valid_dirs_a0_a2": vbf.get("a0-a2", ""),
                "valid_dirs_a1_a2": vbf.get("a1-a2", ""),
                "E_Q": x.get("E_Q", np.nan),
                "A_sparse": x.get("A_sparse", np.nan),
                "eta": x.get("eta", np.nan),
                "rank": x.get("rank", ""),
                "condition": x.get("condition", np.nan),
                "phi_all": x.get("phi_all", np.nan),
                "phi_family_rms": x.get("phi_family_rms", np.nan),
                "major_axis_error_deg": x.get("major_axis_error_deg", np.nan),
                "orientation_reliable": x.get("orientation_reliable", False),
                "q00": x.get("q00", np.nan), "q11": x.get("q11", np.nan),
                "q22": x.get("q22", np.nan), "q01": x.get("q01", np.nan),
                "q02": x.get("q02", np.nan), "q12": x.get("q12", np.nan),
            }
            rows.append(row)
    return rows


def flatten_systematic_rows(results):
    rows = []
    for r in results:
        sid = r["meta"]["id"]
        for x in r["sparse"]["systematic"]:
            rows.append({
                "subvolume_id": sid,
                "n_per_orientation": x["n_per_orientation"],
                "total_sections": x["total_sections"],
                "offsets": ";".join(map(str, x["offsets"])),
                "valid": x.get("valid", True),
                "failure_reason": x.get("failure_reason", ""),
                "rank": x.get("rank", ""),
                "condition": x.get("condition", np.nan),
                "valid_directions": x.get("valid_directions", ""),
                "E_Q": x.get("E_Q", np.nan),
                "A_sparse": x.get("A_sparse", np.nan),
                "eta": x.get("eta", np.nan),
                "phi_family_rms": x.get("phi_family_rms", np.nan),
                "q00": x.get("q00", np.nan), "q11": x.get("q11", np.nan),
                "q22": x.get("q22", np.nan), "q01": x.get("q01", np.nan),
                "q02": x.get("q02", np.nan), "q12": x.get("q12", np.nan),
            })
    return rows


def flatten_refseed_rows(results):
    rows = []
    for r in results:
        for d in r["reference_seeds"]:
            Q = np.asarray(d["Q"], dtype=float)
            row = {
                "subvolume_id": r["meta"]["id"],
                "seed_index": d["seed_index"],
                "random_seed": d["random_seed"],
                "source": d["source"],
                "eta": d["eta"],
                "A": d["A"],
                "ell_geom": d["ell_geom"],
                "valid_directions": d["valid_directions"],
                "total_directions": d["total_directions"],
            }
            row.update(q_components(Q))
            rows.append(row)
    return rows


# =============================================================================
# REPORT
# =============================================================================

def write_report(results, agg, settings, elapsed):
    phi = np.array([r["phi_sub"] for r in results], dtype=float)
    Aref = np.array([r["A_ref"] for r in results], dtype=float)
    eta_ref = np.array([r["seed_eta_median"] for r in results], dtype=float)
    reliable = int(np.sum([r["orientation_reliable"] for r in results]))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Edwards Brown (EdB-1) final real-rock validation\n")
        f.write("=" * 78 + "\n\n")
        f.write("FROZEN REFERENCE PROTOCOL\n")
        f.write("-" * 78 + "\n")
        f.write(f"Directions: {len(fibonacci_hemisphere(N_REF_DIRECTIONS))}\n")
        f.write(f"Points/direction/seed: {settings['ref_points']:,}\n")
        f.write(f"Independent seeds/reference: {settings['ref_seeds']}\n")
        f.write("Consensus: Euclidean mean of seed Q tensors, projected to SPD\n")
        f.write(f"Calibrated consensus uncertainty median: {100*CALIBRATED_REF_U_MEDIAN:.3f}%\n")
        f.write(f"Calibrated consensus uncertainty p95: {100*CALIBRATED_REF_U_P95:.3f}%\n")
        f.write(f"Calibration worst-subvolume p95: {100*CALIBRATED_REF_U_WORST_P95:.3f}%\n")
        f.write("Reference uncertainty is reported separately and is not subtracted from E_Q.\n\n")

        f.write("SPATIAL DESIGN\n")
        f.write("-" * 78 + "\n")
        f.write(f"RAW: {RAW_FILE}\n")
        f.write(f"RAW shape: {RAW_SHAPE}, convention 0=pore, 1=solid (EdB-1 verified)\n")
        f.write(f"Subvolume size: {SUBVOL_N}^3\n")
        f.write(f"Grid centers: {GRID_CENTERS}\n")
        f.write(f"Non-overlapping subvolumes: {len(results)}\n")
        f.write(f"Section counts/orientation: {SECTIONS_PER_ORIENTATION}\n")
        f.write(f"Random offset configurations/count: {settings['random_configs']}\n")
        f.write("Axes a0/a1/a2 are computational array axes.\n\n")

        f.write("REFERENCE SPATIAL HETEROGENEITY\n")
        f.write("-" * 78 + "\n")
        f.write(
            f"Porosity: median={100*np.median(phi):.3f}%  "
            f"range=[{100*np.min(phi):.3f}, {100*np.max(phi):.3f}]%\n"
        )
        f.write(
            f"A_ref: median={np.median(Aref):.4f}  "
            f"range=[{np.min(Aref):.4f}, {np.max(Aref):.4f}]\n"
        )
        f.write(
            f"Seed-level 3D tensor-fit residual: median={100*np.median(eta_ref):.3f}%\n"
        )
        f.write(f"Reliable major-axis flag: {reliable}/{len(results)} subvolumes\n\n")

        f.write("FINAL SPARSE-RECOVERY SUMMARY\n")
        f.write("-" * 78 + "\n")
        for a in agg:
            f.write(
                f"n/orientation={a['n']} total={a['total']}: "
                f"spatial median E_Q={100*a['E_median']:.3f}% "
                f"IQR=[{100*a['E_q1']:.3f}, {100*a['E_q3']:.3f}]%; "
                f"median |dA|={a['Aerr_median']:.5f}; "
                f"median position-IQR={100*a['IQRwidth_median']:.3f} pp; "
                f"median phi-family-RMS={100*a['phi_rms_median']:.3f} pp; "
                f"median sparse residual={100*a['eta_sparse_median']:.3f}%; "
                f"rank-6 random configs median/min={100*a['valid_fraction_median']:.1f}/"
                f"{100*a['valid_fraction_min']:.1f}%; "
                f"E<=10/15/20%: {a['pct_E_le_10']:.1f}/"
                f"{a['pct_E_le_15']:.1f}/{a['pct_E_le_20']:.1f}%\n"
            )

        f.write("\nINTERPRETATION BOUNDARIES\n")
        f.write("-" * 78 + "\n")
        f.write(
            "Three oriented planes are sufficient for full tensor identifiability "
            "when the design matrix has rank six, but one plane per orientation "
            "need not be statistically representative of a heterogeneous rock.\n"
        )
        f.write(
            "Random offset resampling quantifies sensitivity to section position; "
            "spatial replication across 27 non-overlapping subvolumes quantifies "
            "reproducibility within this single ROI.\n"
        )
        f.write(
            "If a random section-position draw does not provide six independent "
            "tensor constraints because too few 1/e crossings are observed within "
            "MAX_LAG, it is retained as non-identifiable rather than replaced by a "
            "pseudoinverse or silently resampled. E_Q summaries are conditional on "
            "rank-six draws, and the rank-six fraction is reported separately.\n"
        )
        f.write(
            "The 27 subvolumes belong to one plug/ROI and are not independent "
            "specimens; no specimen-population inference is implied.\n"
        )
        f.write(
            "Principal-axis angle errors should not be emphasized when the "
            "reference eigenvalue spectrum is nearly degenerate.\n"
        )
        f.write(f"\nWall time this invocation: {elapsed:.1f} s\n")


# =============================================================================
# FIGURE HELPERS
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

    n_sub = len(data[0])
    x = np.arange(1, len(nvals)+1)
    for i in range(n_sub):
        y = [data[j][i] for j in range(len(nvals))]
        ax.plot(x, y, linewidth=0.55, alpha=0.20, color="0.35")

    ax.set_xlabel("Total number of sections")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def representative_subvolume(results):
    phi = np.array([r["phi_sub"] for r in results])
    A = np.array([r["A_ref"] for r in results])
    p0 = np.median(phi)
    a0 = np.median(A)
    # Scale both terms by robust spread so neither dominates.
    ps = max(np.percentile(phi, 75) - np.percentile(phi, 25), 1e-6)
    aas = max(np.percentile(A, 75) - np.percentile(A, 25), 1e-6)
    score = ((phi-p0)/ps)**2 + ((A-a0)/aas)**2
    return results[int(np.argmin(score))]


def load_representative_sections(raw, result):
    m = result["meta"]
    s0, s1, s2 = m["start"]
    e0, e1, e2 = m["stop"]
    c0, c1, c2 = m["center"]

    # Copy only three 512x512 planes.
    a0a1 = np.array(raw[s0:e0, s1:e1, c2], dtype=np.uint8, copy=True)
    a0a2 = np.array(raw[s0:e0, c1, s2:e2], dtype=np.uint8, copy=True)
    a1a2 = np.array(raw[c0, s1:e1, s2:e2], dtype=np.uint8, copy=True)
    return [
        ("a0-a1", a0a1 == PORE_VALUE),
        ("a0-a2", a0a2 == PORE_VALUE),
        ("a1-a2", a1a2 == PORE_VALUE),
    ]


def choose_representative_sparse_Q(result, n=7):
    rows = [
        r for r in result["sparse"]["random"]
        if r["n_per_orientation"] == n
    ]
    target = np.median([r["E_Q"] for r in rows])
    row = min(rows, key=lambda r: abs(r["E_Q"] - target))
    return q_from_components(row), row


def ellipsoid_mesh_from_Q(Q, n_u=56, n_v=34):
    Qn = normalize_det_spd(Q)
    ell, V = principal_lengths_and_axes(Qn)
    u = np.linspace(0.0, 2.0*np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    U, Vv = np.meshgrid(u, v)
    local = np.stack([
        ell[0] * np.cos(U) * np.sin(Vv),
        ell[1] * np.sin(U) * np.sin(Vv),
        ell[2] * np.cos(Vv),
    ], axis=0)
    xyz = np.einsum("ij,jkl->ikl", V, local)
    return xyz[0], xyz[1], xyz[2]


def build_figure(results, agg, rep_sections):
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.43, wspace=0.36)

    # ------------------------------------------------------------------ (a)
    sub = gs[0, 0].subgridspec(1, 3, wspace=0.04)
    axes_a = []
    for i, (name, pore) in enumerate(rep_sections):
        ax = fig.add_subplot(sub[0, i])
        ax.imshow(pore.T, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=8.8, pad=3, fontweight="bold")
        axes_a.append(ax)
    axes_a[1].text(
        0.5, 1.18,
        "Representative Edwards Brown (EdB-1) sections",
        transform=axes_a[1].transAxes,
        ha="center", va="bottom",
        fontsize=11.2, fontweight="bold",
    )
    panel_label(axes_a[0], "(a)", x=-0.07, y=1.10)

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    phi = 100*np.array([r["phi_sub"] for r in results], dtype=float)
    Aref = np.array([r["A_ref"] for r in results], dtype=float)
    ax.scatter(phi, Aref, s=30, alpha=0.82)
    ax.axvline(np.median(phi), linestyle=":", color="black", linewidth=0.9)
    ax.axhline(np.median(Aref), linestyle=":", color="black", linewidth=0.9)
    ax.set_xlabel("Subvolume porosity (%)")
    ax.set_ylabel(r"Reference anisotropy $A_{ref}$")
    ax.set_title("Spatial heterogeneity of 3D reference")
    ax.text(
        0.04, 0.96,
        f"27 non-overlapping 512³ regions\n"
        f"median $A_{{ref}}$ = {np.median(Aref):.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.4, fontweight="bold",
    )
    panel_label(ax, "(b)")

    # Prepare fixed-order arrays.
    E_by_n = {}
    IQR_by_n = {}
    Aerr_by_n = {}
    for n in SECTIONS_PER_ORIENTATION:
        E_by_n[n] = [summary_for_n(r, n)["E_med"] for r in results]
        IQR_by_n[n] = [
            summary_for_n(r, n)["E_q3"] - summary_for_n(r, n)["E_q1"]
            for r in results
        ]
        Aerr_by_n[n] = [
            abs(summary_for_n(r, n)["A_med"] - r["A_ref"])
            for r in results
        ]

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    paired_boxplot(
        ax,
        E_by_n,
        ylabel=r"Tensor-shape error $E_Q$ (%)",
        title="Sparse recovery improves with replication",
        scale=100.0,
    )
    ax.axhline(
        100*CALIBRATED_REF_U_P95,
        linestyle="--", color="black", linewidth=1.0,
    )
    ax.text(
        0.98, 100*CALIBRATED_REF_U_P95 + 0.35,
        "reference p95 = 1.71%",
        ha="right", va="bottom", fontsize=7.8, fontweight="bold",
    )
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0])
    paired_boxplot(
        ax,
        IQR_by_n,
        ylabel=r"Within-region $E_Q$ IQR (pp)",
        title="Position sensitivity decreases",
        scale=100.0,
    )
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    totals = np.array([a["total"] for a in agg], dtype=int)
    p10 = np.array([a["pct_E_le_10"] for a in agg])
    p15 = np.array([a["pct_E_le_15"] for a in agg])
    p20 = np.array([a["pct_E_le_20"] for a in agg])
    ax.plot(totals, p10, marker="o", label=r"$E_Q\leq10\%$")
    ax.plot(totals, p15, marker="s", linestyle="--", label=r"$E_Q\leq15\%$")
    ax.plot(totals, p20, marker="^", linestyle=":", label=r"$E_Q\leq20\%$")
    ax.set_xticks(totals)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel("Subvolumes meeting threshold (%)")
    ax.set_title("Spatial success rate")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    A_sparse = np.array([summary_for_n(r, 7)["A_med"] for r in results])
    lo = 0.97 * min(Aref.min(), A_sparse.min())
    hi = 1.03 * max(Aref.max(), A_sparse.max())
    ax.scatter(Aref, A_sparse, s=32, alpha=0.82)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Reference anisotropy $A_{ref}$")
    ax.set_ylabel(r"Sparse median anisotropy $A_{sparse}$")
    ax.set_title("Anisotropy recovery with 21 sections")
    med_abs = np.median(np.abs(A_sparse - Aref))
    ax.text(
        0.04, 0.95,
        f"median |ΔA| = {med_abs:.3f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.4, fontweight="bold",
    )
    panel_label(ax, "(f)")

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.955)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_all = time.time()
    settings = active_settings()

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\nRAW file not found:\n{RAW_FILE.resolve()}\n\n"
            "The default RAW_FILE is an absolute EdB-1 path. If your file is "
            "elsewhere, edit RAW_FILE at the top of the script."
        )

    validate_raw(RAW_FILE)

    grid = spatial_grid()
    directions = fibonacci_hemisphere(N_REF_DIRECTIONS)
    ref_signature = final_ref_signature(
        settings["ref_points"], settings["ref_seeds"]
    )

    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINAL VALIDATION PROTOCOL")
    print("=" * 78)
    print(f"Subvolumes:                 {len(grid)} non-overlapping 512^3 regions")
    print(f"Reference directions:       {len(directions)}")
    print(f"Reference points/direction: {settings['ref_points']:,}")
    print(f"Reference seeds:            {settings['ref_seeds']}")
    print("Reference consensus:        mean(Q_seed) -> SPD")
    print(f"Random section configs/n:   {settings['random_configs']}")
    print(f"Sections/orientation:       {SECTIONS_PER_ORIENTATION}")
    print(f"Calibrated reference p95:   {100*CALIBRATED_REF_U_P95:.3f}%")
    if not settings["is_final"]:
        print("WARNING: QUICK_MODE=True -- numbers are NOT final manuscript results.")
    else:
        print("Protocol status:            FROZEN FINAL")
    print()

    raw = np.memmap(
        RAW_FILE,
        dtype=RAW_DTYPE,
        mode="r",
        shape=RAW_SHAPE,
        order="C",
    )

    results = []

    for idx, meta in enumerate(grid, start=1):
        sid = meta["id"]
        t_sub = time.time()
        print("=" * 78)
        print(f"[{idx:02d}/{len(grid):02d}] {sid}  center={meta['center']}")
        print("=" * 78)

        # --------------------------------------------------------------
        # Reference seeds: load final cache, import Test-4 cache, or compute.
        # --------------------------------------------------------------
        seed_results = []
        missing_seed_indices = []

        for seed_index in range(settings["ref_seeds"]):
            p = ref_cache_path(sid, seed_index)
            d = load_final_ref_cache(p, ref_signature, meta)
            if d is None:
                imported = try_import_test4_seed(
                    meta, seed_index, ref_signature, settings["ref_points"]
                )
                if imported is not None:
                    save_json_atomic(p, imported)
                    d = imported
                    print(
                        f"    reference seed {seed_index+1:02d}/"
                        f"{settings['ref_seeds']:02d} imported from Test-4 cache"
                    )
            if d is None:
                missing_seed_indices.append(seed_index)
            else:
                seed_results.append(d)

        # --------------------------------------------------------------
        # 2D section cache.
        # --------------------------------------------------------------
        sec_path = section_cache_path(sid)
        section_cache = load_section_cache(sec_path, meta)
        need_sections = section_cache is None

        # Load this 512^3 cube only if a reference seed or section pool is missing.
        phase = None
        if missing_seed_indices or need_sections:
            print(f"    loading {SUBVOL_N}^3 subvolume into RAM...")
            phase = extract_subvolume(raw, meta)
            phi_sub = float(np.mean(phase == PORE_VALUE))
            print(f"    porosity = {100*phi_sub:.3f}%")
        else:
            # Porosity is stored in reference seed cache.
            phi_sub = float(seed_results[0]["phi_sub"])
            print(f"    all heavy data cached; porosity = {100*phi_sub:.3f}%")

        if missing_seed_indices:
            for seed_index in missing_seed_indices:
                d = compute_final_reference_seed(
                    phase,
                    meta,
                    seed_index,
                    ref_signature,
                    settings["ref_points"],
                )
                save_json_atomic(ref_cache_path(sid, seed_index), d)
                seed_results.append(d)
                print(
                    f"    cached reference seed {seed_index:02d} "
                    f"({d['elapsed_sec']:.1f}s)"
                )

        seed_results = sorted(seed_results, key=lambda d: d["seed_index"])
        if len(seed_results) != settings["ref_seeds"]:
            raise RuntimeError(
                f"{sid}: expected {settings['ref_seeds']} reference seeds, "
                f"found {len(seed_results)}"
            )

        # Confirm all seed porosities agree with exact selected cube porosity.
        phi_seed = np.array([d["phi_sub"] for d in seed_results], dtype=float)
        if np.max(np.abs(phi_seed - phi_seed[0])) > 1e-12:
            raise RuntimeError(f"{sid}: inconsistent porosity across seed caches.")
        phi_sub = float(phi_seed[0])

        ref = reference_consensus(seed_results)
        Q_ref = ref["Q_ref"]

        print(
            f"    Q_REF consensus: A={ref['A_ref']:.4f}  "
            f"seed-eta med={100*ref['seed_eta_median']:.2f}%  "
            f"gap={100*ref['major_gap']:.2f}%  "
            f"orientation_reliable={ref['orientation_reliable']}"
        )

        if need_sections:
            print("    computing 39 section-offset measurements...")
            section_cache = precompute_section_pool(phase, meta)
        else:
            print("    39 section-offset measurements loaded from cache")

        # Release cube before random reconstructions.
        if phase is not None:
            del phase
            gc.collect()

        sparse = process_sparse_results(
            meta,
            section_cache,
            Q_ref,
            phi_sub,
            ref["orientation_reliable"],
            settings["random_configs"],
        )

        print("    final sparse summary:")
        for s in sparse["summary"]:
            print(
                f"      n/orient={s['n_per_orientation']} total={s['total_sections']:2d}  "
                f"E_Q med={100*s['E_med']:6.2f}% "
                f"IQR=[{100*s['E_q1']:.2f},{100*s['E_q3']:.2f}]%  "
                f"A med={s['A_med']:.4f}  "
                f"phi-RMS={100*s['phi_rms_med']:.3f} pp  "
                f"rank6={100*s.get('valid_fraction', 1.0):.1f}%"
            )

        result = {
            "meta": meta,
            "phi_sub": phi_sub,
            "Q_ref": [[float(v) for v in row] for row in Q_ref],
            "A_ref": float(ref["A_ref"]),
            "ell_geom_ref": float(ref["ell_geom_ref"]),
            "major_gap": float(ref["major_gap"]),
            "orientation_reliable": bool(ref["orientation_reliable"]),
            "seed_eta_median": float(ref["seed_eta_median"]),
            "seed_eta_q1": float(ref["seed_eta_q1"]),
            "seed_eta_q3": float(ref["seed_eta_q3"]),
            "seed_A_sd": float(ref["seed_A_sd"]),
            "seed_ell_geom_cv": float(ref["seed_ell_geom_cv"]),
            "seed_pair_median": float(ref["seed_pair_median"]),
            "seed_pair_max": float(ref["seed_pair_max"]),
            "reference_seeds": seed_results,
            "sparse": sparse,
        }
        results.append(result)
        print(f"    {sid} completed in {time.time()-t_sub:.1f}s\n")

        del section_cache
        gc.collect()

    del raw

    # ------------------------------------------------------------------
    # Spatial aggregation and final outputs.
    # ------------------------------------------------------------------
    agg = aggregate_spatial(results)

    print("=" * 78)
    print("FINAL SPATIAL-REPLICATION SUMMARY")
    print("=" * 78)
    phi = np.array([r["phi_sub"] for r in results], dtype=float)
    Aref = np.array([r["A_ref"] for r in results], dtype=float)
    eta_ref = np.array([r["seed_eta_median"] for r in results], dtype=float)
    reliable = int(np.sum([r["orientation_reliable"] for r in results]))

    print(
        f"Subvolume porosity: median={100*np.median(phi):.3f}%  "
        f"range=[{100*np.min(phi):.3f},{100*np.max(phi):.3f}]%"
    )
    print(
        f"A_ref consensus:    median={np.median(Aref):.4f}  "
        f"range=[{np.min(Aref):.4f},{np.max(Aref):.4f}]"
    )
    print(f"Seed-level ref eta: median={100*np.median(eta_ref):.3f}%")
    print(f"Reliable major axis: {reliable}/{len(results)} subvolumes")
    print(
        f"Frozen reference uncertainty: median={100*CALIBRATED_REF_U_MEDIAN:.3f}%  "
        f"p95={100*CALIBRATED_REF_U_P95:.3f}%"
    )
    print()

    for a in agg:
        print(
            f"n/orientation={a['n']} total={a['total']:2d}  "
            f"spatial median E_Q={100*a['E_median']:6.2f}%  "
            f"IQR=[{100*a['E_q1']:.2f},{100*a['E_q3']:.2f}]%  "
            f"median |dA|={a['Aerr_median']:.4f}  "
            f"position-IQR={100*a['IQRwidth_median']:.2f} pp  "
            f"rank6cfg={100*a['valid_fraction_median']:.0f}% "
            f"(min {100*a['valid_fraction_min']:.0f}%)  "
            f"E<=10/15/20%: {a['pct_E_le_10']:.0f}/"
            f"{a['pct_E_le_15']:.0f}/{a['pct_E_le_20']:.0f}%"
        )

    # CSV files.
    sub_rows = flatten_subvolume_rows(results)
    sum_rows = flatten_summary_rows(results)
    rnd_rows = flatten_random_rows(results)
    sys_rows = flatten_systematic_rows(results)
    ref_rows = flatten_refseed_rows(results)

    write_csv(
        SUBVOL_CSV_PATH,
        sub_rows,
        [
            "subvolume_id", "grid_i", "grid_j", "grid_k",
            "center_a0", "center_a1", "center_a2",
            "porosity", "A_ref", "ell_geom_ref", "major_gap",
            "orientation_reliable", "seed_eta_median",
            "seed_pair_median", "seed_pair_max", "seed_A_sd",
            "seed_ell_geom_cv",
            "q00", "q11", "q22", "q01", "q02", "q12",
        ],
    )
    write_csv(
        SUMMARY_CSV_PATH,
        sum_rows,
        [
            "subvolume_id", "n_per_orientation", "total_sections",
            "E_median", "E_q1", "E_q3", "E_p05", "E_p95",
            "A_ref", "A_sparse_median", "A_sparse_q1", "A_sparse_q3",
            "abs_A_error", "eta_sparse_median", "phi_rms_median",
            "orientation_reliable", "n_attempted", "n_valid", "n_invalid",
            "valid_fraction",
        ],
    )
    write_csv(
        RANDOM_CSV_PATH,
        rnd_rows,
        [
            "subvolume_id", "trial", "n_per_orientation", "total_sections",
            "offsets_a0_a1", "offsets_a0_a2", "offsets_a1_a2",
            "valid", "failure_reason", "valid_directions",
            "valid_dirs_a0_a1", "valid_dirs_a0_a2", "valid_dirs_a1_a2",
            "E_Q", "A_sparse", "eta", "rank", "condition",
            "phi_all", "phi_family_rms", "major_axis_error_deg",
            "orientation_reliable",
            "q00", "q11", "q22", "q01", "q02", "q12",
        ],
    )
    write_csv(
        SYSTEMATIC_CSV_PATH,
        sys_rows,
        [
            "subvolume_id", "n_per_orientation", "total_sections", "offsets",
            "valid", "failure_reason", "rank", "condition", "valid_directions",
            "E_Q", "A_sparse", "eta", "phi_family_rms",
            "q00", "q11", "q22", "q01", "q02", "q12",
        ],
    )
    write_csv(
        REFSEED_CSV_PATH,
        ref_rows,
        [
            "subvolume_id", "seed_index", "random_seed", "source",
            "eta", "A", "ell_geom", "valid_directions", "total_directions",
            "q00", "q11", "q22", "q01", "q02", "q12",
        ],
    )

    # Representative raw sections for panel (a).
    rep = representative_subvolume(results)
    raw = np.memmap(
        RAW_FILE,
        dtype=RAW_DTYPE,
        mode="r",
        shape=RAW_SHAPE,
        order="C",
    )
    rep_sections = load_representative_sections(raw, rep)
    del raw

    build_figure(results, agg, rep_sections)
    elapsed = time.time() - t_all
    write_report(results, agg, settings, elapsed)

    print("\n" + "=" * 78)
    print("FILES GENERATED")
    print("=" * 78)
    for p in [
        PNG_PATH, SVG_PATH, REPORT_PATH,
        SUBVOL_CSV_PATH, SUMMARY_CSV_PATH, RANDOM_CSV_PATH,
        SYSTEMATIC_CSV_PATH, REFSEED_CSV_PATH,
        REF_CACHE_DIR, SECTION_CACHE_DIR, RESULT_CACHE_DIR,
    ]:
        print(p)

    print("\nFINAL PROTOCOL COMPLETE.")
    if settings["is_final"]:
        print("No further reference-calibration test is required.")
    else:
        print("QUICK_MODE was used; rerun with QUICK_MODE=False for final numbers.")


if __name__ == "__main__":
    main()