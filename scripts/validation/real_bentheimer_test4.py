#!/usr/bin/env python3
"""
Bentheimer real-rock validation: test 4
========================================

Reference-estimator convergence study.

Goal
----
Quantify the Monte Carlo uncertainty of the full-3D reference tensor Q3D_REF
used in Bentheimer tests 1-3, before freezing the real-rock validation figure.

The directional estimator is intentionally kept the SAME as in tests 1-3:

    * same 512^3 subvolume size;
    * same 1/e directional correlation length;
    * same Fibonacci hemisphere + computational axes;
    * same MAX_LAG;
    * same trilinear off-lattice sampling;
    * same tensor fit in y = 1 / ell^2.

Only the number of random base points per 3D direction is varied.

Five spatially separated Bentheimer subvolumes are used, chosen from Test 3 to
span a broad range of preliminary anisotropy / porosity behavior:

    G11, G05, G00, G18, G04

For every subvolume, N_SEEDS independent Monte Carlo seeds are evaluated at
nested point counts:

    3,000; 6,000; 12,000; 25,000; 50,000 points / direction.

IMPORTANT DESIGN DETAIL
-----------------------
For a given seed, the smaller point counts are prefixes of the same 50,000-point
sample.  This makes the convergence curves paired and avoids wasting repeated
interpolation work.  Different seeds are independent.

There is no analytic ground-truth tensor for a real rock.  Therefore the main
uncertainty metric does NOT compare to a claimed truth.  Instead, for every
seed we build a leave-one-seed-out consensus from the OTHER seeds at the
largest point count (50,000 by default):

    Q_target^(-s) = mean_{j != s} Q_j(N_max)

and report

    E_LOO(N,s) = ||Q~_N,s - Q~_target^(-s)||_F / ||Q~_target^(-s)||_F,

where Q~ is determinant-normalized.

We also report pairwise seed-to-seed tensor dispersion, which does not depend on
any consensus target.

This test isolates MONTE CARLO sampling uncertainty of the current reference
estimator.  It does not by itself establish that the estimator is unbiased with
respect to an ideal infinite-domain correlation definition.

RAW convention already validated:
    RAW value 0 = pore
    RAW value 1 = solid

Array axes are called a0/a1/a2 until the physical axis mapping is confirmed.

Outputs
-------
  bentheimer_test4_reference_convergence.png
  bentheimer_test4_reference_convergence.svg
  bentheimer_test4_report.txt
  bentheimer_test4_runs.csv
  bentheimer_test4_subvolume_summary.csv
  bentheimer_test4_global_summary.csv
  bentheimer_test4_pairwise.csv
  bentheimer_test4_cache/          restart-safe seed-level JSON cache

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

Run
---
  python3 real_bentheimer_test4_reference_convergence.py

The full 2500^3 RAW is opened with numpy.memmap.  Only one 512^3 subvolume is
copied to RAM at a time.
"""

from pathlib import Path
import csv
import gc
import json
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates


# =============================================================================
# USER SETTINGS
# =============================================================================

RAW_FILE = Path(
    "kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)
RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8

SUBVOL_N = 512

# Selected non-overlapping Test-3 subvolumes.
# Labels are descriptive only; final anisotropy is re-estimated here.
SELECTED_SUBVOLUMES = (
    {"id": "G11", "center": (1250, 384, 2116), "note": "low preliminary A"},
    {"id": "G05", "center": (384, 1250, 2116), "note": "low-intermediate preliminary A"},
    {"id": "G00", "center": (384, 384, 384), "note": "intermediate preliminary A"},
    {"id": "G18", "center": (2116, 384, 384), "note": "higher preliminary A"},
    {"id": "G04", "center": (384, 1250, 1250), "note": "highest preliminary A"},
)

# Kept identical to the reference estimator used in tests 1-3.
MAX_LAG = 48
N_REF_DIRECTIONS = 64       # Fibonacci directions; three axes are added below.

# Monte Carlo convergence design.
POINT_COUNTS = (3_000, 6_000, 12_000, 25_000, 50_000)
N_SEEDS = 10
SEED = 20260907

# Diagnostic convergence thresholds for the leave-one-out tensor-shape error.
CONVERGENCE_THRESHOLDS = (0.01, 0.02, 0.03)

DPI = 800
QUICK_MODE = False
FORCE_RECOMPUTE = False

# QUICK_MODE keeps all five subvolumes but reduces the convergence grid.
QUICK_POINT_COUNTS = (3_000, 12_000, 25_000)
QUICK_N_SEEDS = 3


# =============================================================================
# OUTPUTS / STYLE
# =============================================================================

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "bentheimer_test4_reference_convergence.png"
SVG_PATH = OUTDIR / "bentheimer_test4_reference_convergence.svg"
REPORT_PATH = OUTDIR / "bentheimer_test4_report.txt"
RUNS_CSV_PATH = OUTDIR / "bentheimer_test4_runs.csv"
SUBVOL_SUMMARY_CSV_PATH = OUTDIR / "bentheimer_test4_subvolume_summary.csv"
GLOBAL_SUMMARY_CSV_PATH = OUTDIR / "bentheimer_test4_global_summary.csv"
PAIRWISE_CSV_PATH = OUTDIR / "bentheimer_test4_pairwise.csv"
CACHE_DIR = OUTDIR / "bentheimer_test4_cache"

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

def project_spd(Q):
    Q = 0.5 * (np.asarray(Q, dtype=float) + np.asarray(Q, dtype=float).T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-8 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T


def normalize_det_spd(Q):
    Q = project_spd(Q)
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
    """Geometric mean of the three principal correlation lengths."""
    Q = project_spd(Q)
    return float(np.linalg.det(Q) ** (-1.0 / 6.0))


def tensor_shape_error(Q_est, Q_ref):
    Qe = normalize_det_spd(Q_est)
    Qr = normalize_det_spd(Q_ref)
    return float(
        np.linalg.norm(Qe - Qr, ord="fro") /
        np.linalg.norm(Qr, ord="fro")
    )


def tensor_pair_error(Q1, Q2):
    """Symmetric determinant-normalized Frobenius separation."""
    A = normalize_det_spd(Q1)
    B = normalize_det_spd(Q2)
    den = 0.5 * (np.linalg.norm(A, ord="fro") + np.linalg.norm(B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / den)


def consensus_Q(Q_list):
    """Euclidean mean in Q-space, projected back to SPD."""
    return project_spd(np.mean(np.stack(Q_list, axis=0), axis=0))


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
            f"RAW size mismatch: expected {expected:,}, found {actual:,}."
        )
    print("Status:   PERFECT MATCH\n")


def subvolume_bounds(center):
    half = SUBVOL_N // 2
    starts = tuple(int(c - half) for c in center)
    stops = tuple(int(s + SUBVOL_N) for s in starts)

    for s, e, d in zip(starts, stops, RAW_SHAPE):
        if s < 0 or e > d:
            raise ValueError(f"Subvolume center {center} falls outside RAW bounds.")
    return starts, stops


def extract_subvolume(raw, center, subvol_id):
    starts, stops = subvolume_bounds(center)
    sl = tuple(slice(s, e) for s, e in zip(starts, stops))

    print("=" * 78)
    print(f"SUBVOLUME {subvol_id}  center={center}")
    print("=" * 78)
    print(f"start={starts}  stop={stops}")
    print(f"Copying {SUBVOL_N}^3 subvolume to RAM...")

    phase = np.array(raw[sl], dtype=np.uint8, copy=True)
    values = np.unique(phase)
    if not np.all(np.isin(values, [0, 1])):
        raise RuntimeError(f"Unexpected phase values: {values.tolist()}")

    phi = float(np.mean(phase == 0))
    print(f"porosity={100*phi:.3f}%\n")
    return phase, starts, stops, phi


# =============================================================================
# DIRECTIONAL REFERENCE ESTIMATOR
# =============================================================================

def fibonacci_hemisphere(n):
    """Approximately uniform axial directions over one hemisphere + 3 axes."""
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


def directional_lengths_nested(phase, directions, max_lag, point_counts, seed):
    """
    Estimate directional 1/e lengths for several nested Monte Carlo sample sizes.

    One maximum-size base-point set is generated.  Each smaller point count is a
    prefix of that set, so every seed yields paired convergence estimates.
    """
    point_counts = tuple(sorted(int(v) for v in point_counts))
    max_points = max(point_counts)

    rng = np.random.default_rng(seed)
    n = phase.shape[0]
    margin = int(np.ceil(max_lag)) + 2
    if 2 * margin >= n:
        raise ValueError("MAX_LAG is too large for the selected subvolume.")

    phi_pore = float(np.mean(phase == 0))
    variance = phi_pore * (1.0 - phi_pore)
    if variance <= 0:
        raise RuntimeError("Degenerate pore/solid subvolume.")

    base = rng.integers(
        low=margin,
        high=n - margin,
        size=(3, max_points),
        endpoint=False,
        dtype=np.int32,
    )

    chi0 = 1.0 - phase[base[0], base[1], base[2]].astype(np.float32)
    f0 = chi0 - np.float32(phi_pore)

    lags = np.arange(max_lag + 1, dtype=np.float32)
    ell_by_n = {
        npts: np.full(len(directions), np.nan, dtype=np.float64)
        for npts in point_counts
    }

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
        ).reshape(max_points, len(lags))

        chi_r = 1.0 - sampled_solid
        fr = chi_r - np.float32(phi_pore)

        # Product matrix is float32 to keep memory moderate; reductions are
        # explicitly accumulated in float64.
        products = f0[:, None] * fr

        for npts in point_counts:
            C = np.mean(products[:npts], axis=0, dtype=np.float64) / variance
            C[0] = 1.0
            ell_by_n[npts][j] = correlation_length(lags, C)

        if ((j + 1) % max(1, len(directions) // 10) == 0
                or j == len(directions) - 1):
            elapsed = time.time() - t0
            print(
                f"    {j+1:3d}/{len(directions)} directions "
                f"({100*(j+1)/len(directions):5.1f}%)  elapsed={elapsed:6.1f}s"
            )

        del coords, sampled_solid, chi_r, fr, products

    return ell_by_n, phi_pore, time.time() - t0


# =============================================================================
# CACHE
# =============================================================================

def active_settings():
    point_counts = QUICK_POINT_COUNTS if QUICK_MODE else POINT_COUNTS
    n_seeds = QUICK_N_SEEDS if QUICK_MODE else N_SEEDS
    return tuple(point_counts), int(n_seeds)


def cache_signature(point_counts, n_seeds):
    return {
        "raw_shape": list(RAW_SHAPE),
        "raw_dtype": str(np.dtype(RAW_DTYPE)),
        "subvol_n": int(SUBVOL_N),
        "max_lag": int(MAX_LAG),
        "n_ref_directions": int(N_REF_DIRECTIONS),
        "actual_point_counts": [int(v) for v in point_counts],
        "n_seeds": int(n_seeds),
        "seed": int(SEED),
        "estimator_version": 1,
    }


def cache_path(subvol_id, seed_index):
    return CACHE_DIR / f"{subvol_id}_seed{seed_index:02d}.json"


def save_json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def load_seed_cache(path, signature):
    if FORCE_RECOMPUTE or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("signature") != signature:
            return None
        return data
    except Exception:
        return None


# =============================================================================
# PER-SEED COMPUTATION
# =============================================================================

def compute_seed_result(phase, subvol_meta, seed_index, point_counts, signature):
    subvol_id = subvol_meta["id"]
    random_seed = int(SEED + 100_000 * (1 + subvol_meta["order"]) + seed_index)

    print(
        f"  seed {seed_index+1:02d}/{signature['n_seeds']:02d}  "
        f"rng_seed={random_seed}  max_points={max(point_counts):,}"
    )

    directions = fibonacci_hemisphere(N_REF_DIRECTIONS)
    ell_by_n, phi_check, elapsed = directional_lengths_nested(
        phase,
        directions,
        max_lag=MAX_LAG,
        point_counts=point_counts,
        seed=random_seed,
    )

    phi_sub = float(np.mean(phase == 0))
    if not np.isclose(phi_sub, phi_check):
        raise RuntimeError("Internal porosity consistency check failed.")

    point_results = {}
    for npts in point_counts:
        ell = ell_by_n[npts]
        valid = np.isfinite(ell) & (ell > 0)
        Q, rank, cond, eta, _, _ = fit_q3d_from_directional_lengths(
            directions, ell
        )

        point_results[str(int(npts))] = {
            "n_points": int(npts),
            "valid_directions": int(np.sum(valid)),
            "total_directions": int(len(ell)),
            "Q": [[float(v) for v in row] for row in Q],
            "rank": int(rank),
            "condition": float(cond),
            "eta": float(eta),
            "A": float(anisotropy_ratio(Q)),
            "ell_geom": float(geometric_correlation_length(Q)),
        }

    return {
        "signature": signature,
        "subvolume": {
            "id": subvol_id,
            "center": list(subvol_meta["center"]),
            "note": subvol_meta["note"],
            "order": int(subvol_meta["order"]),
        },
        "seed_index": int(seed_index),
        "random_seed": random_seed,
        "phi_sub": phi_sub,
        "elapsed_sec": float(elapsed),
        "point_results": point_results,
    }


# =============================================================================
# AGGREGATION
# =============================================================================

def percentile_summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def build_analysis(all_seed_data, point_counts, n_seeds):
    """
    Build leave-one-out and pairwise convergence metrics.

    Returns
    -------
    run_rows, pair_rows, subvol_summary_rows, global_summary_rows, consensus_info
    """
    max_n = max(point_counts)
    run_rows = []
    pair_rows = []
    subvol_summary_rows = []
    consensus_info = {}

    for meta in SELECTED_SUBVOLUMES:
        sid = meta["id"]
        seed_data = sorted(
            all_seed_data[sid], key=lambda d: d["seed_index"]
        )
        if len(seed_data) != n_seeds:
            raise RuntimeError(f"Expected {n_seeds} seeds for {sid}, found {len(seed_data)}")

        Qmax = [
            np.asarray(d["point_results"][str(max_n)]["Q"], dtype=float)
            for d in seed_data
        ]
        Q_cons_all = consensus_Q(Qmax)
        consensus_info[sid] = {
            "Q": Q_cons_all,
            "A": anisotropy_ratio(Q_cons_all),
            "ell_geom": geometric_correlation_length(Q_cons_all),
            "phi": float(seed_data[0]["phi_sub"]),
        }

        # Leave-one-seed-out high-N target for every seed.
        loo_targets = []
        for s in range(n_seeds):
            others = [Qmax[j] for j in range(n_seeds) if j != s]
            loo_targets.append(consensus_Q(others))

        for d in seed_data:
            s = int(d["seed_index"])
            target = loo_targets[s]
            A_target = anisotropy_ratio(target)
            ellg_target = geometric_correlation_length(target)

            for npts in point_counts:
                pr = d["point_results"][str(npts)]
                Q = np.asarray(pr["Q"], dtype=float)
                A = float(pr["A"])
                ellg = float(pr["ell_geom"])

                run_rows.append({
                    "subvolume": sid,
                    "center0": int(meta["center"][0]),
                    "center1": int(meta["center"][1]),
                    "center2": int(meta["center"][2]),
                    "phi_sub": float(d["phi_sub"]),
                    "seed_index": s,
                    "random_seed": int(d["random_seed"]),
                    "n_points": int(npts),
                    "valid_directions": int(pr["valid_directions"]),
                    "total_directions": int(pr["total_directions"]),
                    "rank": int(pr["rank"]),
                    "condition": float(pr["condition"]),
                    "eta": float(pr["eta"]),
                    "A": A,
                    "A_loo_target": A_target,
                    "abs_dA_loo": abs(A - A_target),
                    "ell_geom": ellg,
                    "ell_geom_loo_target": ellg_target,
                    "rel_ell_geom_error": abs(ellg - ellg_target) / ellg_target,
                    "E_loo": tensor_shape_error(Q, target),
                    "Q00": float(Q[0, 0]),
                    "Q11": float(Q[1, 1]),
                    "Q22": float(Q[2, 2]),
                    "Q01": float(Q[0, 1]),
                    "Q02": float(Q[0, 2]),
                    "Q12": float(Q[1, 2]),
                    "seed_elapsed_sec_max_run": float(d["elapsed_sec"]),
                })

        # Pairwise seed-to-seed differences at each point count.
        for npts in point_counts:
            Qn = [
                np.asarray(d["point_results"][str(npts)]["Q"], dtype=float)
                for d in seed_data
            ]
            for i in range(n_seeds):
                for j in range(i + 1, n_seeds):
                    pair_rows.append({
                        "subvolume": sid,
                        "n_points": int(npts),
                        "seed_i": i,
                        "seed_j": j,
                        "pair_tensor_error": tensor_pair_error(Qn[i], Qn[j]),
                    })

        # Subvolume summaries.
        for npts in point_counts:
            rr = [r for r in run_rows if r["subvolume"] == sid and r["n_points"] == npts]
            pp = [r for r in pair_rows if r["subvolume"] == sid and r["n_points"] == npts]

            E = percentile_summary([r["E_loo"] for r in rr])
            PE = percentile_summary([r["pair_tensor_error"] for r in pp])
            dA = percentile_summary([r["abs_dA_loo"] for r in rr])
            dL = percentile_summary([r["rel_ell_geom_error"] for r in rr])
            eta = percentile_summary([r["eta"] for r in rr])
            Avec = percentile_summary([r["A"] for r in rr])

            subvol_summary_rows.append({
                "subvolume": sid,
                "n_points": int(npts),
                "phi_sub": consensus_info[sid]["phi"],
                "A_consensus_50k": consensus_info[sid]["A"],
                "ell_geom_consensus_50k": consensus_info[sid]["ell_geom"],
                "E_median": E["median"],
                "E_q1": E["q1"],
                "E_q3": E["q3"],
                "E_p95": E["p95"],
                "pair_median": PE["median"],
                "pair_q1": PE["q1"],
                "pair_q3": PE["q3"],
                "pair_p95": PE["p95"],
                "abs_dA_median": dA["median"],
                "abs_dA_p95": dA["p95"],
                "rel_ell_geom_median": dL["median"],
                "rel_ell_geom_p95": dL["p95"],
                "eta_median": eta["median"],
                "eta_q1": eta["q1"],
                "eta_q3": eta["q3"],
                "A_median": Avec["median"],
                "A_q1": Avec["q1"],
                "A_q3": Avec["q3"],
            })

    # Global summaries pooled across the five selected subvolumes.
    global_summary_rows = []
    for npts in point_counts:
        rr = [r for r in run_rows if r["n_points"] == npts]
        pp = [r for r in pair_rows if r["n_points"] == npts]

        E = percentile_summary([r["E_loo"] for r in rr])
        PE = percentile_summary([r["pair_tensor_error"] for r in pp])
        dA = percentile_summary([r["abs_dA_loo"] for r in rr])
        dL = percentile_summary([r["rel_ell_geom_error"] for r in rr])
        eta = percentile_summary([r["eta"] for r in rr])

        # Worst subvolume median is useful for a conservative recommendation.
        sv_medians = [
            r["E_median"] for r in subvol_summary_rows if r["n_points"] == npts
        ]
        sv_p95 = [
            r["E_p95"] for r in subvol_summary_rows if r["n_points"] == npts
        ]

        global_summary_rows.append({
            "n_points": int(npts),
            "E_median": E["median"],
            "E_q1": E["q1"],
            "E_q3": E["q3"],
            "E_p90": E["p90"],
            "E_p95": E["p95"],
            "pair_median": PE["median"],
            "pair_q1": PE["q1"],
            "pair_q3": PE["q3"],
            "pair_p95": PE["p95"],
            "abs_dA_median": dA["median"],
            "abs_dA_p95": dA["p95"],
            "rel_ell_geom_median": dL["median"],
            "rel_ell_geom_p95": dL["p95"],
            "eta_median": eta["median"],
            "eta_q1": eta["q1"],
            "eta_q3": eta["q3"],
            "max_subvolume_E_median": float(max(sv_medians)),
            "max_subvolume_E_p95": float(max(sv_p95)),
        })

    return (
        run_rows,
        pair_rows,
        subvol_summary_rows,
        global_summary_rows,
        consensus_info,
    )


# =============================================================================
# CSV / REPORT
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def recommended_point_count(global_summary_rows, threshold):
    """Minimum N whose pooled 95th percentile is below a requested threshold."""
    for row in sorted(global_summary_rows, key=lambda r: r["n_points"]):
        if row["E_p95"] <= threshold:
            return int(row["n_points"])
    return None


def write_report(
    point_counts,
    n_seeds,
    all_seed_data,
    subvol_summary_rows,
    global_summary_rows,
    consensus_info,
):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Bentheimer real-rock validation: test 4\n")
        f.write("3D reference-estimator Monte Carlo convergence\n")
        f.write("=" * 78 + "\n\n")

        f.write(f"RAW file: {RAW_FILE}\n")
        f.write(f"RAW shape: {RAW_SHAPE}\n")
        f.write("RAW convention: 0=pore, 1=solid\n")
        f.write(f"Subvolume shape: {SUBVOL_N}^3\n")
        f.write(f"MAX_LAG: {MAX_LAG}\n")
        f.write(f"Fibonacci directions requested: {N_REF_DIRECTIONS}\n")
        f.write(f"Actual directions incl. axes: {len(fibonacci_hemisphere(N_REF_DIRECTIONS))}\n")
        f.write(f"Point counts: {point_counts}\n")
        f.write(f"Independent seeds/subvolume: {n_seeds}\n\n")

        f.write("Scientific interpretation\n")
        f.write("-" * 78 + "\n")
        f.write(
            "E_LOO is a determinant-normalized tensor-shape error relative to a "
            "leave-one-seed-out consensus constructed from the other seeds at the "
            "largest point count.\n"
        )
        f.write(
            "Pairwise error is a symmetric seed-to-seed tensor-shape separation and "
            "does not rely on a consensus target.\n"
        )
        f.write(
            "This experiment quantifies Monte Carlo uncertainty of the current "
            "reference estimator; it does not prove estimator unbiasedness.\n\n"
        )

        f.write("Selected subvolumes / 50k consensus\n")
        f.write("-" * 78 + "\n")
        for meta in SELECTED_SUBVOLUMES:
            sid = meta["id"]
            ci = consensus_info[sid]
            elapsed = [d["elapsed_sec"] for d in all_seed_data[sid]]
            f.write(
                f"{sid}: center={meta['center']}  phi={100*ci['phi']:.3f}%  "
                f"A_consensus={ci['A']:.5f}  ell_geom={ci['ell_geom']:.5f} px  "
                f"median max-run time={np.median(elapsed):.2f}s\n"
            )
        f.write("\n")

        f.write("GLOBAL CONVERGENCE SUMMARY\n")
        f.write("-" * 78 + "\n")
        for row in global_summary_rows:
            f.write(
                f"N={row['n_points']:6d}  "
                f"E_LOO median={100*row['E_median']:6.3f}%  "
                f"IQR=[{100*row['E_q1']:.3f},{100*row['E_q3']:.3f}]%  "
                f"p95={100*row['E_p95']:.3f}%  "
                f"pair median={100*row['pair_median']:.3f}%  "
                f"pair p95={100*row['pair_p95']:.3f}%  "
                f"|dA| med={row['abs_dA_median']:.5f}  "
                f"geom med={100*row['rel_ell_geom_median']:.3f}%  "
                f"eta med={100*row['eta_median']:.3f}%\n"
            )
        f.write("\n")

        f.write("SUBVOLUME CONVERGENCE AT EACH POINT COUNT\n")
        f.write("-" * 78 + "\n")
        for sid in [m["id"] for m in SELECTED_SUBVOLUMES]:
            f.write(f"\n{sid}\n")
            for row in subvol_summary_rows:
                if row["subvolume"] == sid:
                    f.write(
                        f"  N={row['n_points']:6d}  "
                        f"E med={100*row['E_median']:.3f}%  "
                        f"p95={100*row['E_p95']:.3f}%  "
                        f"pair med={100*row['pair_median']:.3f}%  "
                        f"|dA| med={row['abs_dA_median']:.5f}\n"
                    )
        f.write("\n")

        f.write("THRESHOLD CROSSINGS (pooled E_LOO p95)\n")
        f.write("-" * 78 + "\n")
        for threshold in CONVERGENCE_THRESHOLDS:
            rec = recommended_point_count(global_summary_rows, threshold)
            if rec is None:
                f.write(
                    f"p95 <= {100*threshold:.1f}%: not reached by {max(point_counts):,} points/direction\n"
                )
            else:
                f.write(
                    f"p95 <= {100*threshold:.1f}%: first reached at {rec:,} points/direction\n"
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


def boxplot_no_fliers(ax, data, positions, widths=0.18):
    return ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        showfliers=False,
        patch_artist=False,
        medianprops=dict(linewidth=1.7),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )


def build_figure(
    point_counts,
    run_rows,
    pair_rows,
    subvol_summary_rows,
    global_summary_rows,
):
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)

    x = np.arange(len(point_counts), dtype=float)
    xlabels = [f"{n//1000}k" for n in point_counts]

    # ------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, 0])
    markers = ["o", "s", "^", "D", "v"]
    for marker, meta in zip(markers, SELECTED_SUBVOLUMES):
        sid = meta["id"]
        rows = [
            r for r in subvol_summary_rows if r["subvolume"] == sid
        ]
        rows = sorted(rows, key=lambda r: r["n_points"])
        med = 100*np.array([r["E_median"] for r in rows])
        q1 = 100*np.array([r["E_q1"] for r in rows])
        q3 = 100*np.array([r["E_q3"] for r in rows])
        ax.plot(x, med, marker=marker, label=sid)
        ax.fill_between(x, q1, q3, alpha=0.10)
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Monte Carlo points / direction")
    ax.set_ylabel(r"LOO tensor-shape error $E_{Q}$ (%)")
    ax.set_title("Convergence across real subvolumes")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.18)
    panel_label(ax, "(a)")

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    data = [
        100*np.array([r["E_loo"] for r in run_rows if r["n_points"] == n])
        for n in point_counts
    ]
    boxplot_no_fliers(ax, data, x, widths=0.45)
    p95 = [100*r["E_p95"] for r in global_summary_rows]
    ax.plot(x, p95, "o--", label="pooled 95th percentile")
    ax.axhline(2.0, linestyle=":", linewidth=1.2, label="2% diagnostic target")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Monte Carlo points / direction")
    ax.set_ylabel(r"LOO tensor-shape error $E_{Q}$ (%)")
    ax.set_title("Pooled reference uncertainty")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    data = [
        100*np.array([
            r["pair_tensor_error"] for r in pair_rows if r["n_points"] == n
        ])
        for n in point_counts
    ]
    boxplot_no_fliers(ax, data, x, widths=0.45)
    med_pair = [100*r["pair_median"] for r in global_summary_rows]
    ax.plot(x, med_pair, "o--", label="pairwise median")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Monte Carlo points / direction")
    ax.set_ylabel("Seed-to-seed tensor separation (%)")
    ax.set_title("Pairwise Monte Carlo dispersion")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0])
    data = [
        np.array([r["abs_dA_loo"] for r in run_rows if r["n_points"] == n])
        for n in point_counts
    ]
    boxplot_no_fliers(ax, data, x, widths=0.45)
    med = [r["abs_dA_median"] for r in global_summary_rows]
    ax.plot(x, med, "o--")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Monte Carlo points / direction")
    ax.set_ylabel(r"$|A-A_{LOO}|$")
    ax.set_title("Anisotropy-ratio stability")
    ax.grid(alpha=0.18)
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    data = [
        100*np.array([
            r["rel_ell_geom_error"] for r in run_rows if r["n_points"] == n
        ])
        for n in point_counts
    ]
    boxplot_no_fliers(ax, data, x, widths=0.45)
    med = [100*r["rel_ell_geom_median"] for r in global_summary_rows]
    ax.plot(x, med, "o--")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Monte Carlo points / direction")
    ax.set_ylabel("Geometric correlation-scale error (%)")
    ax.set_title("Absolute scale stability")
    ax.grid(alpha=0.18)
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    eta_med = 100*np.array([r["eta_median"] for r in global_summary_rows])
    eta_q1 = 100*np.array([r["eta_q1"] for r in global_summary_rows])
    eta_q3 = 100*np.array([r["eta_q3"] for r in global_summary_rows])
    ax.plot(x, eta_med, "o-", label="median tensor-fit residual")
    ax.fill_between(x, eta_q1, eta_q3, alpha=0.14, label="IQR")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Monte Carlo points / direction")
    ax.set_ylabel(r"Reference tensor residual $\eta_Q$ (%)")
    ax.set_title("Model residual vs sampling density")
    ax.grid(alpha=0.18)

    rec2 = recommended_point_count(global_summary_rows, 0.02)
    if rec2 is None:
        txt = f"p95(E_Q) < 2% not reached\nby {max(point_counts)//1000}k points/dir"
    else:
        txt = f"p95(E_Q) < 2% first reached\nat {rec2//1000}k points/dir"
    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )
    panel_label(ax, "(f)")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.96)
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

    point_counts, n_seeds = active_settings()
    signature = cache_signature(point_counts, n_seeds)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("REFERENCE-CONVERGENCE DESIGN")
    print("=" * 78)
    print(f"Selected subvolumes:   {len(SELECTED_SUBVOLUMES)}")
    print(f"Subvolume size:        {SUBVOL_N}^3")
    print(f"Directions:            {len(fibonacci_hemisphere(N_REF_DIRECTIONS))}")
    print(f"Point counts:          {point_counts}")
    print(f"Independent seeds:     {n_seeds} / subvolume")
    print(f"Largest point count:   {max(point_counts):,} / direction")
    print("Nested counts/seed:    YES")
    print("Consensus target:      leave-one-seed-out at N_max")
    print()

    raw = np.memmap(
        RAW_FILE,
        dtype=RAW_DTYPE,
        mode="r",
        shape=RAW_SHAPE,
        order="C",
    )

    all_seed_data = {}
    total_jobs = len(SELECTED_SUBVOLUMES) * n_seeds
    completed_jobs = 0
    t_all = time.time()

    for order, meta0 in enumerate(SELECTED_SUBVOLUMES):
        meta = dict(meta0)
        meta["order"] = order
        sid = meta["id"]
        all_seed_data[sid] = []

        # Check whether any seed needs computation before loading 512^3 into RAM.
        missing = []
        for seed_index in range(n_seeds):
            p = cache_path(sid, seed_index)
            cached = load_seed_cache(p, signature)
            if cached is None:
                missing.append(seed_index)
            else:
                all_seed_data[sid].append(cached)
                completed_jobs += 1

        if not missing:
            print(f"[{sid}] all {n_seeds} seeds loaded from cache.\n")
            continue

        phase, starts, stops, phi_sub = extract_subvolume(
            raw, meta["center"], sid
        )
        meta["starts"] = starts
        meta["stops"] = stops

        for seed_index in missing:
            job_no = completed_jobs + 1
            print(
                f"[{job_no:02d}/{total_jobs:02d}] {sid}  "
                f"seed={seed_index:02d}"
            )
            result = compute_seed_result(
                phase,
                meta,
                seed_index,
                point_counts,
                signature,
            )
            result["subvolume"]["starts"] = list(starts)
            result["subvolume"]["stops"] = list(stops)
            save_json_atomic(cache_path(sid, seed_index), result)
            all_seed_data[sid].append(result)
            completed_jobs += 1
            print(
                f"  completed in {result['elapsed_sec']:.1f}s  "
                f"cached -> {cache_path(sid, seed_index).name}\n"
            )

        del phase
        gc.collect()

    del raw
    gc.collect()

    # Ensure cache-loaded and newly computed data are ordered and complete.
    for meta in SELECTED_SUBVOLUMES:
        sid = meta["id"]
        all_seed_data[sid] = sorted(
            all_seed_data[sid], key=lambda d: d["seed_index"]
        )
        if len(all_seed_data[sid]) != n_seeds:
            raise RuntimeError(
                f"Incomplete result set for {sid}: "
                f"{len(all_seed_data[sid])}/{n_seeds} seeds."
            )

    (
        run_rows,
        pair_rows,
        subvol_summary_rows,
        global_summary_rows,
        consensus_info,
    ) = build_analysis(all_seed_data, point_counts, n_seeds)

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("=" * 78)
    print("REFERENCE-CONVERGENCE SUMMARY")
    print("=" * 78)

    for meta in SELECTED_SUBVOLUMES:
        sid = meta["id"]
        ci = consensus_info[sid]
        print(
            f"{sid}: phi={100*ci['phi']:.3f}%  "
            f"A_50k_consensus={ci['A']:.4f}  "
            f"ell_geom={ci['ell_geom']:.3f} px"
        )
    print()

    for row in global_summary_rows:
        print(
            f"N={row['n_points']:6d}  "
            f"E_LOO median={100*row['E_median']:6.3f}%  "
            f"IQR=[{100*row['E_q1']:.3f},{100*row['E_q3']:.3f}]%  "
            f"p95={100*row['E_p95']:.3f}%  "
            f"pair median={100*row['pair_median']:.3f}%  "
            f"pair p95={100*row['pair_p95']:.3f}%  "
            f"|dA| med={row['abs_dA_median']:.4f}  "
            f"geom med={100*row['rel_ell_geom_median']:.3f}%  "
            f"eta med={100*row['eta_median']:.3f}%"
        )

    print()
    print("DIAGNOSTIC THRESHOLD CROSSINGS (pooled E_LOO p95)")
    for threshold in CONVERGENCE_THRESHOLDS:
        rec = recommended_point_count(global_summary_rows, threshold)
        if rec is None:
            print(
                f"  <= {100*threshold:.1f}% : not reached by "
                f"{max(point_counts):,} points/direction"
            )
        else:
            print(
                f"  <= {100*threshold:.1f}% : first reached at "
                f"{rec:,} points/direction"
            )

    total_elapsed = time.time() - t_all
    print(f"\nTotal wall time this invocation: {total_elapsed:.1f} s")
    print()

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    write_csv(RUNS_CSV_PATH, run_rows)
    write_csv(PAIRWISE_CSV_PATH, pair_rows)
    write_csv(SUBVOL_SUMMARY_CSV_PATH, subvol_summary_rows)
    write_csv(GLOBAL_SUMMARY_CSV_PATH, global_summary_rows)

    write_report(
        point_counts,
        n_seeds,
        all_seed_data,
        subvol_summary_rows,
        global_summary_rows,
        consensus_info,
    )

    build_figure(
        point_counts,
        run_rows,
        pair_rows,
        subvol_summary_rows,
        global_summary_rows,
    )

    print("=" * 78)
    print("FILES GENERATED")
    print("=" * 78)
    print(PNG_PATH)
    print(SVG_PATH)
    print(REPORT_PATH)
    print(RUNS_CSV_PATH)
    print(SUBVOL_SUMMARY_CSV_PATH)
    print(GLOBAL_SUMMARY_CSV_PATH)
    print(PAIRWISE_CSV_PATH)
    print(CACHE_DIR)
    print()
    print("Bentheimer test 4 completed successfully.")


if __name__ == "__main__":
    main()