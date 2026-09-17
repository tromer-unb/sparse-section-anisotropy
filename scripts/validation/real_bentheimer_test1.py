#!/usr/bin/env python3
"""
First real-rock validation test for the Bentheimer (Kocurek 15A) dataset.

Goal
----
Compare a 3D pore-correlation tensor estimated directly from a complete
central subvolume with a tensor reconstructed using only three orthogonal
2D sections from the same subvolume.

IMPORTANT SCIENTIFIC TERMINOLOGY
--------------------------------
For a real rock there is no analytic "ground-truth tensor".  Therefore this
script calls the full-3D estimate Q3D_REF (reference tensor), not Q3D_GT.
The sparse estimate Q3D_SPARSE is reconstructed only from the three 2D
sections.  The comparison is therefore:

    full 3D subvolume -> Q3D_REF
    three 2D sections -> Q3D_SPARSE

The RAW dataset convention validated previously is:
    RAW value 0 = pore
    RAW value 1 = solid

The original 2500^3 RAW file is opened with numpy.memmap; it is NOT loaded
entirely into RAM.  Only the selected central subvolume is copied to RAM.

Outputs
-------
  bentheimer_test1.png
  bentheimer_test1.svg
  bentheimer_test1_report.txt
  bentheimer_test1_measurements.csv

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

Run
---
  python3 real_bentheimer_test1.py

For a faster diagnostic run, set QUICK_MODE = True below.
"""

from pathlib import Path
import csv
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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

# Central real-rock subvolume used in this first validation.
SUBVOL_N = 512

# Directional-correlation settings.
MAX_LAG = 48
THETA_STEP_2D_DEG = 5.0

# Full-3D reference estimator: number of directions and stochastic sample
# points used per direction.  These settings do not allocate a 3D FFT.
N_REF_DIRECTIONS = 64
N_REF_POINTS = 12_000

# Reproducibility.
SEED = 20260907

# Fast first-pass option.  It changes only the 3D reference sampling density.
QUICK_MODE = False

# Publication-quality raster output.
DPI = 800


# =============================================================================
# OUTPUTS / STYLE
# =============================================================================

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "bentheimer_test1.png"
SVG_PATH = OUTDIR / "bentheimer_test1.svg"
REPORT_PATH = OUTDIR / "bentheimer_test1_report.txt"
CSV_PATH = OUTDIR / "bentheimer_test1_measurements.csv"

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
    """
    v: shape (N, 3), unit ambient directions.
    Returns rows for v^T Q v with symmetric Q parameterized by 6 numbers.
    """
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


def fit_q3d_from_directional_lengths(vectors, ell):
    """Unweighted least squares in y=1/ell^2, followed by SPD projection."""
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

    # Finite-sample noise can make unconstrained LS very slightly indefinite.
    Q = 0.5 * (Q + Q.T)
    w, E = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-7 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = E @ np.diag(w) @ E.T

    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1])

    ell_pred = ell_from_q3d(Q, V)
    eta = float(
        np.sqrt(np.sum((L - ell_pred)**2) / np.sum(L**2))
    )

    return Q, rank, cond, eta


def ell_from_q3d(Q, vectors):
    vectors = np.asarray(vectors, dtype=float)
    val = np.einsum("ni,ij,nj->n", vectors, Q, vectors)
    return 1.0 / np.sqrt(val)


def principal_lengths_and_axes(Q):
    """Return principal correlation lengths from largest to smallest."""
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
    z = (i + 0.5) / n                 # 0 < z < 1
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    phi = golden * i

    dirs = np.column_stack([
        r * np.cos(phi),
        r * np.sin(phi),
        z,
    ])

    # Add the three computational axes; duplicates are harmless, but we remove
    # nearly duplicate rows for a cleaner design.
    dirs = np.vstack([dirs, np.eye(3)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Unique up to rounding.  u and -u are equivalent for this quadratic model;
    # all generated directions already belong to one hemisphere.
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
    Estimate C(u,r) directly from the complete 3D subvolume without a 3D FFT.

    The same random integer base points are used for all directions.  Base
    points are kept at least max_lag+2 voxels from every boundary, so every
    displaced sample remains inside the subvolume.  Trilinear interpolation is
    used for off-lattice displaced locations.

    RAW convention: phase=0 pore, phase=1 solid.
    """
    rng = np.random.default_rng(seed)
    n = phase.shape[0]
    margin = int(np.ceil(max_lag)) + 2

    if 2 * margin >= n:
        raise ValueError("MAX_LAG is too large for this subvolume.")

    # Exact porosity of the selected subvolume.
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

    # At integer base points: pore indicator chi = 1 - phase.
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
        # Shape before flattening: (3, n_points, n_lags)
        coords = (
            base[:, :, None].astype(np.float32)
            + u[:, None, None].astype(np.float32) * lags[None, None, :]
        )

        # Interpolate the stored solid indicator and convert to pore indicator.
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


def measure_section_2d(phase_section, B, theta_step_deg, max_lag, name):
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
        "name": name,
        "phase": phase_section,
        "pore": pore,
        "B": B,
        "theta": theta,
        "vectors": v,
        "ell": ell,
        "phi": phi,
    }


def central_orthogonal_sections(phase):
    """
    Computational array-coordinate sections.

    We intentionally call the axes a0/a1/a2, not geological X/Y/Z, because the
    physical axis mapping of the RAW file should be confirmed from metadata.
    """
    c = phase.shape[0] // 2

    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    definitions = [
        ("a0-a1", phase[:, :, c], np.column_stack([e0, e1])),
        ("a0-a2", phase[:, c, :], np.column_stack([e0, e2])),
        ("a1-a2", phase[c, :, :], np.column_stack([e1, e2])),
    ]

    results = []
    for name, img, B in definitions:
        results.append(
            measure_section_2d(
                img,
                B,
                theta_step_deg=THETA_STEP_2D_DEG,
                max_lag=MAX_LAG,
                name=name,
            )
        )
    return results


def reconstruct_sparse_q3d(section_results):
    vectors = []
    ell = []

    for s in section_results:
        good = np.isfinite(s["ell"]) & (s["ell"] > 0)
        vectors.append(s["vectors"][good])
        ell.append(s["ell"][good])

    vectors = np.vstack(vectors)
    ell = np.concatenate(ell)

    Q, rank, cond, eta = fit_q3d_from_directional_lengths(vectors, ell)
    return Q, rank, cond, eta, vectors, ell


# =============================================================================
# OUTPUT TABLE / REPORT
# =============================================================================

def write_measurements_csv(ref_dirs, ref_ell, sections):
    fields = [
        "source",
        "theta_deg",
        "v0",
        "v1",
        "v2",
        "ell_pixels",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for v, ell in zip(ref_dirs, ref_ell):
            writer.writerow({
                "source": "3D_reference",
                "theta_deg": "",
                "v0": f"{v[0]:.10g}",
                "v1": f"{v[1]:.10g}",
                "v2": f"{v[2]:.10g}",
                "ell_pixels": f"{ell:.10g}",
            })

        for s in sections:
            for th, v, ell in zip(s["theta"], s["vectors"], s["ell"]):
                writer.writerow({
                    "source": s["name"],
                    "theta_deg": f"{th:.10g}",
                    "v0": f"{v[0]:.10g}",
                    "v1": f"{v[1]:.10g}",
                    "v2": f"{v[2]:.10g}",
                    "ell_pixels": f"{ell:.10g}",
                })


def write_report(
    starts, stops, phi_sub,
    Q_ref, Q_sparse,
    eta_ref, eta_sparse,
    rank_ref, rank_sparse,
    cond_ref, cond_sparse,
    E_Q, A_ref, A_sparse,
    ell_ref_n, ell_sparse_n,
    axis_errors,
    section_results,
):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Bentheimer real-rock tensor validation: test 1\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"RAW file: {RAW_FILE}\n")
        f.write(f"RAW shape: {RAW_SHAPE}\n")
        f.write("RAW convention: 0=pore, 1=solid\n")
        f.write("Axis names a0/a1/a2 are computational array axes.\n\n")

        f.write(f"Subvolume shape: {(SUBVOL_N, SUBVOL_N, SUBVOL_N)}\n")
        f.write(f"Subvolume start: {starts}\n")
        f.write(f"Subvolume stop:  {stops}\n")
        f.write(f"Exact subvolume porosity: {phi_sub:.8f}\n\n")

        f.write("Reference tensor Q3D_REF (full 3D subvolume)\n")
        f.write(np.array2string(Q_ref, precision=8) + "\n")
        f.write(f"rank={rank_ref}, condition={cond_ref:.6g}, eta={eta_ref:.8f}\n\n")

        f.write("Sparse tensor Q3D_SPARSE (three central 2D sections only)\n")
        f.write(np.array2string(Q_sparse, precision=8) + "\n")
        f.write(f"rank={rank_sparse}, condition={cond_sparse:.6g}, eta={eta_sparse:.8f}\n\n")

        f.write("Comparison (determinant-normalized tensor shape)\n")
        f.write(f"E_Q={E_Q:.8f} ({100*E_Q:.3f}%)\n")
        f.write(f"A_ref={A_ref:.8f}\n")
        f.write(f"A_sparse={A_sparse:.8f}\n")
        f.write("Normalized principal lengths, reference: " + np.array2string(ell_ref_n, precision=6) + "\n")
        f.write("Normalized principal lengths, sparse:    " + np.array2string(ell_sparse_n, precision=6) + "\n")
        f.write("Principal-axis errors (deg): " + np.array2string(axis_errors, precision=4) + "\n")
        f.write("Orientation errors should be interpreted cautiously for near-isotropic / nearly degenerate tensors.\n\n")

        f.write("Section porosities\n")
        for s in section_results:
            f.write(f"  {s['name']}: phi={s['phi']:.8f}\n")


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
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )


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


def build_figure(
    section_results,
    ref_dirs,
    ref_ell,
    Q_ref,
    sparse_vectors,
    sparse_ell,
    Q_sparse,
    phi_sub,
    E_Q,
    eta_ref,
    eta_sparse,
    A_ref,
    A_sparse,
    ell_ref_n,
    ell_sparse_n,
    axis_errors,
):
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)

    # ------------------------------------------------------------------ (a)
    sub = gs[0, 0].subgridspec(1, 3, wspace=0.05)
    axes_a = []
    for i, s in enumerate(section_results):
        ax = fig.add_subplot(sub[0, i])
        ax.imshow(s["pore"].T, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(s["name"], fontsize=9.2, pad=3, fontweight="bold")
        axes_a.append(ax)
    axes_a[1].text(
        0.5, 1.16,
        "Central real-rock sections",
        transform=axes_a[1].transAxes,
        ha="center", va="bottom",
        fontsize=11.2, fontweight="bold",
    )
    panel_label(axes_a[0], "(a)", x=-0.05, y=1.08)

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    good = np.isfinite(ref_ell) & (ref_ell > 0)
    pred = ell_from_q3d(Q_ref, ref_dirs[good])
    meas = ref_ell[good]
    lo = 0.95 * min(meas.min(), pred.min())
    hi = 1.05 * max(meas.max(), pred.max())
    ax.scatter(meas, pred, s=22, alpha=0.78)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured 3D correlation length (px)")
    ax.set_ylabel("Tensor-predicted length (px)")
    ax.set_title("Full-3D reference tensor fit")
    ax.text(
        0.96, 0.06,
        f"fit residual = {100*eta_ref:.1f}%",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.6, fontweight="bold",
    )
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    good = np.isfinite(sparse_ell) & (sparse_ell > 0)
    pred = ell_from_q3d(Q_sparse, sparse_vectors[good])
    meas = sparse_ell[good]
    lo = 0.95 * min(meas.min(), pred.min())
    hi = 1.05 * max(meas.max(), pred.max())
    ax.scatter(meas, pred, s=20, alpha=0.72)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured section length (px)")
    ax.set_ylabel("3D tensor-predicted length (px)")
    ax.set_title("Sparse-section tensor fit")
    ax.text(
        0.96, 0.06,
        f"fit residual = {100*eta_sparse:.1f}%",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.6, fontweight="bold",
    )
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0], projection="3d")
    XR, YR, ZR = ellipsoid_mesh_from_Q(Q_ref)
    XS, YS, ZS = ellipsoid_mesh_from_Q(Q_sparse)

    ax.plot_wireframe(
        XR, YR, ZR,
        rstride=2, cstride=3,
        linewidth=0.85, color="black", alpha=0.95,
    )
    ax.plot_wireframe(
        XS, YS, ZS,
        rstride=3, cstride=4,
        linewidth=0.75, color="0.55", alpha=0.95,
    )

    lim = 1.20 * max(
        np.max(np.abs(XR)), np.max(np.abs(YR)), np.max(np.abs(ZR)),
        np.max(np.abs(XS)), np.max(np.abs(YS)), np.max(np.abs(ZS)),
    )
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-52)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("a0", labelpad=-6)
    ax.set_ylabel("a1", labelpad=-6)
    ax.set_zlabel("a2", labelpad=-4)
    ax.set_title("Reference vs sparse 3D tensor")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        except Exception:
            pass

    proxies = [
        Line2D([0], [0], color="black", lw=1.5, label="Full 3D reference"),
        Line2D([0], [0], color="0.55", lw=1.5, label="Three 2D sections"),
    ]
    ax.legend(handles=proxies, frameon=False, loc="lower left")
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    idx = np.arange(1, 4)
    ax.plot(idx, ell_ref_n, marker="o", label="Full 3D reference")
    ax.plot(idx, ell_sparse_n, marker="s", label="Three 2D sections")
    ax.set_xticks(idx, [r"$\ell_1$", r"$\ell_2$", r"$\ell_3$"])
    ax.set_xlabel("Principal direction")
    ax.set_ylabel("Det-normalized correlation length")
    ax.set_title("Principal correlation scales")
    ax.legend(frameon=False)
    ax.text(
        0.04, 0.06,
        "Axis errors: " + ", ".join(f"{v:.1f}°" for v in axis_errors),
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8.3, fontweight="bold",
    )
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("Real-rock validation summary", pad=8)

    lines = [
        f"Subvolume: {SUBVOL_N}³ voxels",
        f"Porosity: {100*phi_sub:.2f}%",
        "",
        f"Reference anisotropy: {A_ref:.3f}",
        f"Sparse anisotropy: {A_sparse:.3f}",
        f"Tensor-shape error: {100*E_Q:.2f}%",
        "",
        f"3D model residual: {100*eta_ref:.2f}%",
        f"Sparse model residual: {100*eta_sparse:.2f}%",
        "",
        "Axes are RAW array coordinates",
        "until metadata mapping is confirmed.",
    ]

    ax.text(
        0.05, 0.90,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.0,
        fontweight="bold",
        linespacing=1.45,
    )
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
    # 1) Full-3D reference tensor from the complete selected subvolume
    # ------------------------------------------------------------------
    n_dirs = 32 if QUICK_MODE else N_REF_DIRECTIONS
    n_pts = 4_000 if QUICK_MODE else N_REF_POINTS

    ref_dirs = fibonacci_hemisphere(n_dirs)
    ref_r, ref_C, ref_ell, phi_check = directional_correlations_3d(
        phase,
        ref_dirs,
        max_lag=MAX_LAG,
        n_points=n_pts,
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

    Q_ref, rank_ref, cond_ref, eta_ref = fit_q3d_from_directional_lengths(
        ref_dirs,
        ref_ell,
    )

    # ------------------------------------------------------------------
    # 2) Sparse tensor using ONLY three central orthogonal 2D sections
    # ------------------------------------------------------------------
    print("=" * 78)
    print("THREE-SECTION SPARSE RECONSTRUCTION")
    print("=" * 78)

    sections = central_orthogonal_sections(phase)
    for s in sections:
        n_valid = np.sum(np.isfinite(s["ell"]))
        print(
            f"{s['name']:6s}  phi={s['phi']:.6f}  "
            f"valid lengths={n_valid}/{len(s['ell'])}"
        )
    print()

    Q_sparse, rank_sparse, cond_sparse, eta_sparse, sparse_vectors, sparse_ell = (
        reconstruct_sparse_q3d(sections)
    )

    # ------------------------------------------------------------------
    # 3) Compare determinant-normalized tensor shapes
    # ------------------------------------------------------------------
    Q_ref_n = normalize_det_spd(Q_ref)
    Q_sparse_n = normalize_det_spd(Q_sparse)

    E_Q = tensor_shape_error(Q_sparse, Q_ref)
    A_ref = anisotropy_ratio(Q_ref)
    A_sparse = anisotropy_ratio(Q_sparse)

    ell_ref_n, axes_ref = principal_lengths_and_axes(Q_ref_n)
    ell_sparse_n, axes_sparse = principal_lengths_and_axes(Q_sparse_n)

    axis_errors = np.array([
        axial_angle_error_deg(axes_ref[:, i], axes_sparse[:, i])
        for i in range(3)
    ])

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print("Q3D_REF (full 3D subvolume):")
    print(Q_ref)
    print()
    print("Q3D_SPARSE (three 2D sections only):")
    print(Q_sparse)
    print()
    print(f"Reference rank / condition: {rank_ref} / {cond_ref:.4f}")
    print(f"Sparse rank / condition:    {rank_sparse} / {cond_sparse:.4f}")
    print(f"Reference tensor residual:  {100*eta_ref:.3f}%")
    print(f"Sparse tensor residual:     {100*eta_sparse:.3f}%")
    print(f"Tensor-shape error E_Q:     {100*E_Q:.3f}%")
    print(f"A_ref:                      {A_ref:.4f}")
    print(f"A_sparse:                   {A_sparse:.4f}")
    print("Normalized principal lengths (ref):   ", np.round(ell_ref_n, 5))
    print("Normalized principal lengths (sparse):", np.round(ell_sparse_n, 5))
    print("Principal-axis errors (deg):           ", np.round(axis_errors, 3))
    if A_ref < 1.10:
        print(
            "NOTE: reference tensor is close to isotropic; principal-axis "
            "angles are then intrinsically unstable and should not be overinterpreted."
        )
    print()

    # ------------------------------------------------------------------
    # 4) Save data products
    # ------------------------------------------------------------------
    write_measurements_csv(ref_dirs, ref_ell, sections)

    write_report(
        starts, stops, phi_sub,
        Q_ref, Q_sparse,
        eta_ref, eta_sparse,
        rank_ref, rank_sparse,
        cond_ref, cond_sparse,
        E_Q, A_ref, A_sparse,
        ell_ref_n, ell_sparse_n,
        axis_errors,
        sections,
    )

    build_figure(
        sections,
        ref_dirs,
        ref_ell,
        Q_ref,
        sparse_vectors,
        sparse_ell,
        Q_sparse,
        phi_sub,
        E_Q,
        eta_ref,
        eta_sparse,
        A_ref,
        A_sparse,
        ell_ref_n,
        ell_sparse_n,
        axis_errors,
    )

    print("=" * 78)
    print("FILES GENERATED")
    print("=" * 78)
    print(PNG_PATH)
    print(SVG_PATH)
    print(REPORT_PATH)
    print(CSV_PATH)
    print()
    print("Bentheimer test 1 completed successfully.")


if __name__ == "__main__":
    main()