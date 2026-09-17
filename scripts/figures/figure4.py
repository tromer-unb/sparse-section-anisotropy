#!/usr/bin/env python3
"""
High-quality Figure 4 for the paper.

Scientific message
------------------
Sparse oriented 2D sections provide complementary restrictions of a common
3D pore-correlation tensor. One section is rank deficient, two sections are
still generically insufficient, whereas three appropriately oriented
sections identify the six independent tensor components. The same inversion
also reveals why poorly conditioned section geometries amplify error.

Generates
---------
  - figure4_highquality.png
  - figure4_highquality.svg
  - figure4_q3d_recovery_results.csv

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

This script is fully standalone. It does not depend on figure1.py,
figure2.py, figure3.py, PoreSpy, TACC data, or any project module.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.stats import spearmanr


# -----------------------------------------------------------------------------
# Settings -- matched to Figures 1--3
# -----------------------------------------------------------------------------
SEED = 73
N = 224
SECTION_N = 144
POROSITY = 0.27
MAX_LAG = 28
THETA_STEP_DEG = 5
DPI = 800

# Principal correlation scales of the underlying stationary Gaussian field.
# Only their ratios matter for the determinant-normalized ground-truth tensor.
PRINCIPAL_SCALES = np.array([7.0, 4.2, 2.3], dtype=float)

# A non-axis-aligned 3D principal frame.  The estimator never receives these
# Euler angles; they are used only to construct the controlled synthetic field.
EULER_DEG = (24.0, 37.0, 19.0)  # rotations about x, y, z

# Pool of independently oriented sections from the same volume.  These are
# measured once and then resampled to quantify the effect of section count and
# inverse-problem conditioning.
N_SECTION_POOL = 20
N_SUBSETS_PER_COUNT = 220
SECTION_COUNTS = (3, 4, 5, 6)

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "figure4_highquality.png"
SVG_PATH = OUTDIR / "figure4_highquality.svg"
CSV_PATH = OUTDIR / "figure4_q3d_recovery_results.csv"

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
    "svg.fonttype": "none",   # editable text in Inkscape
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Basic linear algebra
# -----------------------------------------------------------------------------
def rotation_matrix_xyz(ax_deg, ay_deg, az_deg):
    ax, ay, az = np.deg2rad([ax_deg, ay_deg, az_deg])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(ax), -np.sin(ax)],
        [0.0, np.sin(ax), np.cos(ax)],
    ])
    Ry = np.array([
        [np.cos(ay), 0.0, np.sin(ay)],
        [0.0, 1.0, 0.0],
        [-np.sin(ay), 0.0, np.cos(ay)],
    ])
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az), np.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return Rz @ Ry @ Rx


def normalize_det_spd(Q):
    """Project to SPD and normalize determinant to unity."""
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-10, 1e-8 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = V @ np.diag(w) @ V.T
    det = np.linalg.det(Q)
    return Q / det ** (1.0 / Q.shape[0])


def principal_lengths_and_axes(Q):
    """Return principal lengths (largest first) and corresponding axes."""
    w, V = np.linalg.eigh(Q)
    w = np.maximum(w, 1e-14)
    ell = 1.0 / np.sqrt(w)
    order = np.argsort(ell)[::-1]
    return ell[order], V[:, order]


def axial_angle_error_deg(v1, v2):
    c = np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))


# -----------------------------------------------------------------------------
# Controlled anisotropic 3D Gaussian random field
# -----------------------------------------------------------------------------
def generate_binary_medium_3d(n=N, porosity=POROSITY, seed=SEED):
    """
    Generate a stationary anisotropic Gaussian random field on a periodic cube
    using an ellipsoidal Gaussian spectral filter, then threshold to porosity.

    The principal-frame orientation and scale ratios are known by construction,
    providing a determinant-normalized tensor ground truth.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n, n))
    F = np.fft.rfftn(noise)

    kx = (2.0 * np.pi * np.fft.fftfreq(n))[:, None, None]
    ky = (2.0 * np.pi * np.fft.fftfreq(n))[None, :, None]
    kz = (2.0 * np.pi * np.fft.rfftfreq(n))[None, None, :]

    R = rotation_matrix_xyz(*EULER_DEG)

    # Principal-frame wave-vector components: k_p = R^T k.
    kp0 = R[0, 0] * kx + R[1, 0] * ky + R[2, 0] * kz
    kp1 = R[0, 1] * kx + R[1, 1] * ky + R[2, 1] * kz
    kp2 = R[0, 2] * kx + R[1, 2] * ky + R[2, 2] * kz

    L0, L1, L2 = PRINCIPAL_SCALES
    H = np.exp(-0.5 * ((L0 * kp0) ** 2 + (L1 * kp1) ** 2 + (L2 * kp2) ** 2))

    field = np.fft.irfftn(F * H, s=(n, n, n), axes=(0, 1, 2)).real
    threshold = np.quantile(field, 1.0 - porosity)
    binary = field >= threshold

    # Ground-truth *shape* tensor.  Thresholding changes a common scale factor
    # but preserves the ellipsoidal directional metric for a stationary GRF;
    # determinant normalization removes that unknown scalar factor.
    Q_true = R @ np.diag(1.0 / (PRINCIPAL_SCALES ** 2)) @ R.T
    Q_true = normalize_det_spd(Q_true)
    return binary, Q_true, R


# -----------------------------------------------------------------------------
# Oriented section geometry and extraction
# -----------------------------------------------------------------------------
def basis_from_normal(normal):
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)

    # Select a stable auxiliary direction.
    if abs(n[2]) < 0.85:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([0.0, 1.0, 0.0])

    b1 = np.cross(a, n)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1)
    b2 /= np.linalg.norm(b2)
    return np.column_stack([b1, b2])


def extract_oriented_section(volume, B, size=SECTION_N, origin=None, order=1):
    """Extract a square section with orthonormal ambient basis B (3x2)."""
    if origin is None:
        origin = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    else:
        origin = np.asarray(origin, dtype=float)

    u = np.arange(size, dtype=float) - (size - 1.0) / 2.0
    v = np.arange(size, dtype=float) - (size - 1.0) / 2.0
    U, V = np.meshgrid(u, v, indexing="ij")

    X = origin[0] + B[0, 0] * U + B[0, 1] * V
    Y = origin[1] + B[1, 0] * U + B[1, 1] * V
    Z = origin[2] + B[2, 0] * U + B[2, 1] * V

    valid = (
        (X >= 0) & (X <= volume.shape[0] - 1) &
        (Y >= 0) & (Y <= volume.shape[1] - 1) &
        (Z >= 0) & (Z <= volume.shape[2] - 1)
    )

    coords = np.array([X, Y, Z])
    sampled = map_coordinates(
        volume.astype(float),
        coords,
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    binary = sampled >= 0.5
    binary[~valid] = False
    return binary, valid


# -----------------------------------------------------------------------------
# Section autocorrelation and directional correlation lengths
# -----------------------------------------------------------------------------
def autocorrelation_field(binary, valid=None):
    if valid is None:
        valid = np.ones_like(binary, dtype=bool)

    x = binary.astype(float)
    m = valid.astype(float)
    mean = (x * m).sum() / m.sum()
    f = (x - mean) * m

    numerator = fftconvolve(f, f[::-1, ::-1], mode="full")
    pairs = fftconvolve(m, m[::-1, ::-1], mode="full")

    cov = np.zeros_like(numerator, dtype=float)
    ok = pairs > 0.5
    cov[ok] = numerator[ok] / pairs[ok]

    center = (binary.shape[0] - 1, binary.shape[1] - 1)
    c0 = cov[center]
    if not np.isfinite(c0) or c0 <= 0:
        raise RuntimeError("Invalid zero-lag covariance.")

    return cov / c0, center


def directional_acf(acf_field, center, theta_deg, max_lag=MAX_LAG):
    """
    Section coordinates are (xi1, xi2) = array axes (0, 1), so theta=0
    follows the first basis vector B[:,0].
    """
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


def correlation_length(r, c, threshold=np.exp(-1.0)):
    for i in range(1, len(c)):
        if c[i] <= threshold:
            x1, x2 = r[i - 1], r[i]
            y1, y2 = c[i - 1], c[i]
            if np.isclose(y1, y2):
                return float(x2)
            return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))
    return np.nan


def measure_section(binary, valid, B):
    acf_field, center = autocorrelation_field(binary, valid)
    theta = np.arange(0.0, 180.0, THETA_STEP_DEG, dtype=float)
    ell = np.empty_like(theta)

    for i, th in enumerate(theta):
        r, c = directional_acf(acf_field, center, th, MAX_LAG)
        ell[i] = correlation_length(r, c)

    return {
        "B": np.asarray(B, dtype=float),
        "theta": theta,
        "ell": ell,
        "binary": binary,
        "valid": valid,
    }


# -----------------------------------------------------------------------------
# 3D inverse problem
# -----------------------------------------------------------------------------
def design_rows_for_section(B, theta_deg):
    th = np.deg2rad(theta_deg)
    u = np.vstack([np.cos(th), np.sin(th)])  # 2 x N
    v = B @ u                               # 3 x N
    vx, vy, vz = v
    return np.column_stack([
        vx * vx,
        vy * vy,
        vz * vz,
        2.0 * vx * vy,
        2.0 * vx * vz,
        2.0 * vy * vz,
    ])


def assemble_inverse_problem(measurements):
    A_blocks = []
    y_blocks = []

    for m in measurements:
        good = np.isfinite(m["ell"]) & (m["ell"] > 0)
        th = m["theta"][good]
        ell = m["ell"][good]
        A_blocks.append(design_rows_for_section(m["B"], th))
        y_blocks.append(1.0 / (ell * ell))

    A = np.vstack(A_blocks)
    y = np.concatenate(y_blocks)
    return A, y


def qvec_to_matrix(q):
    return np.array([
        [q[0], q[3], q[4]],
        [q[3], q[1], q[5]],
        [q[4], q[5], q[2]],
    ], dtype=float)


def fit_q3d(measurements):
    A, y = assemble_inverse_problem(measurements)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    if rank < 6:
        return None, A, y, rank, np.inf

    q, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = qvec_to_matrix(q)

    # Project to SPD before determinant normalization.
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-10, 1e-7 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = V @ np.diag(w) @ V.T

    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1])
    return Q, A, y, rank, cond


def tensor_shape_error(Q_hat, Q_true):
    Qh = normalize_det_spd(Q_hat)
    Qt = normalize_det_spd(Q_true)
    return float(np.linalg.norm(Qh - Qt, ord="fro") / np.linalg.norm(Qt, ord="fro"))


def singular_spectrum_from_bases(bases):
    theta = np.arange(0.0, 180.0, THETA_STEP_DEG, dtype=float)
    A = np.vstack([design_rows_for_section(B, theta) for B in bases])
    s = np.linalg.svd(A, compute_uv=False)
    s = s / s[0]
    # Always return six values for plotting.
    if len(s) < 6:
        s = np.pad(s, (0, 6 - len(s)), constant_values=0.0)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    return s[:6], rank


# -----------------------------------------------------------------------------
# Random section pool and subset benchmark
# -----------------------------------------------------------------------------
def random_unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def build_section_pool(volume, rng, n_sections=N_SECTION_POOL):
    center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    pool = []

    attempts = 0
    while len(pool) < n_sections and attempts < 5 * n_sections:
        attempts += 1
        normal = random_unit_vector(rng)
        B = basis_from_normal(normal)

        # Small offset along the section normal.  SECTION_N is chosen so the
        # square remains safely inside the cube for these offsets.
        offset = rng.uniform(-4.0, 4.0)
        origin = center + offset * normal
        binary, valid = extract_oriented_section(volume, B, origin=origin, order=1)

        if valid.mean() < 0.995:
            continue

        m = measure_section(binary, valid, B)
        if np.sum(np.isfinite(m["ell"])) < 0.9 * len(m["ell"]):
            continue

        pool.append(m)

    if len(pool) < n_sections:
        raise RuntimeError("Could not build the requested random section pool.")
    return pool


def subset_benchmark(pool, Q_true, rng):
    records = []
    n_pool = len(pool)

    for nsec in SECTION_COUNTS:
        accepted = 0
        attempts = 0
        while accepted < N_SUBSETS_PER_COUNT and attempts < 20 * N_SUBSETS_PER_COUNT:
            attempts += 1
            idx = rng.choice(n_pool, size=nsec, replace=False)
            subset = [pool[i] for i in idx]
            Qhat, A, y, rank, cond = fit_q3d(subset)
            if Qhat is None or rank < 6 or not np.isfinite(cond):
                continue

            err = tensor_shape_error(Qhat, Q_true)
            records.append((nsec, cond, err))
            accepted += 1

        if accepted < N_SUBSETS_PER_COUNT:
            raise RuntimeError(f"Insufficient full-rank subsets for n={nsec}.")

    return np.asarray(records, dtype=float)


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
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
    Q = normalize_det_spd(Q)
    ell, V = principal_lengths_and_axes(Q)

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    U, Vv = np.meshgrid(u, v)

    local = np.stack([
        ell[0] * np.cos(U) * np.sin(Vv),
        ell[1] * np.sin(U) * np.sin(Vv),
        ell[2] * np.cos(Vv),
    ], axis=0)

    xyz = np.einsum("ij,jkl->ikl", V, local)
    return xyz[0], xyz[1], xyz[2], ell, V


def setup_clean_3d(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("x", labelpad=-6, fontweight="bold")
    ax.set_ylabel("y", labelpad=-6, fontweight="bold")
    ax.set_zlabel("z", labelpad=-4, fontweight="bold")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Generating controlled 3D porous medium...")
    volume, Q_true, R_true = generate_binary_medium_3d()

    # Three mutually orthogonal laboratory sections.  Their joint directional
    # design has rank six even though the anisotropy principal frame is rotated.
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    B_xy = np.column_stack([ex, ey])
    B_xz = np.column_stack([ex, ez])
    B_yz = np.column_stack([ey, ez])
    rep_bases = [B_xy, B_xz, B_yz]
    rep_names = ["XY", "XZ", "YZ"]

    print("Measuring representative orthogonal sections...")
    rep_measurements = []
    for B in rep_bases:
        img, valid = extract_oriented_section(volume, B, order=1)
        rep_measurements.append(measure_section(img, valid, B))

    Q_hat, A_rep, y_rep, rank_rep, cond_rep = fit_q3d(rep_measurements)
    if Q_hat is None:
        raise RuntimeError("Representative 3-section reconstruction is rank deficient.")

    Q_true_n = normalize_det_spd(Q_true)
    Q_hat_n = normalize_det_spd(Q_hat)
    E_rep = tensor_shape_error(Q_hat, Q_true)

    ell_true, axes_true = principal_lengths_and_axes(Q_true_n)
    ell_hat, axes_hat = principal_lengths_and_axes(Q_hat_n)
    axis_errors = np.array([
        axial_angle_error_deg(axes_hat[:, i], axes_true[:, i])
        for i in range(3)
    ])

    print("Building random oriented-section pool...")
    rng = np.random.default_rng(SEED + 1000)
    pool = build_section_pool(volume, rng)

    print("Running sparse-section subset benchmark...")
    bench = subset_benchmark(pool, Q_true, rng)
    np.savetxt(
        CSV_PATH,
        bench,
        delimiter=",",
        header="n_sections,condition_number,tensor_shape_error",
        comments="",
    )

    # Observability spectra for 1, 2, and 3 orthogonal sections.
    spectra = []
    ranks = []
    for nsec in (1, 2, 3):
        s, rank = singular_spectrum_from_bases(rep_bases[:nsec])
        spectra.append(s)
        ranks.append(rank)

    # Summary values for benchmark.
    errors_by_n = {
        n: bench[bench[:, 0] == n, 2]
        for n in SECTION_COUNTS
    }
    triplets = bench[bench[:, 0] == 3]
    rho, rho_p = spearmanr(np.log10(triplets[:, 1]), triplets[:, 2])

    print("Representative recovery:")
    print(f"  rank = {rank_rep}")
    print(f"  condition number = {cond_rep:.3f}")
    print(f"  determinant-normalized tensor error = {100*E_rep:.2f}%")
    print(f"  principal-axis errors = {axis_errors.round(2)} deg")
    print("Median tensor error by section count:")
    for n in SECTION_COUNTS:
        print(f"  n={n}: {100*np.median(errors_by_n[n]):.2f}%")
    print(f"Triplet conditioning correlation: Spearman rho={rho:.3f}")

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.34,
        height_ratios=[1.0, 1.0],
    )

    # (a) Representative sections -- nested 1x3 layout.
    sub = gs[0, 0].subgridspec(1, 3, wspace=0.06)
    axes_a = []
    for i, (m, name) in enumerate(zip(rep_measurements, rep_names)):
        ax = fig.add_subplot(sub[0, i])
        shown = np.where(m["valid"], m["binary"].astype(float), np.nan)
        ax.imshow(shown.T, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=9.2, pad=3, fontweight="bold")
        axes_a.append(ax)
    axes_a[1].text(
        0.5, 1.16,
        "Three complementary sections",
        transform=axes_a[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11.2,
        fontweight="bold",
    )
    panel_label(axes_a[0], "(a)", x=-0.05, y=1.08)

    # (b) Singular-value spectra / observability.
    ax_b = fig.add_subplot(gs[0, 1])
    x_sv = np.arange(1, 7)
    markers = ["o", "s", "^"]
    for i, nsec in enumerate((1, 2, 3)):
        s = np.maximum(spectra[i], 1e-6)
        ax_b.plot(
            x_sv, s,
            marker=markers[i],
            label=f"{nsec} section{'s' if nsec > 1 else ''} (rank {ranks[i]})",
        )
    ax_b.set_yscale("log")
    ax_b.set_xticks(x_sv)
    ax_b.set_ylim(5e-5, 1.35)
    ax_b.set_xlabel("Singular-value index", fontweight="bold")
    ax_b.set_ylabel(r"Normalized $\sigma_i$", fontweight="bold")
    ax_b.set_title("Observability of the 3D inversion", pad=8, fontweight="bold")
    ax_b.legend(frameon=False, loc="lower left")
    panel_label(ax_b, "(b)")

    # (c) Ground-truth and recovered determinant-normalized ellipsoids.
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")
    XT, YT, ZT, _, _ = ellipsoid_mesh_from_Q(Q_true_n)
    XH, YH, ZH, _, _ = ellipsoid_mesh_from_Q(Q_hat_n)
    ax_c.plot_wireframe(
        XT, YT, ZT,
        rstride=2, cstride=3,
        linewidth=0.85,
        color="black",
        alpha=0.95,
    )
    ax_c.plot_wireframe(
        XH, YH, ZH,
        rstride=3, cstride=4,
        linewidth=0.75,
        color="0.55",
        alpha=0.95,
    )
    lim = 1.22 * max(np.max(np.abs(XT)), np.max(np.abs(YT)), np.max(np.abs(ZT)),
                     np.max(np.abs(XH)), np.max(np.abs(YH)), np.max(np.abs(ZH)))
    ax_c.set_xlim(-lim, lim)
    ax_c.set_ylim(-lim, lim)
    ax_c.set_zlim(-lim, lim)
    ax_c.set_box_aspect((1, 1, 1))
    ax_c.view_init(elev=22, azim=-52)
    setup_clean_3d(ax_c)
    ax_c.set_title("Recovered 3D tensor shape", pad=6, fontweight="bold")
    proxies = [
        Line2D([0], [0], color="black", lw=1.5, label="Ground truth"),
        Line2D([0], [0], color="0.55", lw=1.5, label="Recovered"),
    ]
    ax_c.legend(handles=proxies, frameon=False, loc="lower left", bbox_to_anchor=(0.00, 0.00))
    ax_c.text2D(
        0.97, 0.91,
        f"$E_Q$ = {100*E_rep:.1f}%",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    panel_label(ax_c, "(c)")

    # (d) Principal correlation lengths and orientation errors.
    ax_d = fig.add_subplot(gs[1, 0])
    idx = np.arange(1, 4)
    ax_d.plot(idx, ell_true, marker="o", label="Ground truth")
    ax_d.plot(idx, ell_hat, marker="s", label="Recovered")
    ax_d.set_xticks(idx, [r"$\ell_1$", r"$\ell_2$", r"$\ell_3$"])
    ax_d.set_xlabel("Principal direction", fontweight="bold")
    ax_d.set_ylabel("Normalized correlation length", fontweight="bold")
    ax_d.set_title("Principal scales and directions", pad=8, fontweight="bold")
    ax_d.legend(frameon=False, loc="upper right")
    ax_d.text(
        0.04, 0.06,
        "Axis errors: " + ", ".join(f"{v:.1f}°" for v in axis_errors),
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
    )
    panel_label(ax_d, "(d)")

    # (e) Tensor error versus number of sections.
    ax_e = fig.add_subplot(gs[1, 1])
    data = [100.0 * errors_by_n[n] for n in SECTION_COUNTS]
    bp = ax_e.boxplot(
        data,
        positions=np.arange(len(SECTION_COUNTS)) + 1,
        widths=0.58,
        showfliers=False,
        patch_artist=False,
        medianprops=dict(linewidth=1.8, color="black"),
        boxprops=dict(linewidth=1.1, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )
    ax_e.set_xticks(np.arange(len(SECTION_COUNTS)) + 1, [str(n) for n in SECTION_COUNTS])
    ax_e.set_xlabel("Number of oriented sections", fontweight="bold")
    ax_e.set_ylabel(r"Tensor shape error $E_Q$ (%)", fontweight="bold")
    ax_e.set_title("Redundancy improves reconstruction", pad=8, fontweight="bold")
    panel_label(ax_e, "(e)")

    # (f) Conditioning versus triplet reconstruction error.
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.scatter(
        triplets[:, 1],
        100.0 * triplets[:, 2],
        s=22,
        alpha=0.72,
        edgecolors="none",
    )
    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    positive_err = 100.0 * triplets[:, 2]
    ymin = max(0.5, 0.8 * np.nanmin(positive_err[positive_err > 0]))
    ymax = 1.25 * np.nanmax(positive_err)
    ax_f.set_ylim(ymin, ymax)
    ax_f.set_xlabel("Design-matrix condition number", fontweight="bold")
    ax_f.set_ylabel(r"Tensor shape error $E_Q$ (%)", fontweight="bold")
    ax_f.set_title("Geometry controls error amplification", pad=8, fontweight="bold")
    ax_f.text(
        0.97, 0.95,
        rf"Spearman $\rho$ = {rho:.2f}",
        transform=ax_f.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    panel_label(ax_f, "(f)")

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.96)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Figure 4 generated successfully.")
    print(f"PNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()