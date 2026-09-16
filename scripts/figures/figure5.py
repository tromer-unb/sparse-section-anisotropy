#!/usr/bin/env python3
"""
High-quality Figure 5 for the paper.

Scientific message
------------------
The section-based 3D pore-correlation tensor reconstruction is robust across
porosity, anisotropy magnitude, random 3D orientation, finite section size,
directional angular sampling, and oblique-section interpolation.  This is the
systematic synthetic benchmark immediately preceding validation on real rocks.

Generates
---------
  - figure5_highquality.png
  - figure5_highquality.svg
  - figure5_robustness_results.csv

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

This script is fully standalone.  It does not depend on figure1.py--figure4.py,
PoreSpy, TACC data, or any project module.

Runtime
-------
The default benchmark is intentionally more demanding than Figures 1--4.  On a
modern desktop it should typically take a few minutes.  For quick layout tests,
set QUICK_MODE = True.  For manuscript-quality statistics, increase
N_MAIN_REALIZATIONS and N_SENS_REALIZATIONS before the final submission.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


# -----------------------------------------------------------------------------
# Settings -- matched to Figures 1--4
# -----------------------------------------------------------------------------
SEED = 503
DPI = 800

# Main synthetic benchmark
N_MAIN = 128
SECTION_N_MAIN = 128
POROSITIES = (0.15, 0.25, 0.35)
ANISOTROPY_RATIOS = (1.0, 1.5, 2.0, 3.0, 4.0)
BASE_SCALE = 4.5
N_MAIN_REALIZATIONS = 12

# Sensitivity benchmark (one common controlled case)
N_SENS = 160
SENS_POROSITY = 0.25
SENS_ANISOTROPY = 3.0
N_SENS_REALIZATIONS = 10
FOV_SIZES = (64, 80, 96, 112, 128, 144)
ANGLE_STEPS_DEG = (2.5, 5.0, 10.0, 15.0, 30.0)
INTERP_SECTION_N = 104

# Directional-correlation settings
MAX_LAG = 32
PRIMARY_THETA_STEP_DEG = 5.0

# Set True only for a rapid visual check of the script.
QUICK_MODE = False

# Set False after the first run if you only want to edit the figure layout.
# The script will then reload the CSV instead of recomputing all volumes.
RECOMPUTE = True

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "figure5_highquality.png"
SVG_PATH = OUTDIR / "figure5_highquality.svg"
CSV_PATH = OUTDIR / "figure5_robustness_results.csv"

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
    "svg.fonttype": "none",  # editable text in Inkscape
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Linear algebra and tensor helpers
# -----------------------------------------------------------------------------
def random_rotation_matrix(rng):
    """Uniform random rotation in SO(3), generated from a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    # Quaternion (qx, qy, qz, qw) -> active rotation matrix.
    R = np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx*qx + qy*qy)],
    ], dtype=float)
    return R


def scales_from_anisotropy(A, base=BASE_SCALE):
    """
    Principal correlation scales with constant geometric mean.
    Lmax/Lmin = A and Lmax*Lmid*Lmin = base^3.
    """
    s = np.sqrt(float(A))
    return base * np.array([s, 1.0, 1.0 / s], dtype=float)


def normalize_det_spd(Q):
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-8 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = V @ np.diag(w) @ V.T
    return Q / np.linalg.det(Q) ** (1.0 / Q.shape[0])


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


def tensor_shape_error(Q_hat, Q_true):
    Qh = normalize_det_spd(Q_hat)
    Qt = normalize_det_spd(Q_true)
    return float(np.linalg.norm(Qh - Qt, ord="fro") / np.linalg.norm(Qt, ord="fro"))


# -----------------------------------------------------------------------------
# Controlled stationary anisotropic Gaussian random field
# -----------------------------------------------------------------------------
def generate_gaussian_field_3d(n, scales, R, seed):
    """Generate one continuous stationary anisotropic Gaussian field."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n, n))
    F = np.fft.rfftn(noise)

    kx = (2.0 * np.pi * np.fft.fftfreq(n))[:, None, None]
    ky = (2.0 * np.pi * np.fft.fftfreq(n))[None, :, None]
    kz = (2.0 * np.pi * np.fft.rfftfreq(n))[None, None, :]

    # Principal-frame wave vectors: k_p = R^T k.
    kp0 = R[0, 0] * kx + R[1, 0] * ky + R[2, 0] * kz
    kp1 = R[0, 1] * kx + R[1, 1] * ky + R[2, 1] * kz
    kp2 = R[0, 2] * kx + R[1, 2] * ky + R[2, 2] * kz

    L0, L1, L2 = scales
    H = np.exp(-0.5 * ((L0 * kp0) ** 2 + (L1 * kp1) ** 2 + (L2 * kp2) ** 2))
    field = np.fft.irfftn(F * H, s=(n, n, n), axes=(0, 1, 2)).real
    return field


def threshold_field(field, porosity):
    threshold = np.quantile(field, 1.0 - float(porosity))
    return field >= threshold


def ground_truth_tensor(scales, R):
    Q = R @ np.diag(1.0 / (np.asarray(scales) ** 2)) @ R.T
    return normalize_det_spd(Q)


# -----------------------------------------------------------------------------
# Oriented section extraction
# -----------------------------------------------------------------------------
def basis_from_normal(normal):
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    if abs(n[2]) < 0.85:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    b1 = np.cross(a, n)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1)
    b2 /= np.linalg.norm(b2)
    return np.column_stack([b1, b2])


def orthogonal_bases():
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    return [
        np.column_stack([ex, ey]),  # XY
        np.column_stack([ex, ez]),  # XZ
        np.column_stack([ey, ez]),  # YZ
    ]


def oblique_bases():
    """Three fixed, complementary, well-conditioned oblique planes."""
    normals = [
        (1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (2.0, 1.0, -1.0),
        (-1.0, 2.0, 1.0),
    ]
    return [basis_from_normal(n) for n in normals]


def extract_oriented_section(volume, B, size, origin=None, order=1):
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

    sampled = map_coordinates(
        volume.astype(float),
        np.array([X, Y, Z]),
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    binary = sampled >= 0.5
    binary[~valid] = False
    return binary, valid


# -----------------------------------------------------------------------------
# Directional correlation measurements
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


def directional_acf(acf_field, center, theta_deg, max_lag):
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


def measure_section(binary, valid, B, theta_step_deg=PRIMARY_THETA_STEP_DEG, max_lag=MAX_LAG):
    acf_field, center = autocorrelation_field(binary, valid)
    theta = np.arange(0.0, 180.0, float(theta_step_deg), dtype=float)
    ell = np.empty_like(theta)
    max_lag = int(min(max_lag, max(6, binary.shape[0] // 4)))

    for i, th in enumerate(theta):
        r, c = directional_acf(acf_field, center, th, max_lag)
        ell[i] = correlation_length(r, c)

    return {"B": np.asarray(B, dtype=float), "theta": theta, "ell": ell}


# -----------------------------------------------------------------------------
# Direct 3D tensor inversion
# -----------------------------------------------------------------------------
def design_rows_for_section(B, theta_deg):
    th = np.deg2rad(theta_deg)
    u = np.vstack([np.cos(th), np.sin(th)])
    v = B @ u
    vx, vy, vz = v
    return np.column_stack([
        vx * vx,
        vy * vy,
        vz * vz,
        2.0 * vx * vy,
        2.0 * vx * vz,
        2.0 * vy * vz,
    ])


def qvec_to_matrix(q):
    return np.array([
        [q[0], q[3], q[4]],
        [q[3], q[1], q[5]],
        [q[4], q[5], q[2]],
    ], dtype=float)


def fit_q3d(measurements):
    A_blocks, y_blocks = [], []
    for m in measurements:
        good = np.isfinite(m["ell"]) & (m["ell"] > 0)
        if np.sum(good) < 6:
            return None, np.inf, 0
        A_blocks.append(design_rows_for_section(m["B"], m["theta"][good]))
        y_blocks.append(1.0 / (m["ell"][good] ** 2))

    A = np.vstack(A_blocks)
    y = np.concatenate(y_blocks)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    if rank < 6:
        return None, np.inf, rank

    q, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = qvec_to_matrix(q)

    # SPD projection for finite-sample noise.
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-7 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = V @ np.diag(w) @ V.T

    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1])
    return Q, cond, rank


def reconstruct_from_volume(volume, Q_true, bases, section_size,
                            theta_step_deg=PRIMARY_THETA_STEP_DEG,
                            interpolation_order=1):
    measurements = []
    for B in bases:
        img, valid = extract_oriented_section(
            volume, B, size=section_size, order=interpolation_order
        )
        measurements.append(
            measure_section(
                img, valid, B,
                theta_step_deg=theta_step_deg,
                max_lag=MAX_LAG,
            )
        )

    Q_hat, cond, rank = fit_q3d(measurements)
    if Q_hat is None:
        return np.nan, np.nan, np.nan, cond

    A_hat = anisotropy_ratio(Q_hat)
    E_Q = tensor_shape_error(Q_hat, Q_true)

    A_true = anisotropy_ratio(Q_true)
    if A_true <= 1.05:
        angle_error = np.nan
    else:
        _, axes_t = principal_lengths_and_axes(Q_true)
        _, axes_h = principal_lengths_and_axes(Q_hat)
        angle_error = axial_angle_error_deg(axes_t[:, 0], axes_h[:, 0])

    return A_hat, E_Q, angle_error, cond


# -----------------------------------------------------------------------------
# Benchmark execution and CSV I/O
# -----------------------------------------------------------------------------
CSV_FIELDS = [
    "experiment",
    "seed",
    "porosity",
    "anisotropy_true",
    "section_size",
    "theta_step_deg",
    "interpolation_order",
    "anisotropy_recovered",
    "tensor_shape_error",
    "orientation_error_deg",
    "condition_number",
]


def add_record(records, experiment, seed, porosity, anisotropy_true,
               section_size, theta_step_deg, interpolation_order,
               A_hat, E_Q, angle_error, cond):
    records.append({
        "experiment": experiment,
        "seed": int(seed),
        "porosity": float(porosity),
        "anisotropy_true": float(anisotropy_true),
        "section_size": int(section_size),
        "theta_step_deg": float(theta_step_deg),
        "interpolation_order": int(interpolation_order),
        "anisotropy_recovered": float(A_hat),
        "tensor_shape_error": float(E_Q),
        "orientation_error_deg": float(angle_error),
        "condition_number": float(cond),
    })


def write_records(records):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_records():
    records = []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {"experiment": row["experiment"]}
            for k in CSV_FIELDS[1:]:
                value = float(row[k])
                if k in ("seed", "section_size", "interpolation_order"):
                    value = int(round(value))
                rec[k] = value
            records.append(rec)
    return records


def run_benchmarks():
    records = []
    n_main_reps = 4 if QUICK_MODE else N_MAIN_REALIZATIONS
    n_sens_reps = 4 if QUICK_MODE else N_SENS_REALIZATIONS
    ratios = ANISOTROPY_RATIOS if not QUICK_MODE else (1.0, 2.0, 4.0)

    # ----- Main porosity x anisotropy benchmark -----
    print("Running main porosity x anisotropy benchmark...")
    bases_main = orthogonal_bases()
    for ia, Atrue in enumerate(ratios):
        scales = scales_from_anisotropy(Atrue)
        for rep in range(n_main_reps):
            seed_field = SEED + 10000 * ia + rep
            rng_rot = np.random.default_rng(SEED + 500000 + 10000 * ia + rep)
            R = random_rotation_matrix(rng_rot)
            field = generate_gaussian_field_3d(N_MAIN, scales, R, seed_field)
            Q_true = ground_truth_tensor(scales, R)

            for phi in POROSITIES:
                volume = threshold_field(field, phi)
                A_hat, E_Q, angle_error, cond = reconstruct_from_volume(
                    volume,
                    Q_true,
                    bases_main,
                    section_size=SECTION_N_MAIN,
                    theta_step_deg=PRIMARY_THETA_STEP_DEG,
                    interpolation_order=1,
                )
                add_record(
                    records, "main", seed_field, phi, Atrue,
                    SECTION_N_MAIN, PRIMARY_THETA_STEP_DEG, 1,
                    A_hat, E_Q, angle_error, cond,
                )

    # ----- Sensitivity benchmarks share the same underlying volumes -----
    print("Running field-of-view, angular-sampling, and interpolation sensitivities...")
    scales = scales_from_anisotropy(SENS_ANISOTROPY)
    bases_orth = orthogonal_bases()
    bases_obl = oblique_bases()

    for rep in range(n_sens_reps):
        seed_field = SEED + 900000 + rep
        rng_rot = np.random.default_rng(SEED + 950000 + rep)
        R = random_rotation_matrix(rng_rot)
        field = generate_gaussian_field_3d(N_SENS, scales, R, seed_field)
        volume = threshold_field(field, SENS_POROSITY)
        Q_true = ground_truth_tensor(scales, R)

        # Field-of-view sensitivity using centered orthogonal sections.
        fov_sizes = FOV_SIZES if not QUICK_MODE else (64, 96, 128)
        for size in fov_sizes:
            A_hat, E_Q, angle_error, cond = reconstruct_from_volume(
                volume, Q_true, bases_orth,
                section_size=size,
                theta_step_deg=PRIMARY_THETA_STEP_DEG,
                interpolation_order=1,
            )
            add_record(
                records, "fov", seed_field, SENS_POROSITY, SENS_ANISOTROPY,
                size, PRIMARY_THETA_STEP_DEG, 1,
                A_hat, E_Q, angle_error, cond,
            )

        # Angular sampling sensitivity at a fixed representative section size.
        angle_steps = ANGLE_STEPS_DEG if not QUICK_MODE else (5.0, 15.0, 30.0)
        for step in angle_steps:
            A_hat, E_Q, angle_error, cond = reconstruct_from_volume(
                volume, Q_true, bases_orth,
                section_size=128,
                theta_step_deg=step,
                interpolation_order=1,
            )
            add_record(
                records, "angular", seed_field, SENS_POROSITY, SENS_ANISOTROPY,
                128, step, 1,
                A_hat, E_Q, angle_error, cond,
            )

        # Oblique-section interpolation sensitivity.
        for order in (0, 1):
            A_hat, E_Q, angle_error, cond = reconstruct_from_volume(
                volume, Q_true, bases_obl,
                section_size=INTERP_SECTION_N,
                theta_step_deg=PRIMARY_THETA_STEP_DEG,
                interpolation_order=order,
            )
            add_record(
                records, "interpolation", seed_field, SENS_POROSITY, SENS_ANISOTROPY,
                INTERP_SECTION_N, PRIMARY_THETA_STEP_DEG, order,
                A_hat, E_Q, angle_error, cond,
            )

    write_records(records)
    return records


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
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


def select(records, experiment=None, porosity=None, anisotropy=None,
           section_size=None, theta_step=None, interpolation_order=None):
    out = []
    for r in records:
        if experiment is not None and r["experiment"] != experiment:
            continue
        if porosity is not None and not np.isclose(r["porosity"], porosity):
            continue
        if anisotropy is not None and not np.isclose(r["anisotropy_true"], anisotropy):
            continue
        if section_size is not None and r["section_size"] != section_size:
            continue
        if theta_step is not None and not np.isclose(r["theta_step_deg"], theta_step):
            continue
        if interpolation_order is not None and r["interpolation_order"] != interpolation_order:
            continue
        out.append(r)
    return out


def values(rows, field):
    arr = np.array([r[field] for r in rows], dtype=float)
    return arr[np.isfinite(arr)]


def med_iqr(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    q1, med, q3 = np.quantile(arr, [0.25, 0.50, 0.75])
    return med, med - q1, q3 - med


def errorbar_series(ax, xvals, groups, field, label=None, color=None,
                    marker="o", scale=1.0):
    meds, lo, hi = [], [], []
    for rows in groups:
        med, l, h = med_iqr(values(rows, field) * scale)
        meds.append(med); lo.append(l); hi.append(h)
    ax.errorbar(
        xvals, meds, yerr=np.vstack([lo, hi]),
        marker=marker, capsize=3.0, label=label, color=color,
    )
    return np.asarray(meds)


# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------
def main():
    if RECOMPUTE or not CSV_PATH.exists():
        records = run_benchmarks()
    else:
        print(f"Loading existing benchmark results: {CSV_PATH}")
        records = read_records()

    # Use the available anisotropy ratios in case QUICK_MODE generated the CSV.
    main_rows = select(records, experiment="main")
    A_values = sorted(set(r["anisotropy_true"] for r in main_rows))
    phi_values = sorted(set(r["porosity"] for r in main_rows))

    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    phi_colors = {phi: colors[i % len(colors)] for i, phi in enumerate(phi_values)}

    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

    # ------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, 0])
    all_min = min(A_values)
    all_max = max(A_values)
    ax.plot([all_min, all_max], [all_min, all_max], linestyle="--", color="black", linewidth=1.1, label="Ideal")
    for phi in phi_values:
        groups = [select(records, experiment="main", porosity=phi, anisotropy=A) for A in A_values]
        errorbar_series(
            ax, A_values, groups, "anisotropy_recovered",
            label=rf"$\phi={phi:.2f}$", color=phi_colors[phi], marker="o"
        )
    ax.set_xlabel("Prescribed anisotropy ratio")
    ax.set_ylabel("Recovered anisotropy ratio")
    ax.set_title("Anisotropy magnitude recovery")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "(a)")

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    for phi in phi_values:
        groups = [select(records, experiment="main", porosity=phi, anisotropy=A) for A in A_values]
        errorbar_series(
            ax, A_values, groups, "tensor_shape_error",
            label=rf"$\phi={phi:.2f}$", color=phi_colors[phi], marker="o", scale=100.0
        )
    ax.set_xlabel("Prescribed anisotropy ratio")
    ax.set_ylabel("Tensor-shape error (%)")
    ax.set_title("Tensor recovery across porosity")
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    A_orient = [A for A in A_values if A > 1.05]
    for phi in phi_values:
        groups = [select(records, experiment="main", porosity=phi, anisotropy=A) for A in A_orient]
        errorbar_series(
            ax, A_orient, groups, "orientation_error_deg",
            label=rf"$\phi={phi:.2f}$", color=phi_colors[phi], marker="o"
        )
    ax.set_xlabel("Prescribed anisotropy ratio")
    ax.set_ylabel("Major-axis error (degrees)")
    ax.set_title("Principal-direction recovery")
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0])
    fov_values = sorted(set(r["section_size"] for r in select(records, experiment="fov")))
    fov_groups = [select(records, experiment="fov", section_size=s) for s in fov_values]
    errorbar_series(ax, fov_values, fov_groups, "tensor_shape_error", marker="o", scale=100.0)
    ax.set_xlabel("Section width (pixels)")
    ax.set_ylabel("Tensor-shape error (%)")
    ax.set_title("Finite-section sensitivity")
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    step_values = sorted(set(r["theta_step_deg"] for r in select(records, experiment="angular")))
    step_groups = [select(records, experiment="angular", theta_step=s) for s in step_values]
    errorbar_series(ax, step_values, step_groups, "tensor_shape_error", marker="o", scale=100.0)
    ax.set_xlabel("Directional sampling step (degrees)")
    ax.set_ylabel("Tensor-shape error (%)")
    ax.set_title("Angular-sampling sensitivity")
    ax.set_xticks(step_values)
    ax.set_xticklabels([f"{v:g}" for v in step_values], rotation=28, ha="right")
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    rows_nn = select(records, experiment="interpolation", interpolation_order=0)
    rows_lin = select(records, experiment="interpolation", interpolation_order=1)
    data_nn = values(rows_nn, "tensor_shape_error") * 100.0
    data_lin = values(rows_lin, "tensor_shape_error") * 100.0

    bp = ax.boxplot(
        [data_nn, data_lin],
        tick_labels=["Nearest\nneighbor", "Trilinear\n+ threshold"],
        widths=0.55,
        showfliers=False,
        patch_artist=False,
    )

    # Paired realizations: connect the same synthetic volume across methods.
    by_seed_nn = {r["seed"]: 100.0 * r["tensor_shape_error"] for r in rows_nn if np.isfinite(r["tensor_shape_error"])}
    by_seed_lin = {r["seed"]: 100.0 * r["tensor_shape_error"] for r in rows_lin if np.isfinite(r["tensor_shape_error"])}
    common = sorted(set(by_seed_nn).intersection(by_seed_lin))
    for j, seed in enumerate(common):
        y0 = by_seed_nn[seed]
        y1 = by_seed_lin[seed]
        ax.plot([1, 2], [y0, y1], linewidth=0.8, alpha=0.38)
        jitter = ((j % 5) - 2) * 0.018
        ax.scatter([1 + jitter, 2 + jitter], [y0, y1], s=13, zorder=3)

    ax.set_ylabel("Tensor-shape error (%)")
    ax.set_title("Oblique-section resampling")
    panel_label(ax, "(f)")

    # Global formatting.
    for ax in fig.axes:
        if hasattr(ax, "grid"):
            ax.grid(False)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.965)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------ Summary
    main_err = values(main_rows, "tensor_shape_error") * 100.0
    main_Aerr = np.abs(values(main_rows, "anisotropy_recovered") - np.array([
        r["anisotropy_true"] for r in main_rows if np.isfinite(r["anisotropy_recovered"])
    ]))
    orient = values(main_rows, "orientation_error_deg")

    print("\nFigure 5 generated successfully.")
    print(f"PNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {CSV_PATH}")
    if len(main_err):
        print(f"Main benchmark median tensor-shape error: {np.median(main_err):.2f}%")
    if len(orient):
        print(f"Main benchmark median major-axis error: {np.median(orient):.2f} degrees")
    if len(data_nn) and len(data_lin):
        print(f"Oblique nearest-neighbor median error: {np.median(data_nn):.2f}%")
        print(f"Oblique trilinear median error: {np.median(data_lin):.2f}%")


if __name__ == "__main__":
    main()